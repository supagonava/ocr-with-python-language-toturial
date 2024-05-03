import base64
import io
import json
import os
import boto3
import pandas as pd

from typing import Dict, List
from botocore.config import Config
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def convert_aws_geometry_bounding_box_to_system_bbox(block: dict, page_width: int, page_height: int):
    bounding_box = block["Geometry"]["BoundingBox"]
    bbox_width = int(bounding_box["Width"] * page_width)
    bbox_height = int(bounding_box["Height"] * page_height)
    bbox_left = int(bounding_box["Left"] * page_width)
    bbox_top = int(bounding_box["Top"] * page_height)

    point1 = (bbox_left, bbox_top)
    point2 = (bbox_left + bbox_width, bbox_top + bbox_height)
    bbox = {"pt1": point1, "pt2": point2, "l": point1[0], "t": point1[1], "r": point2[0], "b": point2[1]}
    bbox_data = {"bbox": bbox, "text": block["Text"]}
    return bbox_data


def sort_words_to_pretty_text(words_with_boxes: list[dict] = [], space_height_threshold=5, space_width_threshold=4):
    botoms = list(set([t["bbox"]["b"] for t in words_with_boxes]))
    botoms = sorted(botoms)
    text_df = pd.DataFrame([{"text": t["text"], "left": t["bbox"]["l"], "right": t["bbox"]["r"], "top": t["bbox"]["t"], "bottom": t["bbox"]["b"]} for t in words_with_boxes])
    used_botoms = []
    line_message: str = ""
    for btm in botoms:
        lines: pd.DataFrame = text_df[~text_df.index.isin(used_botoms) & (text_df["bottom"] <= btm + space_height_threshold) & (text_df["bottom"] >= btm - space_height_threshold)]
        if not lines.empty:
            used_botoms.extend(lines.index)
            lines = lines.sort_values(by="left")
            prev_word = lines.iloc[0]
            line_message += prev_word["text"]
            for _, word in lines[1:].iterrows():
                join_text = ""
                if word["left"] - prev_word["right"] > space_width_threshold:
                    join_text = " "
                line_message += join_text + word["text"]
                prev_word = word
            line_message += "\n"

    return line_message


def is_inside(A, B):
    def is_point_inside_rect(point, rect):
        return rect[0][0] <= point[0] <= rect[1][0] and rect[0][1] <= point[1] <= rect[2][1]

    return all(is_point_inside_rect(b_point, A) for b_point in B)


def word_bbox_coordinates(word):
    return [
        (word["bbox"]["l"], word["bbox"]["t"]),
        (word["bbox"]["r"], word["bbox"]["t"]),
        (word["bbox"]["r"], word["bbox"]["b"]),
        (word["bbox"]["l"], word["bbox"]["b"]),
    ]


def get_rows_columns_map(table_result, blocks_map, words: List[Dict] = None, page_width: int = 0, page_height: int = 0):
    rows: Dict[int, List] = {}
    merged_cells: Dict[int, List] = {}
    scores: Dict[int, List] = {}
    polygon: Dict[int, List] = {}

    for relationship in table_result["Relationships"]:
        if relationship["Type"] in ["CHILD", "MERGED_CELL"]:
            for child_id in relationship["Ids"]:
                cell: dict = blocks_map[child_id]
                row_index = cell.get("RowIndex")
                column_index = cell.get("ColumnIndex")
                row_span = cell.get("RowSpan")
                column_span = cell.get("ColumnSpan")

                if cell["BlockType"] == "CELL":
                    if row_index not in polygon:
                        polygon[row_index] = []

                    if row_index not in rows:
                        # create new row
                        rows[row_index] = []

                    if row_index not in scores:
                        # create new row
                        scores[row_index] = []

                    # get confidence score
                    scores[row_index].append(str(cell["Confidence"]))

                    geometry: dict = cell.get("Geometry", {})
                    polygon_block: List[Dict] = geometry.get("Polygon")
                    polygon_block_calculated = [{"X": int(pgb.get("X", 0) * page_width), "Y": int(pgb.get("Y", 0) * page_height)} for pgb in polygon_block]
                    polygon[row_index].append(polygon_block_calculated)
                    words_inside = [word for word in words if is_inside(A=[tuple(plg.values()) for plg in polygon_block_calculated], B=word_bbox_coordinates(word))]

                    # get cell text
                    pretty_text = "-"
                    if words_inside:
                        pretty_text = sort_words_to_pretty_text(words_inside)

                    rows[row_index].append(pretty_text.strip())

                elif cell["BlockType"] == "MERGED_CELL":
                    if row_index not in merged_cells:
                        merged_cells[row_index] = {}
                    if column_index not in merged_cells[row_index]:
                        merged_cells[row_index][column_index] = {}

                    merged_cells[row_index][column_index].update(
                        {
                            "row_span": row_span,
                            "column_span": column_span,
                        }
                    )
    return rows, scores, merged_cells, polygon


def get_data_table(aws_analyze_data: dict, words: List[Dict] = None, page_width: int = 0, page_height: int = 0):

    response = aws_analyze_data
    # Get the text blocks
    blocks: List[Dict] = response.get("Blocks", [])
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block.get("Id")] = block
        if block.get("BlockType") == "TABLE":
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        return table_blocks

    tables: List[Dict] = []
    for index, table in enumerate(table_blocks):
        rows, scores, merged_cells, polygon = get_rows_columns_map(table, blocks_map, words=words, page_width=page_width, page_height=page_height)
        column_count = max(len(row) for row in rows.values())
        table_data = {
            "id": f"table-{index+1}",
            "rows": rows,
            "scores": scores,
            "merged_cells": merged_cells,
            "polygon": polygon,
            "row_count": len(rows.keys()),
            "column_count": column_count,
        }
        tables.append(table_data)

    return tables


def get_aws_textannotations_formatedtext(response: dict, page_width=0, page_height=0):
    # deep copy
    blocks_str = json.dumps(response["Blocks"], ensure_ascii=False)
    blocks: list[Dict] = json.loads(blocks_str)

    format_text = ""
    text_annotations = []
    text_lines = []

    for block in blocks:
        bbox_data = {"bbox": None, "text": None}
        try:
            if str(block.get("BlockType", "PAGE")).lower() != "page":
                bbox_data = convert_aws_geometry_bounding_box_to_system_bbox(block, page_width, page_height)
        except:
            pass

        if block["BlockType"] == "WORD":
            text_annotations.append(bbox_data)

        elif block["BlockType"] == "LINE":
            block.update({"bbox_data": bbox_data})
            text_lines.append(block)

    sorted_lines = sorted(text_lines, key=lambda x: (x["Geometry"]["BoundingBox"]["Top"], x["Geometry"]["BoundingBox"]["Left"]))
    for line in sorted_lines:
        format_text += "{}\n".format(line["Text"])

    return {"text_annotations": text_annotations, "format_text": format_text}


def aws_textract_image(image_data, use_extract_table=False):

    if type(image_data) == str:
        image_data = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_data))
    if image.mode != "RGB":
        image = image.convert("RGB")
        image_io = io.BytesIO()
        image.save(image_io, "JPEG")
        image_data = image_io.getvalue()

    boto3_config = Config(retries={"max_attempts": 10, "mode": "standard"})
    client = boto3.client(
        "textract",
        region_name="ap-southeast-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=boto3_config,
    )

    if not use_extract_table:
        return client.detect_document_text(Document={"Bytes": image_data})
    else:
        return client.analyze_document(Document={"Bytes": image_data}, FeatureTypes=["TABLES"])


# Without table AWS_TEXTRACT 1.50 / 1000
# With table AWS_TABLE 15 / 1000

if __name__ == "__main__":
    image_bin = open(os.path.join("images", "test-4.png"), "rb").read()
    width, height = Image.open(io.BytesIO(image_bin)).size

    # without table
    without_table_result: dict = {}
    without_table_result = aws_textract_image(image_bin, use_extract_table=False)
    with open("python/results/aws-original-textract-without-table.json", "w") as writer:
        writer.write(json.dumps(without_table_result, indent=2, ensure_ascii=False))

    without_table_result = get_aws_textannotations_formatedtext(without_table_result, page_width=width, page_height=height)
    with open("python/results/aws-textract-without-table.json", "w") as writer:
        writer.write(json.dumps(without_table_result, indent=2, ensure_ascii=False))

    # with table
    with_table_result: dict = {}
    with_table_result_raw = aws_textract_image(image_bin, use_extract_table=True)
    with open("python/results/aws-original-textract-with-table.json", "w") as writer:
        writer.write(json.dumps(with_table_result_raw, indent=2, ensure_ascii=False))

    with_table_result = get_aws_textannotations_formatedtext(with_table_result_raw, page_width=width, page_height=height)
    with_table_result.update(
        {
            "tables": get_data_table(
                aws_analyze_data=with_table_result_raw,
                words=with_table_result.get("text_annotations"),
                page_width=width,
                page_height=height,
            )
        }
    )
    with open("python/results/aws-textract-with-table.json", "w") as writer:
        writer.write(json.dumps(with_table_result, indent=2, ensure_ascii=False))
