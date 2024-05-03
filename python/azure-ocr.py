import base64
import io
import json
import os
import requests
import requests
import base64

from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentWord, DocumentTableCell, DocumentTable, DocumentParagraph
from dotenv import load_dotenv

load_dotenv()


AZURE_VISION_ENDPOINT = os.environ["AZURE_VISION_ENDPOINT"]
AZURE_VISION_KEY = os.environ["AZURE_VISION_KEY"]
AZURE_FORMREGONIZER_ENDPOINT = os.environ["AZURE_FORMREGONIZER_ENDPOINT"]
AZURE_FORMREGONIZER_KEY = os.environ["AZURE_FORMREGONIZER_KEY"]


def is_point_inside_polygon(point, polygon):
    """Check if a point is inside a polygon using the ray casting method."""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def is_within_table(paragraph: DocumentParagraph, tables: list[DocumentTable]):
    """Check if a paragraph is within any of the tables."""
    for table in tables:
        table_polygon = [(point.x, point.y) for point in table.bounding_regions[0].polygon]
        for point in paragraph.bounding_regions[0].polygon:
            if is_point_inside_polygon((point.x, point.y), table_polygon):
                return True
    return False


def get_text_annotation(word: DocumentWord):
    point1 = list(word.polygon[0])
    point2 = list(word.polygon[2])
    bbox = {"pt1": point1, "pt2": point2, "l": point1[0], "t": point1[-1], "r": point2[0], "b": point2[-1]}
    text_annotation = {"bbox": bbox, "text": word.content}
    return text_annotation


def get_azure_textannotations_formatedtext(response_dict: dict):
    text_annotations = []
    format_text = ""

    for block in response_dict["readResult"]["blocks"]:
        for line in block["lines"]:
            format_text += "{}\n".format(line["text"])

            for word in line["words"]:
                bounding_box = word["boundingPolygon"]
                x_list = [c["x"] for c in bounding_box]
                y_list = [c["y"] for c in bounding_box]
                point1 = (min(x_list), min(y_list))
                point2 = (max(x_list), max(y_list))
                bbox = {"pt1": point1, "pt2": point2, "l": point1[0], "t": point1[-1], "r": point2[0], "b": point2[-1]}
                bbox_result = {"bbox": bbox, "text": word["text"]}
                text_annotations.append(bbox_result)
    return {
        "text_annotations": text_annotations,
        "format_text": format_text,
    }


def azure_extracttext(image_data: str | bytes, use_extract_table=False):
    tables = []
    if type(image_data) == str:
        image_data = base64.b64decode(image_data)

    document_analysis_client = DocumentAnalysisClient(
        endpoint=AZURE_FORMREGONIZER_ENDPOINT,
        credential=AzureKeyCredential(AZURE_FORMREGONIZER_KEY),
    )

    poller = document_analysis_client.begin_analyze_document("prebuilt-layout" if use_extract_table else "prebuilt-read", image_data)
    result = poller.result()
    text_annotations = []
    text_content = ""

    for page in result.pages:
        for word in page.words:
            if word.polygon:
                text_annotations.append(get_text_annotation(word))

    for i, paragraph in enumerate(result.paragraphs):
        if not is_within_table(paragraph, result.tables):
            paragraph_content = (paragraph.content).replace("\n", "").strip()
            text_content += "{}\n".format(paragraph_content)

    for table_idx, table in enumerate(result.tables):
        table_cells: list[DocumentTableCell] = table.cells
        table_dict = {
            "id": f"Table {table_idx+1}",
            "merged_cells": {},
            "polygon": {},
            "rows": {},
            "scores": {},
            "row_count": table.row_count,
            "column_count": table.column_count,
        }

        for cell in table_cells:
            if not table_dict["polygon"].get(str(cell.row_index)):
                table_dict["polygon"][str(cell.row_index)] = []

            if not table_dict["rows"].get(str(cell.row_index)):
                table_dict["rows"][str(cell.row_index)] = []

            table_dict["polygon"][str(cell.row_index)].append([list(b.polygon) for b in cell.bounding_regions])
            table_dict["rows"][str(cell.row_index)].append(cell.content)

        tables.append(table_dict)

    return {
        "text_annotations": text_annotations,
        "format_text": text_content,
        "tables": tables,
    }


# def azure_ocr_without_table(image_data: str | bytes):
#     if type(image_data) == str:
#         image_data = base64.b64decode(image_data)

#     AZURE_OCR_URL = AZURE_VISION_ENDPOINT + "/computervision/imageanalysis:analyze"
#     params = {
#         "features": "read",
#         "model-version": "latest",
#         "gender-neutral-caption": "false",
#         "api-version": "2023-10-01",
#         "detect-orientation": "true",
#     }

#     # Set Content-Type to octet-stream
#     headers = {"ocp-apim-subscription-key": AZURE_VISION_KEY, "Content-Type": "application/octet-stream"}

#     # put the byte array into your post request
#     response = requests.post(AZURE_OCR_URL, headers=headers, params=params, data=image_data)
#     if response.status_code == 200:
#         response_json: dict = response.json()
#         result_data = get_azure_textannotations_formatedtext(response_json)

#         return result_data


if __name__ == "__main__":
    image_bin = open(os.path.join("images", "test-1.png"), "rb").read()
    width, height = Image.open(io.BytesIO(image_bin)).size

    # without table
    without_table_result: dict = {}
    without_table_result = azure_extracttext(image_bin, use_extract_table=False)
    with open("python/results/azure-textract-without-table.json", "w") as writer:
        writer.write(json.dumps(without_table_result, indent=2, ensure_ascii=False))

    # with table
    with_table_result: dict = {}
    with_table_result_raw = azure_extracttext(image_bin, use_extract_table=True)
    with open("python/results/azure-textract-with-table.json", "w") as writer:
        writer.write(json.dumps(with_table_result_raw, indent=2, ensure_ascii=False))
