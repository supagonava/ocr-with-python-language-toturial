import base64
import io
from typing import List

from PIL import Image
from azure_ocr import azure_extracttext

EXTRACTION_PROMPT = """You`re helpful to extract fields from document text.
You response will be in json format that fit for python json loads

<DocumentDescriptionOptional>
{document_description}
</DocumentDescriptionOptional>

Value Conditions:
- Exclude the following characters from all values: `"`, `'`, `/`, and `\`.
- Ensure all data is included; do not omit any for the sake of brevity.
- Refrain from adding comments to the JSON text response.
- For numbers, avoid using comma-separated values. For example, convert `100,000.00` to `100000.00`.
- For dates, adhere to the format 'YYYY-MM-DDTHH:mm:ss'. For example, use '1900-01-01T01:01:01'.
- If the data for a field is uncertain, assign a `null` value to it.

<ResponseFormatInstruction>
{format_instructions}
</ResponseFormatInstruction>

<TextToExtract>
{plain_text}
</TextToExtract>
"""

EXTRACTION_PROMPT_VISION = """You`re helpful to extract fields from document text.
You response will be in json format that fit for python json loads

<DocumentDescriptionOptional>
{document_description}
</DocumentDescriptionOptional>

Value Conditions:
- Exclude the following characters from all values: `"`, `'`, `/`, and `\`.
- Ensure all data is included; do not omit any for the sake of brevity.
- Refrain from adding comments to the JSON text response.
- For numbers, avoid using comma-separated values. For example, convert `100,000.00` to `100000.00`.
- For dates, adhere to the format 'YYYY-MM-DDTHH:mm:ss'. For example, use '1900-01-01T01:01:01'.
- If the data for a field is uncertain, assign a `null` value to it.

<ResponseFormatInstruction>
{format_instructions}
</ResponseFormatInstruction>
"""

import json
import os
from dotenv import load_dotenv
import requests

load_dotenv()

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE")
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")


GPT35TURBO = "gpt-35-turbo-16k"
GPT4TURBO = "gpt-4-turbo"
GPT4VISION = "gpt-4-vision"


def extract_data_from_images(
    images: List[Image.Image] = [],
    document_description: str = "None",
    format_instructions: dict = {},
    prompt: str = EXTRACTION_PROMPT_VISION,
):
    messages = [{"type": "text", "text": "Extract infomation from document image."}]
    # max 10 pages
    for idx, image in enumerate(images[:10]):
        image_binary = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(image_binary, format="JPEG", optimize=True)
        image_binary = image_binary.getvalue()
        image_base64 = base64.b64encode(image_binary).decode("utf-8")
        messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})

    url = f"https://cogopenaiscgjwddocutil1.openai.azure.com/openai/deployments/gpt-4-vision/chat/completions?api-version=2024-02-15-preview"
    payload = {
        "enhancements": {"ocr": {"enabled": True}, "grounding": {"enabled": False}},
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(
                            document_description=document_description,
                            format_instructions=json.dumps(format_instructions, ensure_ascii=False, indent=1),
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": messages,
            },
        ],
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 4096,
        "stream": False,
    }

    headers = {"api-key": OPENAI_API_KEY, "content-type": "application/json"}
    response = requests.request("POST", url, json=payload, headers=headers)
    return response.json()


def extract_data_from_plaintext(
    text: str,
    engine="gpt-35-turbo-16k",
    document_description: str = "None",
    format_instructions: dict = {},
    prompt: str = EXTRACTION_PROMPT,
):
    url = "{baseurl}/deployments/{model}/chat/completions".format(baseurl=OPENAI_API_BASE, model=engine)
    querystring = {"api-version": OPENAI_API_VERSION}
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(
                            document_description=document_description,
                            format_instructions=json.dumps(format_instructions, ensure_ascii=False, indent=1),
                            plain_text=text,
                        ),
                    }
                ],
            }
        ],
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 4096,
        "stream": False,
    }

    headers = {"api-key": OPENAI_API_KEY, "content-type": "application/json"}
    response = requests.request("POST", url, json=payload, headers=headers, params=querystring)
    return response.json()


if __name__ == "__main__":
    image = open("images/test-5.png", "rb").read()
    json_format = {
        "quotation_no": "String",
        "quotation_date": "String Parse to dateformat dd/mm/yyyy",
        "quotation_exp_date": "String Parse to dateformat dd/mm/yyyy",
    }
    ocr_result = azure_extracttext(image)
    gpt_35_result = extract_data_from_plaintext(
        text=ocr_result.get("format_text"),
        document_description=None,
        format_instructions=json_format,
    )
    print("gpt_35_result", gpt_35_result["choices"][0]["message"])
    with open("src/results/gpt_35_result.json", "w") as writer:
        writer.write(json.dumps(gpt_35_result, indent=2, ensure_ascii=False))

    gpt_4vision_result = extract_data_from_images(
        images=[Image.open(io.BytesIO(image))],
        document_description=None,
        format_instructions=json_format,
    )
    print("gpt_4vision_result", gpt_4vision_result["choices"][0]["message"])
    with open("src/results/gpt_4vision_result.json", "w") as writer:
        writer.write(json.dumps(gpt_4vision_result, indent=2, ensure_ascii=False))
