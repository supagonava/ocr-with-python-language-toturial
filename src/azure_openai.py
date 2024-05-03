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
        # "enhancements": {
        #     "ocr": {"enabled": enable_ocr_enhancement},
        #     "grounding": {"enabled": False},
        # },
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
    ocr_result = azure_extracttext(image)
    gpt_35_result = extract_data_from_plaintext(
        text=ocr_result.get("format_text"),
        document_description=None,
        format_instructions={
            "quotation_no": "String",
            "quotation_date": "String Parse to dateformat dd/mm/yyyy",
            "quotation_exp_date": "String Parse to dateformat dd/mm/yyyy",
        },
    )
    print(gpt_35_result["choices"][0]["message"])
