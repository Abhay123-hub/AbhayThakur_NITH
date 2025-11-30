# invoice_extractor.py

import json
import base64
from io import BytesIO
from typing import List, Tuple

import fitz  # PyMuPDF
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import os, json, base64
from io import BytesIO
import pdfplumber, shutil
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError
# --------------------
# OpenAI client
# --------------------
load_dotenv()
client = OpenAI()  # assumes OPENAI_API_KEY in env


# --------------------
# Pydantic models (internal)
# --------------------

class BillItem(BaseModel):
    item_name: str
    item_amount: float = 0.0
    item_rate: float = 0.0
    item_quantity: float = 0.0


class PageLineItems(BaseModel):
    page_no: str
    page_type: str = "Bill Detail"
    bill_items: List[BillItem] = Field(default_factory=list)


class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class InvoiceData(BaseModel):
    pagewise_line_items: List[PageLineItems] = Field(default_factory=list)
    total_item_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

class FinalOutput(BaseModel):
    is_success: bool
    data: InvoiceData



def validate_invoice_json(invoice_json: dict) -> FinalOutput:
    """
    Validate raw LLM JSON using Pydantic models.
    This ensures strict schema matching.
    """
    return FinalOutput(**invoice_json)




def extract_text_digital(pdf_path, page_index):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        return page.extract_text() or ""

def page_to_image_b64(pdf_path, page_index, all_images=None):
    if all_images:
        img = all_images[page_index]
        # If cached object is bytes from PyMuPDF
        if isinstance(img, bytes):
            return base64.b64encode(img).decode()

    doc = fitz.open(pdf_path)
    pix = doc[page_index].get_pixmap()
    
    # Convert to JPEG using PIL
    pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    
    return base64.b64encode(buf.getvalue()).decode()

# --------------------
# Small helpers
# --------------------

def fix_inconsistent_amounts(bill_items: List[BillItem], tol: float = 1e-2) -> List[BillItem]:
    """If rate * qty != amount by a lot, trust rate * qty."""
    for item in bill_items:
        r = item.item_rate or 0.0
        q = item.item_quantity or 0.0
        a = item.item_amount or 0.0
        if r > 0 and q > 0:
            expected = r * q
            if abs(expected - a) > tol:
                item.item_amount = round(expected, 2)
    return bill_items


def clean_items(bill_items: List[BillItem]) -> List[BillItem]:
    """Drop TOTAL rows and exact duplicates."""
    cleaned = []
    seen = set()
    for item in bill_items:
        name = (item.item_name or "").upper().strip()
        key = (name, item.item_amount, item.item_rate, item.item_quantity)
        if "TOTAL" in name:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


# --------------------
# Vision OCR
# --------------------

def ocr_page_with_vision(b64_img):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract invoice text"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]
        }]
    )
    return resp.choices[0].message.content


# --------------------
# Structuring text → JSON
# --------------------


def extract_structured_from_text(text):
    prompt = f"""
You are extracting structured medical/hospital billing data for insurance processing.
The input may be a digital PDF, scanned PDF, or handwritten invoice image.

First, determine the main bill details and the billing table content, then extract **only valid billing rows**.

### Table understanding:
Invoices are usually displayed as a table with columns similar to:
- S.No / Serial Number
- Service / Description / Item
- Rate / Unit Price
- Quantity / Qty / Units / Days
- Net Amount / Net Amt / Total Amount / Charge for that row

### Your task:
1. Treat each page as one bill document.
2. For each table row that represents a real billable charge, return ONE structured object inside `bill_items`.
3. Use only the "Net Amount / Net Amt" column for `item_amount`.
4. Use the "Rate / Unit Price" column for `item_rate`.
5. Use the "Quantity / Units / Days / Qty / Units" column for `item_quantity`.
6. IGNORE ALL of the following:
   - Section/heading rows (e.g., CONSULTATION, ROOM CHARGES, LAB CHARGES, RADIOLOGY, SURGERY).
   - Subtotal rows or calculated totals for a section (e.g., 1950.00, 4850.00, 12000.00 etc.).
   - Grand total, round off, discount, payable amount.
   - Any row whose description is only a total label (TOTAL, Totals, Grand Total, Subtotal, Net Total).
   - Duplicate or repeated rows referring to the same table entry.
   - Watermarks, logos, links, request formats, or unrelated handwritten notes.

7. If a row is a medical bill but rate or quantity is missing, set them to `0.0` (never null).
8. If a row is unrelated or invalid, do not include it in JSON.
9. Do **not merge multiple rows into one. One row = one `bill_items` object.

Output JSON in this exact format:
{{
  "is_success": True or False,
  "data": {{
    "pagewise_line_items": [
      {{
        "page_no": "integer as string",
        "page_type": "Bill Detail",
        "bill_items": [
          {{
            "item_name": "string",
            "item_amount": 0.0,
            "item_rate": 0.0 ,
            "item_quantity": 0.0
          }}
        ]
      }}
    ],
    "total_item_count": 0,
    "token_usage": {{
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    }}
  }}
}}
Only return JSON.





Rules:
- item_amount, item_rate, item_quantity must be numbers (no strings).
- If Rate or Quantity is missing on the row, use 0.0, not null.
- Do not include any row whose Description contains only words like "Total", "Totals", "Grand Total".

Only return valid JSON. No explanation.

Bill text (can include other noise, focus on the bill table):

\"\"\"{text}\"\"\"
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    
    usage = getattr(resp, "usage", None)

    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

   
    try:
        data = json.loads(raw)

        if usage is not None:
          # make sure the nested object exists
          data.setdefault("data", {})
          data["data"].setdefault("token_usage", {})

          
          data["data"]["token_usage"]["input_tokens"]  = usage.prompt_tokens
          data["data"]["token_usage"]["output_tokens"] = usage.completion_tokens
          data["data"]["token_usage"]["total_tokens"]  = usage.total_tokens

        # Return string again so your outer code still does json.loads(...)
        return json.dumps(data)

    except json.JSONDecodeError:
        # If something weird happens, just return raw (so you can debug)
        return raw

# --------------------
# Main PDF pipeline
# --------------------


def process_pdf_bytes(pdf_path) -> dict:
    """
    Core function used by FastAPI.
    Takes a PDF file path and returns the final response dict
    with schema as required by the hackathon.
    """
    doc = fitz.open(pdf_path)

    pagewise: List[PageLineItems] = []
    total_input = total_output = total_tok = 0

    for idx, _ in enumerate(doc):
        page_no = idx + 1

        # 1) Try to get digital text
        text = extract_text_digital(pdf_path, idx)

        if len(text.strip()) > 30:
            print(f"Page {page_no} → digital text detected")
            invoice_text = text
        else:
            print(f"Page {page_no} → scanned/handwritten detected, running Vision OCR")
            b64 = page_to_image_b64(pdf_path, idx, all_images=None)
            invoice_text = ocr_page_with_vision(b64)

        # 2) Turn invoice_text into structured JSON
        structured_json_str = extract_structured_from_text(invoice_text)
        print(f"Extracted JSON from Page {page_no}:\n", structured_json_str)
        try:
         structured_json = json.loads(structured_json_str)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode failed on page {page_no}: {e}")
            print("Skipping this page due to invalid JSON from LLM")
            continue

        try:
            validated_output = validate_invoice_json(structured_json)
            pli = validated_output.data.pagewise_line_items

            if not pli:
                print(f"⚠️ Page {page_no}: no pagewise_line_items found, skipping this page")
                continue  # <-- now correctly under the if

            # Take the first page entry
            page_obj = pli[0]

            # Ensure page_no matches current page (and matches your Pydantic type)
            page_obj.page_no = str(page_no)

            # Fix math and clean rows
            page_obj.bill_items = fix_inconsistent_amounts(page_obj.bill_items)
            page_obj.bill_items = clean_items(page_obj.bill_items)

            # Store cleaned result
            pagewise.append(page_obj)

            # Accumulate token usage from this page
            total_input += validated_output.data.token_usage.input_tokens
            total_output += validated_output.data.token_usage.output_tokens
            total_tok += validated_output.data.token_usage.total_tokens

            print(f"✅ Page {page_no} validated successfully")

        except ValidationError as e:
            print(f"❌ Validation failed on page {page_no}:", e)
            print("Skipping this page due to bad format")
            continue

    # Build final response in hackathon format
    total_items = sum(len(p.bill_items) for p in pagewise)

    final_response = {
        "is_success": True,
        "token_usage": {
            "total_tokens": total_tok,
            "input_tokens": total_input,
            "output_tokens": total_output,
        },
        "data": {
            "pagewise_line_items": [p.model_dump() for p in pagewise],
            "total_item_count": total_items,
        },
    }
    return final_response

