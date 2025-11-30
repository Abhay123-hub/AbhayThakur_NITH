from fastapi import FastAPI
from pydantic import BaseModel
import requests
import uuid
import os

#change this if your extraction file has a different name
from invoice_extractor import process_pdf_bytes


app = FastAPI()


class ExtractRequest(BaseModel):
    document: str  # URL to the PDF


@app.post("/extract-bill-data")
def extract_bill_data(req: ExtractRequest):
    """
    Hackathon-required endpoint.
    Input:
      {
        "document": "<public URL to PDF>"
      }

    Output (on success, status 200):
      {
        "is_success": true,
        "token_usage": {
          "total_tokens": int,
          "input_tokens": int,
          "output_tokens": int
        },
        "data": {
          "pagewise_line_items": [...],
          "total_item_count": int
        }
      }
    """
    # ---------- 1. Download the PDF from the given URL ----------
    try:
        resp = requests.get(req.document, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        # On download error, still return a valid JSON (is_success = false)
        return {
            "is_success": False,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0,
            },
            "error": f"Failed to download document: {e}",
        }

    # ---------- 2. Save to a temporary PDF file ----------
    temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
    try:
        with open(temp_filename, "wb") as f:
            f.write(resp.content)

        # ---------- 3. Run your existing extraction pipeline ----------
        result = process_pdf_bytes(temp_filename)
        # Expecting result to already be in format:
        # {
        #   "is_success": True/False,
        #   "token_usage": {...},
        #   "data": {...}
        # }

    finally:
        # ---------- 4. Clean up temp file ----------
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except OSError:
            # not critical if delete fails
            pass

    # ---------- 5. Make sure schema matches hackathon spec ----------
    # Fill any missing parts defensively, just in case.

    if not isinstance(result, dict):
        # If somehow your function didn't return a dict, wrap it.
        return {
            "is_success": False,
            "token_usage": {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            },
            "data": {
                "pagewise_line_items": [],
                "total_item_count": 0,
            },
            "error": "Extractor did not return a valid dict",
        }

    # Ensure top-level keys exist
    result.setdefault("is_success", True)
    result.setdefault("token_usage", {})
    result.setdefault("data", {})

    # Ensure token_usage fields exist and are integers
    tu = result["token_usage"]
    tu.setdefault("total_tokens", 0)
    tu.setdefault("input_tokens", 0)
    tu.setdefault("output_tokens", 0)

    # Ensure data fields exist
    data = result["data"]
    data.setdefault("pagewise_line_items", [])
    data.setdefault("total_item_count", 0)

    # Now result matches the required schema shape
    return result
