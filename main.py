
import re

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Invoice OCR API")

# Update this path if Tesseract is installed elsewhere.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

MAX_UPLOAD_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}


def extract_invoice_fields(text: str) -> dict:
    """Extract common invoice fields from OCR text."""

    invoice_match = re.search(
        r"\b(?:invoice\s*(?:number|no\.?|#))\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9\-\/]*)",
        text,
        re.IGNORECASE,
    )

    amount_match = re.search(
        r"\b(?:grand\s+total|total\s+amount|amount\s+due|total\s+due|total)"
        r"\s*[:\-]?\s*(?:INR|Rs\.?|₹|\$)?\s*"
        r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        text,
        re.IGNORECASE,
    )

    return {
        "invoice_number": invoice_match.group(1) if invoice_match else None,
        "total_amount": (
            float(amount_match.group(1).replace(",", ""))
            if amount_match
            else None
        ),
    }


@app.get("/")
def home():
    return {"message": "Invoice OCR API is running"}


@app.post("/ocr")
async def read_invoice(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty",
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image size must be 5 MB or less",
        )

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image",
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    try:
        extracted_text = pytesseract.image_to_string(processed)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="OCR processing failed. Check Tesseract installation.",
        ) from exc

    clean_text = extracted_text.strip()
    invoice_fields = extract_invoice_fields(clean_text)

    return {
        "filename": file.filename,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "ocr_text": clean_text,
        "invoice_number": invoice_fields["invoice_number"],
        "total_amount": invoice_fields["total_amount"],
    }
