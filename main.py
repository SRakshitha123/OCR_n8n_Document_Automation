
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import pytesseract
import numpy as np
from pathlib import Path

app = FastAPI(title="Invoice OCR API")

# Update this path if Tesseract is installed elsewhere.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

@app.get("/")
def home():
    return {"message": "Invoice OCR API is running"}

@app.post("/ocr")
async def read_invoice(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty"
        )

    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must be 5 MB or less"
        )

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    try:
        extracted_text = pytesseract.image_to_string(processed)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="OCR processing failed. Check Tesseract installation."
        ) from exc

    return {
        "filename": file.filename,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "ocr_text": extracted_text.strip()
    }
