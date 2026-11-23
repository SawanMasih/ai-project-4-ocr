# Text Recognition with OCR (AI Project 4)

Reads text from an image using Tesseract OCR and OpenCV.

## How it works
1. Load image
2. Pre-process: grayscale, Gaussian blur, deskew, adaptive threshold
3. Recognise text with pytesseract
4. Keep only words with confidence >= 80%
5. Save annotated output image

## Setup
Install the Tesseract engine, then:
pip install -r requirements.txt

## Run
python ocr_recognition.py test_image.png

## Results
Average confidence: 95.5% (PASS, minimum is 80%)

![Output](output_annotated.png)