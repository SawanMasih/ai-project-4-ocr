"""
DecodeLabs - Project 4: Image or Text Recognition (Basic)
Path 1: OCR with pytesseract + OpenCV

Pipeline (IPO model):
  Input   -> load image (H x W x 3 array)
  Process -> grayscale -> Gaussian blur -> deskew -> adaptive threshold -> Tesseract
  Output  -> recognised text, per-word confidence, 80% confidence filter, annotated image

Usage:
  python ocr_recognition.py                 # creates and uses a sample image
  python ocr_recognition.py my_image.png    # use your own image
Requires: pip install opencv-python pytesseract numpy   (+ Tesseract engine installed)
"""
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

CONF_THRESHOLD = 80      # project minimum standard (percent)
PSM_MODE = 6             # 6 = single uniform block of text; try 3 or 11 for other layouts


def make_sample_image(path="sample_input.png"):
    """Create a sample image with text so the script runs out of the box."""
    img = np.full((300, 900, 3), 255, np.uint8)
    cv2.putText(img, "DecodeLabs AI Project 4", (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(img, "Text Recognition with OCR", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 40), 3)
    noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)   # add some noise
    img = cv2.subtract(img, noise)
    cv2.imwrite(path, img)
    return path


def deskew(gray):
    """Estimate text rotation and rotate it back to horizontal."""
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # Step 1: grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)               # Step 2: Gaussian blur
    straight = deskew(blur)                                # Step 3: deskew
    thresh = cv2.adaptiveThreshold(                        # Step 4: adaptive threshold
        straight, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10)
    return gray, thresh


def recognise(thresh):
    config = f"--oem 3 --psm {PSM_MODE}"
    data = pytesseract.image_to_data(thresh, config=config, output_type=Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if txt.strip() and conf >= 0:
            words.append({"text": txt, "conf": conf,
                          "box": (data["left"][i], data["top"][i],
                                  data["width"][i], data["height"][i])})
    return words


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else make_sample_image()
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")

    gray, thresh = preprocess(img)
    cv2.imwrite("output_preprocessed.png", thresh)
    words = recognise(thresh)

    kept = [w for w in words if w["conf"] >= CONF_THRESHOLD]     # the 80% gate
    dropped = [w for w in words if w["conf"] < CONF_THRESHOLD]

    annotated = img.copy()
    for w in kept:
        x, y, bw, bh = w["box"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        cv2.putText(annotated, f'{w["text"]} {w["conf"]:.0f}%', (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("output_annotated.png", annotated)

    print("=" * 55)
    print(f"Image: {path}   Size: {img.shape[1]}x{img.shape[0]}   PSM: {PSM_MODE}")
    print("=" * 55)
    print(f"{'WORD':<25}{'CONFIDENCE':>12}   STATUS")
    for w in words:
        status = "ACCEPT" if w["conf"] >= CONF_THRESHOLD else "DROP"
        print(f'{w["text"]:<25}{w["conf"]:>11.1f}%   {status}')
    print("-" * 55)
    print("Recognised text (conf >= 80%):", " ".join(w["text"] for w in kept))
    if kept:
        avg = sum(w["conf"] for w in kept) / len(kept)
        print(f"Average confidence: {avg:.1f}%  ->",
              "PASS (>= 80%)" if avg >= CONF_THRESHOLD else "FAIL")
    print(f"Accepted: {len(kept)}  Dropped: {len(dropped)}")
    print("Saved: output_preprocessed.png, output_annotated.png")


if __name__ == "__main__":
    main()
