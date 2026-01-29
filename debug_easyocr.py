"""Debug EasyOCR output for handwritten invoices"""
import sys
import os
import cv2
import numpy as np
import easyocr
import pdf2image

def debug_easyocr(file_path):
    # Load image
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        pil_images = pdf2image.convert_from_path(file_path, dpi=300)
        img = np.array(pil_images[0])
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(file_path)

    print(f"Image size: {img.shape}")

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    # Get results
    results = reader.readtext(img)

    print(f"\nEasyOCR found {len(results)} text regions:\n")

    # Group by Y
    rows = {}
    for bbox, text, conf in results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        row_key = int(y_center / 30) * 30
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append({
            'text': text,
            'x': bbox[0][0],
            'conf': conf
        })

    # Print grouped rows
    for y in sorted(rows.keys()):
        row_items = sorted(rows[y], key=lambda x: x['x'])
        row_text = ' | '.join([f"{item['text']} ({item['conf']:.2f})" for item in row_items])
        print(f"Y={y:4d}: {row_text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_easyocr.py <invoice_path>")
        sys.exit(1)
    debug_easyocr(sys.argv[1])
