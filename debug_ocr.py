"""
Debug OCR output to understand what Tesseract sees.
This will show each word with its position information.
"""

import sys
import os

try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    import pdf2image
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install opencv-python pytesseract pdf2image pillow")
    sys.exit(1)

def load_image(file_path):
    """Load image from file (PDF or image)"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        pil_images = pdf2image.convert_from_path(file_path, dpi=300)
        img_array = np.array(pil_images[0])
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array
    else:
        return cv2.imread(file_path)

def debug_ocr(file_path):
    """Debug OCR output with position information"""
    print(f"\n{'='*80}")
    print(f"DEBUG OCR: {file_path}")
    print(f"{'='*80}\n")

    # Load image
    img = load_image(file_path)
    if img is None:
        print("Failed to load image")
        return

    print(f"Image size: {img.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Enhance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(thresh)

    # Get OCR data with positions
    ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config='--oem 3 --psm 6')

    # Collect words with positions
    words = []
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        text = str(ocr_data['text'][i]).strip()
        conf = int(ocr_data['conf'][i])

        if text and conf > 10:
            words.append({
                'text': text,
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'conf': conf
            })

    # Sort by y then x
    words.sort(key=lambda w: (w['y'], w['x']))

    print(f"\nTotal words detected: {len(words)}\n")

    # Group into rows
    rows = []
    current_row = []
    last_y = -100
    y_tolerance = 20

    for word in words:
        if abs(word['y'] - last_y) > y_tolerance:
            if current_row:
                current_row.sort(key=lambda w: w['x'])
                rows.append(current_row)
            current_row = [word]
            last_y = word['y']
        else:
            current_row.append(word)

    if current_row:
        current_row.sort(key=lambda w: w['x'])
        rows.append(current_row)

    print(f"Rows detected: {len(rows)}\n")
    print("-" * 80)

    # Print each row
    for i, row in enumerate(rows):
        y = row[0]['y']
        row_text = ' | '.join([w['text'] for w in row])

        # Check if this looks like an item row (has numbers)
        has_numbers = any(w['text'].replace('.', '').replace(',', '').isdigit() for w in row)
        marker = ">>>" if has_numbers else "   "

        print(f"{marker} Row {i:3d} (y={y:4d}): {row_text}")

        # For item rows, show more detail
        if has_numbers and len(row) >= 3:
            print(f"       Cells: {[w['text'] for w in row]}")
            print(f"       X-pos: {[w['x'] for w in row]}")
            print()

    print("-" * 80)

    # Also show raw text
    print("\n\nRAW OCR TEXT:")
    print("="*80)
    raw_text = pytesseract.image_to_string(pil_img, config='--oem 3 --psm 6')
    print(raw_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_ocr.py <invoice_path>")
        print("Example: python debug_ocr.py invoice.pdf")
        sys.exit(1)

    debug_ocr(sys.argv[1])
