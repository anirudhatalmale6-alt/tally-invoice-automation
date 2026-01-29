"""
Table Detection OCR Module for Invoice Processing
Uses OpenCV for table detection and Tesseract for OCR
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
import logging

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InvoiceItem:
    """Represents a single line item from an invoice"""
    item_code: str = ""
    description: str = ""
    quantity: float = 0.0
    unit: str = "Nos"
    rate: float = 0.0
    amount: float = 0.0
    project: str = ""


@dataclass
class InvoiceData:
    """Represents extracted invoice data"""
    voucher_number: str = ""
    vendor_name: str = ""
    vendor_address: str = ""
    vendor_trn: str = ""
    invoice_date: str = ""
    customer_name: str = ""
    items: List[InvoiceItem] = field(default_factory=list)
    subtotal: float = 0.0
    vat_amount: float = 0.0
    vat_rate: float = 5.0
    discount: float = 0.0
    total_amount: float = 0.0
    project_name: str = ""
    attachment_path: str = ""
    ocr_confidence: float = 0.0
    needs_review: bool = False
    raw_text: str = ""
    extraction_notes: List[str] = field(default_factory=list)


class TableInvoiceOCR:
    """
    Invoice OCR using OpenCV table detection and Tesseract
    """

    def __init__(self):
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract is required. Install pytesseract.")
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required. Install opencv-python.")
        logger.info("Table OCR initialized with OpenCV and Tesseract")

    def extract_from_file(self, file_path: str) -> InvoiceData:
        """Extract invoice data from a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        voucher_number = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()

        # Load image(s)
        images = self._load_images(file_path, ext)

        # Get full text for header extraction
        full_text = ""
        all_items = []

        for img in images:
            # Get full page OCR text
            text = pytesseract.image_to_string(img)
            full_text += text + "\n"

            # Try to extract table items using line-by-line OCR with position info
            items = self._extract_items_with_positions(img, text)
            all_items.extend(items)

        # Parse header information
        invoice = self._parse_header(full_text)
        invoice.voucher_number = voucher_number
        invoice.attachment_path = file_path
        invoice.items = all_items
        invoice.raw_text = full_text

        # Calculate totals if not found
        if invoice.total_amount == 0 and all_items:
            invoice.total_amount = sum(item.amount for item in all_items)

        # Set confidence based on extraction success
        if all_items:
            invoice.ocr_confidence = 85.0
            invoice.needs_review = False
        else:
            invoice.ocr_confidence = 50.0
            invoice.needs_review = True
            invoice.extraction_notes.append("No items extracted - manual review needed")

        return invoice

    def _load_images(self, file_path: str, ext: str) -> List[np.ndarray]:
        """Load images from file"""
        images = []

        if ext == '.pdf':
            if PDF2IMAGE_AVAILABLE:
                pil_images = pdf2image.convert_from_path(file_path, dpi=300)
                for pil_img in pil_images:
                    img_array = np.array(pil_img)
                    # Convert RGB to BGR for OpenCV
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    images.append(img_array)
        else:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)

        return images

    def _extract_items_with_positions(self, img: np.ndarray, full_text: str) -> List[InvoiceItem]:
        """Extract items using OCR with position data to maintain table structure"""
        items = []

        # Convert to grayscale for better OCR
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Get OCR data with positions
        pil_img = Image.fromarray(gray)
        ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

        # Group words by line (similar y-coordinates)
        lines = self._group_words_by_line(ocr_data)

        logger.info(f"Found {len(lines)} text lines in document")

        # Process each line to find item rows
        for line_words in lines:
            line_text = ' '.join([w['text'] for w in line_words])
            item = self._parse_item_line(line_text, line_words)
            if item:
                items.append(item)

        # If no items found with position-based extraction, try pattern matching on full text
        if not items:
            items = self._extract_items_from_text(full_text)

        return items

    def _group_words_by_line(self, ocr_data: dict) -> List[List[dict]]:
        """Group OCR words by line based on y-coordinate"""
        words = []
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text and int(ocr_data['conf'][i]) > 30:  # Filter low confidence
                words.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i],
                    'conf': int(ocr_data['conf'][i])
                })

        if not words:
            return []

        # Sort by y-coordinate
        words.sort(key=lambda w: w['y'])

        # Group words with similar y-coordinates (within threshold)
        lines = []
        current_line = [words[0]]
        y_threshold = 15  # pixels

        for word in words[1:]:
            if abs(word['y'] - current_line[0]['y']) < y_threshold:
                current_line.append(word)
            else:
                # Sort line by x-coordinate
                current_line.sort(key=lambda w: w['x'])
                lines.append(current_line)
                current_line = [word]

        if current_line:
            current_line.sort(key=lambda w: w['x'])
            lines.append(current_line)

        return lines

    def _parse_item_line(self, line_text: str, line_words: List[dict]) -> Optional[InvoiceItem]:
        """Parse a line of text to extract item information"""
        # Skip header/footer lines
        skip_patterns = ['item', 'code', 'description', 'qty', 'quantity', 'rate', 'amount',
                        'unit', 'price', 'sl', 'no.', 'total', 'subtotal', 'vat', 'tax',
                        'discount', 'net', 'grand', 'five hundred', 'iban', 'bank', 'a/c',
                        'signature', 'stamp', 'receiver', 'customer', 'date', 'invoice',
                        'trn', 'sold goods', 'exchange', 'return', 'condition', 'delivery',
                        'declare', 'dispute', 'authorised']

        line_lower = line_text.lower()
        if any(skip in line_lower for skip in skip_patterns):
            return None

        # Extract numbers from the line
        numbers = re.findall(r'(\d+\.?\d*)', line_text)
        numbers = [float(n) for n in numbers]

        # Filter out unreasonable numbers (too large = likely IBAN, phone numbers)
        numbers = [n for n in numbers if n < 100000]

        # Need at least some numbers for qty/rate/amount
        if len(numbers) < 2:
            return None

        # Try to find item description (text that's not just numbers)
        description_parts = []
        for word in line_words:
            text = word['text']
            # Keep text that's not purely numeric and not a unit
            if not re.match(r'^[\d\.,]+$', text):
                if text.upper() not in ['NOS', 'PCS', 'KG', 'MTR', 'LTR', 'DRUM', 'BOX', 'SET', 'EA', 'EACH', 'UNIT']:
                    description_parts.append(text)

        description = ' '.join(description_parts)

        # Check for item code at the beginning (4-6 digit number)
        item_code = ""
        code_match = re.match(r'^(\d{4,6})\s+', line_text)
        if code_match:
            item_code = code_match.group(1)

        # Find unit
        unit = "NOS"
        unit_match = re.search(r'\b(NOS|PCS|KG|MTR|LTR|DRUM|BOX|SET|EA|EACH|UNIT)\b', line_text, re.IGNORECASE)
        if unit_match:
            unit = unit_match.group(1).upper()

        # Clean up description
        description = re.sub(r'\d{4,6}', '', description)  # Remove item codes
        description = re.sub(r'\b(NOS|PCS|KG|MTR|LTR|DRUM|BOX|SET|EA|EACH|UNIT)\b', '', description, flags=re.IGNORECASE)
        description = re.sub(r'\s+', ' ', description).strip()

        # Need a meaningful description
        if len(description) < 3:
            return None

        # Assign numbers: typically last number is amount, second to last is rate, third to last is qty
        qty = 1.0
        rate = 0.0
        amount = 0.0

        if len(numbers) >= 3:
            qty = numbers[-3]
            rate = numbers[-2]
            amount = numbers[-1]
        elif len(numbers) == 2:
            qty = numbers[0]
            amount = numbers[1]
            if qty > 0:
                rate = amount / qty
        elif len(numbers) == 1:
            amount = numbers[0]

        # Validate: qty should be reasonable
        if qty > 10000 or amount > 50000:
            return None

        return InvoiceItem(
            item_code=item_code,
            description=description,
            quantity=qty,
            unit=unit,
            rate=rate,
            amount=amount
        )

    def _extract_items_from_text(self, text: str) -> List[InvoiceItem]:
        """Fallback: Extract items using pattern matching on full text"""
        items = []

        # Pattern: Item code + Description + numbers
        # Example: "14504 Bitumen Polycoat WB 200ltr Henkel DRUM 1.00 330.00 330.00"
        patterns = [
            # Pattern with item code
            r'(\d{4,6})\s+([A-Za-z][A-Za-z\s\d\-\./\'\"]+?)\s+(DRUM|NOS|PCS|KG|MTR|LTR|BOX|SET)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
            # Pattern without item code
            r'([A-Za-z][A-Za-z\s\d\-\./\'\"]{5,30}?)\s+(DRUM|NOS|PCS|KG|MTR|LTR|BOX|SET)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 6:
                        item_code, description, unit, qty, rate, amount = match
                    else:
                        item_code = ""
                        description, unit, qty, rate, amount = match

                    # Skip if looks like header/footer
                    if any(skip in description.lower() for skip in ['total', 'subtotal', 'vat', 'tax', 'iban']):
                        continue

                    item = InvoiceItem(
                        item_code=str(item_code).strip(),
                        description=description.strip(),
                        quantity=float(qty),
                        unit=unit.upper(),
                        rate=float(rate),
                        amount=float(amount)
                    )
                    items.append(item)
                except (ValueError, IndexError):
                    continue

            if items:
                break

        return items

    def _parse_header(self, text: str) -> InvoiceData:
        """Parse invoice header information"""
        invoice = InvoiceData()
        invoice.vendor_name = self._extract_vendor(text)
        invoice.invoice_date = self._extract_date(text)
        invoice.vendor_trn = self._extract_trn(text)
        invoice.total_amount = self._extract_amount(text, ['net amount', 'total', 'grand total'])
        invoice.vat_amount = self._extract_amount(text, ['vat', 'tax 5%', 'vat 5%'])
        invoice.discount = self._extract_amount(text, ['discount'])
        invoice.subtotal = self._extract_amount(text, ['subtotal', 'sub total'])
        invoice.project_name = self._extract_project(text)
        return invoice

    def _extract_vendor(self, text: str) -> str:
        """Extract vendor name"""
        patterns = [
            r'(Media\s*General\s*Trading\s*(?:LLC)?)',
            r'(Al\s*Junaibi\s*Building\s*Materials)',
            r'(Laspinas\s*Building\s*Materials)',
            r'(Mohammed\s*Sofa[^\n]*)',
            r'([A-Za-z\s]+Building\s*Materials)',
            r'([A-Za-z\s]+Trading\s*(?:LLC)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 5:
                    return name
        return ""

    def _extract_date(self, text: str) -> str:
        """Extract invoice date"""
        patterns = [
            r'Date\s*:?\s*(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})',
            r'Date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                return self._normalize_date(date_str)
        return ""

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date format"""
        formats = [
            '%d-%b-%Y', '%d/%b/%Y', '%d-%m-%Y', '%d/%m/%Y',
            '%d-%b-%y', '%d/%b/%y', '%d-%m-%y', '%d/%m/%y',
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.year < 100:
                    dt = dt.replace(year=dt.year + 2000)
                return dt.strftime('%d-%b-%Y')
            except ValueError:
                continue
        return date_str

    def _extract_trn(self, text: str) -> str:
        """Extract Tax Registration Number"""
        match = re.search(r'TRN\s*:?\s*(\d{15})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    def _extract_amount(self, text: str, keywords: List[str]) -> float:
        """Extract amount by keywords"""
        for keyword in keywords:
            pattern = rf'{keyword}\s*:?\s*(\d+[,\d]*\.?\d*)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    pass
        return 0.0

    def _extract_project(self, text: str) -> str:
        """Extract project name from stamp"""
        patterns = [
            r'project\s*:?\s*([A-Za-z0-9\s\-]+)',
            r'site\s*:?\s*([A-Za-z0-9\s\-]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project = match.group(1).strip()
                if len(project) > 2:
                    return project
        return ""


def process_invoice(file_path: str) -> InvoiceData:
    """Process a single invoice"""
    ocr = TableInvoiceOCR()
    return ocr.extract_from_file(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = process_invoice(sys.argv[1])
        print(f"Vendor: {result.vendor_name}")
        print(f"Date: {result.invoice_date}")
        print(f"Items: {len(result.items)}")
        for item in result.items:
            print(f"  - {item.description}: {item.quantity} x {item.rate} = {item.amount}")
        print(f"Total: {result.total_amount}")
        print(f"VAT: {result.vat_amount}")
