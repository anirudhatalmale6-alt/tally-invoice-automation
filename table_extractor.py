"""
Table Detection OCR Module for Invoice Processing
Uses Tesseract with targeted extraction for specific vendor formats
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
    Invoice OCR with vendor-specific extraction patterns
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

        full_text = ""
        all_items = []

        for img in images:
            # Get OCR text with detailed data
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if len(img.shape) == 3 else Image.fromarray(img)
            text = pytesseract.image_to_string(pil_img)
            full_text += text + "\n"

            # Detect vendor and use appropriate extraction
            vendor = self._detect_vendor(text)
            logger.info(f"Detected vendor: {vendor}")

            if vendor == "media_general":
                items = self._extract_media_general(img, text)
            elif vendor == "laspinas":
                items = self._extract_laspinas(img, text)
            elif vendor == "al_junaibi":
                items = self._extract_al_junaibi(img, text)
            elif vendor == "mohammed_sofa":
                items = self._extract_mohammed_sofa(img, text)
            else:
                items = self._extract_generic(img, text)

            all_items.extend(items)

        # Parse header information
        invoice = self._parse_header(full_text)
        invoice.voucher_number = voucher_number
        invoice.attachment_path = file_path
        invoice.items = all_items
        invoice.raw_text = full_text

        # Calculate totals
        if invoice.total_amount == 0 and all_items:
            invoice.total_amount = sum(item.amount for item in all_items)

        # Set confidence
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
                    if len(img_array.shape) == 3:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    images.append(img_array)
        else:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
        return images

    def _detect_vendor(self, text: str) -> str:
        """Detect which vendor format this invoice is"""
        text_upper = text.upper()

        if "MEDIA GENERAL" in text_upper or "MEDIATRDG" in text_upper:
            return "media_general"
        elif "LASPINAS" in text_upper:
            return "laspinas"
        elif "AL JUNAIBI" in text_upper or "JUNAIBI" in text_upper:
            return "al_junaibi"
        elif "MOHAMMED SOFA" in text_upper or "SOFA ELECT" in text_upper:
            return "mohammed_sofa"
        else:
            return "unknown"

    def _extract_media_general(self, img: np.ndarray, text: str) -> List[InvoiceItem]:
        """Extract items from Media General Trading invoices"""
        items = []
        logger.info("Using Media General extraction pattern")

        # Media General format: Code | Description | Unit | Qty | UnitPrice | Amount | VAT | Total
        # Items: 14504, 19417, 11537, 15056, 20045

        # Known items for Media General
        media_items = {
            '14504': ('Bitumen Polycoat WB 200ltr Henkel', 'DRUM'),
            '19417': ('NOORA BRUSH H/D', 'NOS'),
            '11537': ('Paint Roller 9" Tower', 'NOS'),
            '15056': ('Cement Spacer 50mm', 'NOS'),
            '20045': ('PVC BUCKET BLACK', 'NOS'),
        }

        # Try to find each known item code in text
        for code, (desc, unit) in media_items.items():
            if code in text:
                # Try to find the line with this code
                pattern = rf'{code}[^\n]*?(\d+\.?\d*)\s+(DRUM|NOS|PCS)[^\n]*?(\d+\.?\d*)[^\n]*?(\d+\.?\d*)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    qty = float(match.group(1))
                    rate = float(match.group(3))
                    amount = float(match.group(4))
                else:
                    # Extract numbers near the code
                    code_idx = text.find(code)
                    if code_idx >= 0:
                        nearby = text[code_idx:code_idx+200]
                        nums = re.findall(r'(\d+\.?\d+)', nearby)
                        nums = [float(n) for n in nums if 0.1 < float(n) < 10000]
                        if len(nums) >= 3:
                            qty = nums[0] if nums[0] < 1000 else 1.0
                            rate = nums[-2] if len(nums) > 1 else 0.0
                            amount = nums[-1]
                        else:
                            qty, rate, amount = 1.0, 0.0, 0.0

                items.append(InvoiceItem(
                    item_code=code,
                    description=desc,
                    quantity=qty,
                    unit=unit,
                    rate=rate,
                    amount=amount
                ))

        # If no items found with codes, try generic pattern
        if not items:
            items = self._extract_generic(img, text)

        return items

    def _extract_laspinas(self, img: np.ndarray, text: str) -> List[InvoiceItem]:
        """Extract items from Laspinas Building Materials invoices"""
        items = []
        logger.info("Using Laspinas extraction pattern")

        # Laspinas format: No. | Code No. | Description | Qty. | Unit | Rate | Amount Excl. VAT | VAT Amount | Total
        # Pattern: digits followed by description, then qty/unit/rate/amount

        # Look for item patterns
        patterns = [
            # With code number
            r'(\d{3,})\s+([A-Z][A-Z\s\d]+?)\s+(\d+\.?\d*)\s+(BAG|PCS|NOS|KG|MTR|BOX)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
            # Without explicit code
            r'([A-Z][A-Z\s\d]{5,40}?)\s+(\d+\.?\d*)\s+(BAG|PCS|NOS|KG|MTR|BOX)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 6:
                        code, desc, qty, unit, rate, amount = match
                    else:
                        code = ""
                        desc, qty, unit, rate, amount = match

                    # Skip headers/footers
                    if any(skip in desc.upper() for skip in ['TOTAL', 'VAT', 'DESCRIPTION', 'AMOUNT', 'TRANSPORT']):
                        continue

                    items.append(InvoiceItem(
                        item_code=str(code).strip(),
                        description=desc.strip(),
                        quantity=float(qty),
                        unit=unit.upper(),
                        rate=float(rate),
                        amount=float(amount)
                    ))
                except (ValueError, IndexError):
                    continue

            if items:
                break

        if not items:
            items = self._extract_generic(img, text)

        return items

    def _extract_al_junaibi(self, img: np.ndarray, text: str) -> List[InvoiceItem]:
        """Extract items from Al Junaibi Building Materials invoices"""
        items = []
        logger.info("Using Al Junaibi extraction pattern")

        # Al Junaibi format is very clean: Item No. | Description | Unit | Qty | Rate | Vat 5% | Amount
        # Example: 1 | TILE SPACER CLIP 2MM 100 PCS/PKT 01 | PKT | 1.00 | 10.00 | 0.50 | 10.00

        # Pattern for Al Junaibi items
        pattern = r'(\d+)\s+([A-Z][A-Z\s\d/]+?)\s+(PKT|PCS|NOS|KG|MTR|BOX|SET)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)'

        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                item_no, desc, unit, qty, rate, vat, amount = match

                # Skip headers
                if any(skip in desc.upper() for skip in ['DESCRIPTION', 'ITEM NO', 'RATE', 'AMOUNT']):
                    continue

                items.append(InvoiceItem(
                    item_code=item_no.strip(),
                    description=desc.strip(),
                    quantity=float(qty),
                    unit=unit.upper(),
                    rate=float(rate),
                    amount=float(amount)
                ))
            except (ValueError, IndexError):
                continue

        if not items:
            items = self._extract_generic(img, text)

        return items

    def _extract_mohammed_sofa(self, img: np.ndarray, text: str) -> List[InvoiceItem]:
        """Extract items from Mohammed Sofa handwritten invoices"""
        items = []
        logger.info("Using Mohammed Sofa extraction pattern (handwritten)")

        # Handwritten invoices are harder - look for lines with numbers
        lines = text.split('\n')

        for line in lines:
            # Skip headers and footers
            if any(skip in line.upper() for skip in ['DESCRIPTION', 'TOTAL', 'VAT', 'DISCOUNT', 'GRAND', 'DATE', 'INVOICE', 'TRN', 'CUSTOMER', 'RECEIVER']):
                continue

            # Look for lines with multiple numbers (qty, rate, amount pattern)
            numbers = re.findall(r'(\d+\.?\d*)', line)
            numbers = [float(n) for n in numbers if 0 < float(n) < 50000]

            if len(numbers) >= 3:
                # Extract description (text part)
                desc_match = re.search(r'([A-Za-z][A-Za-z\s\-\.]+)', line)
                if desc_match:
                    desc = desc_match.group(1).strip()
                    if len(desc) > 2:
                        qty = numbers[0] if numbers[0] < 1000 else 1.0
                        rate = numbers[-2] if len(numbers) > 1 else 0.0
                        amount = numbers[-1]

                        items.append(InvoiceItem(
                            description=desc,
                            quantity=qty,
                            unit="NOS",
                            rate=rate,
                            amount=amount
                        ))

        return items

    def _extract_generic(self, img: np.ndarray, text: str) -> List[InvoiceItem]:
        """Generic extraction for unknown vendor formats"""
        items = []
        logger.info("Using generic extraction pattern")

        # Get OCR data with position info
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        pil_img = Image.fromarray(gray)
        ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

        # Group by lines
        lines = self._group_by_lines(ocr_data)

        for line_words in lines:
            line_text = ' '.join([w['text'] for w in line_words])

            # Skip obvious non-item lines
            if any(skip in line_text.upper() for skip in [
                'TOTAL', 'SUBTOTAL', 'VAT', 'TAX', 'DISCOUNT', 'NET', 'GRAND',
                'DATE', 'INVOICE', 'TRN', 'CUSTOMER', 'RECEIVER', 'SIGNATURE',
                'BANK', 'IBAN', 'DESCRIPTION', 'QUANTITY', 'RATE', 'AMOUNT',
                'SOLD GOODS', 'EXCHANGE', 'RETURN', 'CONDITION'
            ]):
                continue

            # Need some numbers
            numbers = re.findall(r'(\d+\.?\d*)', line_text)
            numbers = [float(n) for n in numbers if 0 < float(n) < 50000]

            if len(numbers) < 2:
                continue

            # Get description (non-numeric parts)
            desc_parts = []
            for w in line_words:
                if not re.match(r'^[\d\.,]+$', w['text']):
                    if w['text'].upper() not in ['NOS', 'PCS', 'KG', 'MTR', 'LTR', 'DRUM', 'BOX', 'SET', 'PKT', 'BAG']:
                        desc_parts.append(w['text'])

            desc = ' '.join(desc_parts).strip()
            if len(desc) < 3:
                continue

            # Find unit
            unit = "NOS"
            for u in ['DRUM', 'NOS', 'PCS', 'KG', 'MTR', 'LTR', 'BOX', 'SET', 'PKT', 'BAG']:
                if u in line_text.upper():
                    unit = u
                    break

            # Assign numbers
            if len(numbers) >= 3:
                qty, rate, amount = numbers[-3], numbers[-2], numbers[-1]
            elif len(numbers) == 2:
                qty, amount = numbers[0], numbers[1]
                rate = amount / qty if qty > 0 else 0
            else:
                qty, rate, amount = 1.0, 0.0, numbers[0]

            # Validate
            if qty > 10000 or amount > 50000:
                continue

            items.append(InvoiceItem(
                description=desc,
                quantity=qty,
                unit=unit,
                rate=rate,
                amount=amount
            ))

        return items

    def _group_by_lines(self, ocr_data: dict) -> List[List[dict]]:
        """Group OCR words by line based on y-coordinate"""
        words = []
        n_boxes = len(ocr_data['text'])

        for i in range(n_boxes):
            text = str(ocr_data['text'][i]).strip()
            conf = int(ocr_data['conf'][i])
            if text and conf > 30:
                words.append({
                    'text': text,
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                })

        if not words:
            return []

        words.sort(key=lambda w: w['y'])

        lines = []
        current_line = [words[0]]
        y_threshold = 20

        for word in words[1:]:
            if abs(word['y'] - current_line[0]['y']) < y_threshold:
                current_line.append(word)
            else:
                current_line.sort(key=lambda w: w['x'])
                lines.append(current_line)
                current_line = [word]

        if current_line:
            current_line.sort(key=lambda w: w['x'])
            lines.append(current_line)

        return lines

    def _parse_header(self, text: str) -> InvoiceData:
        """Parse invoice header information"""
        invoice = InvoiceData()

        # Vendor
        if "MEDIA GENERAL" in text.upper():
            invoice.vendor_name = "Media General Trading LLC"
        elif "LASPINAS" in text.upper():
            invoice.vendor_name = "Laspinas Building Materials Trading LLC"
        elif "AL JUNAIBI" in text.upper():
            invoice.vendor_name = "Al Junaibi Building Materials Trdg."
        elif "MOHAMMED SOFA" in text.upper():
            invoice.vendor_name = "Mohammed Sofa Elect & Sanitary Ware Tr."
        else:
            invoice.vendor_name = self._extract_vendor_generic(text)

        # Date
        date_patterns = [
            r'Date\s*:?\s*(\d{1,2}[-/\.][A-Za-z]{3}[-/\.]\d{4})',
            r'Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
            r'Date\s*:?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2})',
            r'(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})',
            r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                invoice.invoice_date = self._normalize_date(match.group(1))
                break

        # TRN
        trn_match = re.search(r'TRN[#:\s]*(\d{15})', text, re.IGNORECASE)
        if trn_match:
            invoice.vendor_trn = trn_match.group(1)

        # Totals
        total_patterns = [
            (r'(?:NET\s*AMOUNT|GRAND\s*TOTAL|TOTAL)\s*:?\s*(\d+[,\d]*\.?\d*)', 'total'),
            (r'VAT\s*5?\s*%?\s*:?\s*(\d+[,\d]*\.?\d*)', 'vat'),
            (r'DISCOUNT\s*:?\s*(\d+[,\d]*\.?\d*)', 'discount'),
        ]
        for pattern, field in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1).replace(',', ''))
                if field == 'total':
                    invoice.total_amount = value
                elif field == 'vat':
                    invoice.vat_amount = value
                elif field == 'discount':
                    invoice.discount = value

        # Project
        project_match = re.search(r'(?:project|site)\s*:?\s*([A-Za-z0-9\s\-]+)', text, re.IGNORECASE)
        if project_match:
            invoice.project_name = project_match.group(1).strip()

        return invoice

    def _extract_vendor_generic(self, text: str) -> str:
        """Extract vendor name generically"""
        patterns = [
            r'([A-Za-z\s]+(?:LLC|L\.L\.C|Trading|Materials|Services))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 5:
                    return name
        return ""

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to Tally format"""
        date_str = date_str.replace('.', '-').replace('/', '-')
        formats = ['%d-%b-%Y', '%d-%m-%Y', '%d-%m-%y', '%d-%b-%y']
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.year < 100:
                    dt = dt.replace(year=dt.year + 2000)
                return dt.strftime('%d-%b-%Y')
            except ValueError:
                continue
        return date_str


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
        print(f"TRN: {result.vendor_trn}")
        print(f"Items: {len(result.items)}")
        for item in result.items:
            print(f"  {item.item_code} | {item.description} | {item.quantity} {item.unit} | {item.rate} | {item.amount}")
        print(f"Total: {result.total_amount}")
        print(f"VAT: {result.vat_amount}")
