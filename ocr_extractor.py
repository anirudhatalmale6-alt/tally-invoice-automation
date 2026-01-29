"""
OCR Extraction Module for Invoice Processing
Handles both printed and handwritten invoices using multiple OCR approaches
"""

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

# OCR Libraries
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

from config import OCR_CONFIDENCE_THRESHOLD, FLAG_LOW_CONFIDENCE

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
    project: str = ""  # Project allocation


@dataclass
class InvoiceData:
    """Represents extracted invoice data"""
    voucher_number: str = ""
    vendor_name: str = ""
    vendor_address: str = ""
    vendor_trn: str = ""  # Tax Registration Number
    invoice_date: str = ""
    customer_name: str = ""
    items: List[InvoiceItem] = field(default_factory=list)
    subtotal: float = 0.0
    vat_amount: float = 0.0
    vat_rate: float = 5.0
    discount: float = 0.0
    total_amount: float = 0.0
    project_name: str = ""  # Overall project for the invoice
    attachment_path: str = ""
    ocr_confidence: float = 0.0
    needs_review: bool = False
    raw_text: str = ""
    extraction_notes: List[str] = field(default_factory=list)


class InvoiceOCR:
    """
    Main OCR class for extracting data from invoice images/PDFs
    Uses multiple OCR engines for better accuracy
    """

    def __init__(self, use_easyocr: bool = True, use_tesseract: bool = True):
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.use_tesseract = use_tesseract and TESSERACT_AVAILABLE

        if self.use_easyocr:
            # Initialize EasyOCR with English and Arabic support
            self.reader = easyocr.Reader(['en', 'ar'], gpu=False)
            logger.info("EasyOCR initialized with English and Arabic support")

        if not self.use_easyocr and not self.use_tesseract:
            raise RuntimeError("No OCR engine available. Install pytesseract or easyocr.")

    def extract_from_file(self, file_path: str) -> InvoiceData:
        """
        Extract invoice data from a file (PDF or image)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get voucher number from filename
        voucher_number = os.path.splitext(os.path.basename(file_path))[0]

        # Convert PDF to images if needed
        images = self._load_images(file_path)

        # Perform OCR
        all_text = []
        confidence_scores = []

        for img in images:
            text, confidence = self._perform_ocr(img)
            all_text.append(text)
            confidence_scores.append(confidence)

        combined_text = "\n".join(all_text)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        # Parse the extracted text
        invoice_data = self._parse_invoice_text(combined_text)
        invoice_data.voucher_number = voucher_number
        invoice_data.attachment_path = file_path
        invoice_data.ocr_confidence = avg_confidence
        invoice_data.raw_text = combined_text

        # Flag for review if confidence is low
        if avg_confidence < OCR_CONFIDENCE_THRESHOLD and FLAG_LOW_CONFIDENCE:
            invoice_data.needs_review = True
            invoice_data.extraction_notes.append(
                f"Low OCR confidence ({avg_confidence:.1f}%). Manual review recommended."
            )

        return invoice_data

    def _load_images(self, file_path: str) -> List[Image.Image]:
        """Load images from file (PDF or image format)"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            if not PDF2IMAGE_AVAILABLE:
                raise RuntimeError("pdf2image not available. Install it to process PDFs.")
            images = pdf2image.convert_from_path(file_path, dpi=300)
            return images
        else:
            img = Image.open(file_path)
            return [img]

    def _perform_ocr(self, image: Image.Image) -> tuple:
        """
        Perform OCR on an image using available engines
        Returns (text, confidence_score)
        """
        results = []

        if self.use_easyocr:
            try:
                # EasyOCR works better with handwritten text
                import numpy as np
                img_array = np.array(image)
                ocr_results = self.reader.readtext(img_array)

                text_parts = []
                confidences = []
                for (bbox, text, conf) in ocr_results:
                    text_parts.append(text)
                    confidences.append(conf * 100)

                easyocr_text = " ".join(text_parts)
                easyocr_conf = sum(confidences) / len(confidences) if confidences else 0
                results.append(('easyocr', easyocr_text, easyocr_conf))
                logger.debug(f"EasyOCR confidence: {easyocr_conf:.1f}%")
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")

        if self.use_tesseract:
            try:
                # Tesseract for printed text
                tesseract_text = pytesseract.image_to_string(image)
                # Get confidence from detailed data
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data['conf'] if int(c) > 0]
                tesseract_conf = sum(confidences) / len(confidences) if confidences else 0
                results.append(('tesseract', tesseract_text, tesseract_conf))
                logger.debug(f"Tesseract confidence: {tesseract_conf:.1f}%")
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")

        if not results:
            return "", 0

        # Use the result with highest confidence
        best_result = max(results, key=lambda x: x[2])
        logger.info(f"Using {best_result[0]} result (confidence: {best_result[2]:.1f}%)")
        return best_result[1], best_result[2]

    def _parse_invoice_text(self, text: str) -> InvoiceData:
        """
        Parse OCR text to extract structured invoice data
        """
        invoice = InvoiceData()
        invoice.raw_text = text

        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Extract vendor name (usually at the top of invoice)
        invoice.vendor_name = self._extract_vendor_name(lines, text)

        # Extract TRN (Tax Registration Number)
        invoice.vendor_trn = self._extract_trn(text)

        # Extract date
        invoice.invoice_date = self._extract_date(text)

        # Extract project name (from stamp)
        invoice.project_name = self._extract_project(text)

        # Extract items
        invoice.items = self._extract_items(lines, text)

        # Extract totals
        invoice.subtotal = self._extract_amount(text, ['subtotal', 'sub total', 'sub-total'])
        invoice.vat_amount = self._extract_amount(text, ['vat', 'tax'])
        invoice.discount = self._extract_amount(text, ['discount', 'disc'])
        invoice.total_amount = self._extract_amount(text, ['total', 'net amount', 'grand total', 'amount due'])

        # If no project found in stamp, add note
        if not invoice.project_name:
            invoice.extraction_notes.append("No project stamp detected. Please assign project manually.")

        return invoice

    def _extract_vendor_name(self, lines: List[str], text: str) -> str:
        """Extract vendor/company name from invoice"""
        # Known vendor patterns from UAE invoices
        known_vendors = [
            r'(Media\s*General\s*Trading\s*(?:LLC|L\.L\.C)?)',
            r'(Al\s*Junaibi\s*Building\s*Materials)',
            r'(Laspinas\s*Building\s*Materials)',
            r'(Mohammed\s*Sofa\s*(?:Elect(?:rical)?(?:\s*&)?\s*Sanitary\s*Ware)?)',
            r'([A-Za-z\s]+Building\s*Materials)',
            r'([A-Za-z\s]+Trading\s*(?:LLC|L\.L\.C)?)',
        ]

        for pattern in known_vendors:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 5:
                    return name

        # Common patterns for company names
        patterns = [
            r'(?:from|vendor|supplier|company)[\s:]+([A-Za-z\s&\.\-]+(?:LLC|L\.L\.C|Trading|Materials|Services|Est|CO|Company)?)',
            r'^([A-Z][A-Za-z\s&\.\-]+(?:LLC|L\.L\.C|Trading|Materials|Services|Est|CO|Company))',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if not any(fp in name.upper() for fp in ['STAMP', 'SIGNATURE', 'AUTHORISED', 'CUSTOMER', 'ADDRESS']):
                    return name

        # Fallback: Look for LLC or Trading in first few lines
        for line in lines[:15]:
            if any(keyword in line.upper() for keyword in ['LLC', 'L.L.C', 'TRADING', 'MATERIALS', 'EST.', 'CONTRACTING']):
                # Clean up the line
                name = re.sub(r'[\(\)]', '', line)
                name = re.sub(r'\s+', ' ', name).strip()
                # Filter out false positives
                if len(name) > 5 and not any(fp in name.upper() for fp in ['STAMP', 'SIGNATURE', 'AUTHORISED', 'CUSTOMER']):
                    return name

        return ""

    def _extract_trn(self, text: str) -> str:
        """Extract Tax Registration Number"""
        patterns = [
            r'TRN[\s:]*(\d{15})',
            r'Tax\s*Reg(?:istration)?[\s#:]*(\d{15})',
            r'(\d{15})',  # UAE TRN is 15 digits
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                trn = match.group(1)
                if len(trn) == 15:
                    return trn
        return ""

    def _extract_date(self, text: str) -> str:
        """Extract invoice date"""
        # First try patterns with explicit "Date" keyword
        date_patterns_with_keyword = [
            r'(?:date|dated|invoice\s*date|inv\s*date)[\s:\.]*(\d{1,2}[-/\.]\s*[A-Za-z]{3}[-/\.]\s*\d{4})',
            r'(?:date|dated|invoice\s*date|inv\s*date)[\s:\.]*(\d{1,2}[-/\.]\s*[A-Za-z]{3}[-/\.]\s*\d{2})',
            r'(?:date|dated|invoice\s*date|inv\s*date)[\s:\.]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
            r'(?:date|dated|invoice\s*date|inv\s*date)[\s:\.]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2})',
        ]

        for pattern in date_patterns_with_keyword:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).replace(' ', '')
                normalized = self._normalize_date(date_str)
                if normalized and self._is_valid_date(normalized):
                    return normalized

        # Then try standalone date patterns (more restrictive)
        standalone_patterns = [
            r'(\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})',  # 27-Mar-2024
            r'(\d{1,2}[-/][A-Za-z]{3}[-/]\d{2})\b',  # 27-Mar-24
        ]

        for pattern in standalone_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for date_str in matches:
                normalized = self._normalize_date(date_str)
                if normalized and self._is_valid_date(normalized):
                    return normalized

        return ""

    def _is_valid_date(self, date_str: str) -> bool:
        """Check if date is reasonable (not too far in past or future)"""
        try:
            dt = datetime.strptime(date_str, '%d-%b-%Y')
            year = dt.year
            # Accept dates from 2020 to 2030
            return 2020 <= year <= 2030
        except ValueError:
            return False

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to Tally format (DD-MMM-YYYY)"""
        formats = [
            '%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d/%m/%y',
            '%d-%b-%Y', '%d/%b/%Y', '%d-%b-%y', '%d/%b/%y',
            '%d-%B-%Y', '%d/%B/%Y',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%d-%b-%Y')
            except ValueError:
                continue

        return date_str

    def _extract_project(self, text: str) -> str:
        """Extract project name from stamp or text"""
        patterns = [
            r'project[\s:]*([A-Za-z0-9\s\-_]+)',
            r'proj[\s:]*([A-Za-z0-9\s\-_]+)',
            r'site[\s:]*([A-Za-z0-9\s\-_]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project = match.group(1).strip()
                # Clean up - remove common noise words
                project = re.sub(r'\b(invoice|date|no|number)\b', '', project, flags=re.IGNORECASE)
                project = project.strip()
                if len(project) > 2:
                    return project
        return ""

    def _extract_items(self, lines: List[str], text: str) -> List[InvoiceItem]:
        """Extract line items from invoice"""
        items = []

        # Look for item patterns
        # Pattern: Item code (optional), Description, Qty, Unit, Rate, Amount
        item_pattern = r'(\d{4,6})?\s*([A-Za-z][A-Za-z\s\-\d\.]+?)\s+(\d+\.?\d*)\s*(NOS|PCS|KG|MTR|LTR|DRUM|BOX|SET|EA|EACH|UNIT)?\s*(\d+\.?\d*)\s+(\d+\.?\d*)'

        matches = re.findall(item_pattern, text, re.IGNORECASE)

        for match in matches:
            item_code, description, qty, unit, rate, amount = match

            # Validate - skip if it looks like a total or header
            if any(skip in description.upper() for skip in ['TOTAL', 'SUBTOTAL', 'VAT', 'TAX', 'DISCOUNT']):
                continue

            try:
                item = InvoiceItem(
                    item_code=item_code.strip() if item_code else "",
                    description=description.strip(),
                    quantity=float(qty) if qty else 0,
                    unit=unit.upper() if unit else "NOS",
                    rate=float(rate) if rate else 0,
                    amount=float(amount) if amount else 0,
                )

                # Validate the item
                if item.description and (item.quantity > 0 or item.amount > 0):
                    items.append(item)
            except (ValueError, AttributeError):
                continue

        # Alternative parsing for simpler invoices
        if not items:
            items = self._extract_items_simple(lines)

        return items

    def _extract_items_simple(self, lines: List[str]) -> List[InvoiceItem]:
        """Simple item extraction for handwritten invoices"""
        items = []

        for line in lines:
            # Skip header/footer lines
            if any(skip in line.upper() for skip in ['TOTAL', 'SUBTOTAL', 'VAT', 'TAX', 'DATE', 'INVOICE', 'TRN']):
                continue

            # Look for lines with numbers (qty, rate, amount)
            numbers = re.findall(r'(\d+\.?\d*)', line)
            if len(numbers) >= 2:
                # Extract description (text before numbers)
                desc_match = re.match(r'^([A-Za-z][A-Za-z\s\-\.]+)', line)
                if desc_match:
                    description = desc_match.group(1).strip()
                    if len(description) > 3:
                        try:
                            item = InvoiceItem(
                                description=description,
                                quantity=float(numbers[0]) if len(numbers) > 0 else 0,
                                rate=float(numbers[-2]) if len(numbers) > 1 else 0,
                                amount=float(numbers[-1]) if len(numbers) > 0 else 0,
                            )
                            items.append(item)
                        except ValueError:
                            continue

        return items

    def _extract_amount(self, text: str, keywords: List[str]) -> float:
        """Extract an amount value based on keywords"""
        for keyword in keywords:
            patterns = [
                rf'{keyword}[\s:]*AED?\s*(\d+[,\d]*\.?\d*)',
                rf'{keyword}[\s:]*(\d+[,\d]*\.?\d*)',
                rf'(\d+[,\d]*\.?\d*)\s*{keyword}',
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        return float(amount_str)
                    except ValueError:
                        continue
        return 0.0


def process_invoice(file_path: str) -> InvoiceData:
    """
    Convenience function to process a single invoice
    """
    ocr = InvoiceOCR()
    return ocr.extract_from_file(file_path)


def process_invoice_folder(folder_path: str) -> List[InvoiceData]:
    """
    Process all invoices in a folder
    """
    from config import SUPPORTED_FORMATS

    ocr = InvoiceOCR()
    results = []

    for filename in os.listdir(folder_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_FORMATS:
            file_path = os.path.join(folder_path, filename)
            try:
                invoice_data = ocr.extract_from_file(file_path)
                results.append(invoice_data)
                logger.info(f"Processed: {filename} (confidence: {invoice_data.ocr_confidence:.1f}%)")
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

    return results


if __name__ == "__main__":
    # Test with sample invoice
    import sys
    if len(sys.argv) > 1:
        result = process_invoice(sys.argv[1])
        print(f"Vendor: {result.vendor_name}")
        print(f"Voucher #: {result.voucher_number}")
        print(f"Date: {result.invoice_date}")
        print(f"Project: {result.project_name}")
        print(f"Items: {len(result.items)}")
        for item in result.items:
            print(f"  - {item.description}: {item.quantity} x {item.rate} = {item.amount}")
        print(f"Total: {result.total_amount}")
        print(f"VAT: {result.vat_amount}")
        print(f"Confidence: {result.ocr_confidence:.1f}%")
        print(f"Needs Review: {result.needs_review}")
