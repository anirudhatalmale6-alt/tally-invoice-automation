"""
Tally Invoice Automation - Main Application
Automates data entry from scanned invoices into Tally ERP 9

Usage:
    python main.py                      # Process all invoices in configured folder
    python main.py invoice.pdf          # Process single invoice
    python main.py /path/to/folder      # Process all invoices in specific folder
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict

from config import (
    INVOICE_FOLDER, SUPPORTED_FORMATS, OUTPUT_FOLDER, FLAGGED_FOLDER,
    LOG_FILE, LOG_LEVEL, OCR_CONFIDENCE_THRESHOLD
)
from ocr_extractor import InvoiceOCR, InvoiceData, process_invoice
from tally_api import TallyAPI, TallyConnectionError
from voucher_creator import PurchaseVoucherCreator, VoucherResult

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InvoiceProcessor:
    """
    Main processor class that orchestrates the entire workflow:
    1. Read invoice files
    2. Extract data using OCR
    3. Create masters (ledgers, items, projects) if needed
    4. Create purchase vouchers in Tally
    5. Flag unclear invoices for manual review
    """

    def __init__(self, invoice_folder: str = None):
        self.invoice_folder = invoice_folder or INVOICE_FOLDER
        self.output_folder = os.path.join(self.invoice_folder, OUTPUT_FOLDER)
        self.flagged_folder = os.path.join(self.invoice_folder, FLAGGED_FOLDER)

        # Create output folders
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.flagged_folder, exist_ok=True)

        # Initialize components
        self.ocr = None
        self.tally_api = None
        self.voucher_creator = None

        # Processing statistics
        self.stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'flagged': 0,
            'ledgers_created': 0,
            'items_created': 0,
            'projects_created': 0,
        }

        # Results for reporting
        self.results: List[Dict[str, Any]] = []

    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing Invoice Processor...")

        # Initialize OCR
        try:
            self.ocr = InvoiceOCR()
            logger.info("OCR engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            return False

        # Initialize Tally connection
        try:
            self.tally_api = TallyAPI()
            if not self.tally_api.test_connection():
                logger.warning("Cannot connect to Tally. Running in extraction-only mode.")
                self.tally_api = None
            else:
                logger.info("Connected to Tally successfully")
                self.voucher_creator = PurchaseVoucherCreator(self.tally_api)
        except TallyConnectionError as e:
            logger.warning(f"Tally connection failed: {e}. Running in extraction-only mode.")
            self.tally_api = None

        return True

    def get_invoice_files(self) -> List[str]:
        """Get list of invoice files to process"""
        if not os.path.exists(self.invoice_folder):
            logger.error(f"Invoice folder not found: {self.invoice_folder}")
            return []

        files = []
        for filename in os.listdir(self.invoice_folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_FORMATS:
                file_path = os.path.join(self.invoice_folder, filename)
                files.append(file_path)

        logger.info(f"Found {len(files)} invoice files to process")
        return sorted(files)

    def process_single_invoice(self, file_path: str) -> Dict[str, Any]:
        """Process a single invoice file"""
        filename = os.path.basename(file_path)
        result = {
            'file': filename,
            'voucher_number': '',
            'vendor': '',
            'date': '',
            'total': 0,
            'items_count': 0,
            'ocr_confidence': 0,
            'status': 'pending',
            'message': '',
            'needs_review': False,
            'extraction_notes': [],
            'created_ledgers': [],
            'created_items': [],
            'created_projects': [],
        }

        try:
            # Step 1: Extract data using OCR
            logger.info(f"Processing: {filename}")
            invoice_data = self.ocr.extract_from_file(file_path)

            result['voucher_number'] = invoice_data.voucher_number
            result['vendor'] = invoice_data.vendor_name
            result['date'] = invoice_data.invoice_date
            result['total'] = invoice_data.total_amount
            result['items_count'] = len(invoice_data.items)
            result['ocr_confidence'] = invoice_data.ocr_confidence
            result['needs_review'] = invoice_data.needs_review
            result['extraction_notes'] = invoice_data.extraction_notes

            # Log extraction results
            logger.info(f"  Vendor: {invoice_data.vendor_name}")
            logger.info(f"  Date: {invoice_data.invoice_date}")
            logger.info(f"  Items: {len(invoice_data.items)}")
            logger.info(f"  Total: {invoice_data.total_amount}")
            logger.info(f"  OCR Confidence: {invoice_data.ocr_confidence:.1f}%")

            # Step 2: Check if needs manual review
            if invoice_data.needs_review:
                result['status'] = 'flagged'
                result['message'] = 'Low OCR confidence - needs manual review'
                self._move_to_flagged(file_path, invoice_data)
                self.stats['flagged'] += 1
                logger.warning(f"  Flagged for manual review: {filename}")
                return result

            # Step 3: Create voucher in Tally (if connected)
            if self.tally_api and self.voucher_creator:
                voucher_result = self.voucher_creator.create_voucher(invoice_data)

                result['created_ledgers'] = voucher_result.created_ledgers
                result['created_items'] = voucher_result.created_items
                result['created_projects'] = voucher_result.created_projects

                self.stats['ledgers_created'] += len(voucher_result.created_ledgers)
                self.stats['items_created'] += len(voucher_result.created_items)
                self.stats['projects_created'] += len(voucher_result.created_projects)

                if voucher_result.success:
                    result['status'] = 'success'
                    result['message'] = voucher_result.message
                    self._move_to_processed(file_path)
                    self.stats['successful'] += 1
                    logger.info(f"  Voucher created successfully")
                else:
                    result['status'] = 'failed'
                    result['message'] = voucher_result.message
                    self.stats['failed'] += 1
                    logger.error(f"  Failed to create voucher: {voucher_result.message}")
            else:
                # Extraction only mode
                result['status'] = 'extracted'
                result['message'] = 'Data extracted (Tally not connected)'
                self._save_extracted_data(invoice_data)
                logger.info(f"  Data extracted and saved")

        except Exception as e:
            result['status'] = 'error'
            result['message'] = str(e)
            self.stats['failed'] += 1
            logger.error(f"  Error processing {filename}: {e}")

        return result

    def process_all(self) -> Dict[str, Any]:
        """Process all invoices in the folder"""
        if not self.initialize():
            return {'error': 'Failed to initialize processor'}

        files = self.get_invoice_files()
        self.stats['total'] = len(files)

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting batch processing of {len(files)} invoices")
        logger.info(f"{'='*60}\n")

        for file_path in files:
            result = self.process_single_invoice(file_path)
            self.results.append(result)

        # Generate summary report
        summary = self._generate_summary()
        self._save_report()

        return summary

    def _move_to_processed(self, file_path: str):
        """Move processed file to output folder"""
        try:
            filename = os.path.basename(file_path)
            dest = os.path.join(self.output_folder, filename)
            shutil.move(file_path, dest)
        except Exception as e:
            logger.warning(f"Could not move file to processed: {e}")

    def _move_to_flagged(self, file_path: str, invoice_data: InvoiceData):
        """Move flagged file and save extraction data for review"""
        try:
            filename = os.path.basename(file_path)
            dest = os.path.join(self.flagged_folder, filename)
            shutil.copy(file_path, dest)

            # Save extracted data for review
            json_path = os.path.join(self.flagged_folder, f"{invoice_data.voucher_number}_data.json")
            with open(json_path, 'w') as f:
                data = {
                    'voucher_number': invoice_data.voucher_number,
                    'vendor_name': invoice_data.vendor_name,
                    'invoice_date': invoice_data.invoice_date,
                    'items': [asdict(item) for item in invoice_data.items],
                    'total_amount': invoice_data.total_amount,
                    'vat_amount': invoice_data.vat_amount,
                    'ocr_confidence': invoice_data.ocr_confidence,
                    'extraction_notes': invoice_data.extraction_notes,
                    'raw_text': invoice_data.raw_text[:2000],  # First 2000 chars
                }
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Could not move file to flagged: {e}")

    def _save_extracted_data(self, invoice_data: InvoiceData):
        """Save extracted data when Tally is not connected"""
        try:
            json_path = os.path.join(self.output_folder, f"{invoice_data.voucher_number}_data.json")
            with open(json_path, 'w') as f:
                data = {
                    'voucher_number': invoice_data.voucher_number,
                    'vendor_name': invoice_data.vendor_name,
                    'vendor_trn': invoice_data.vendor_trn,
                    'invoice_date': invoice_data.invoice_date,
                    'items': [asdict(item) for item in invoice_data.items],
                    'subtotal': invoice_data.subtotal,
                    'vat_amount': invoice_data.vat_amount,
                    'discount': invoice_data.discount,
                    'total_amount': invoice_data.total_amount,
                    'project_name': invoice_data.project_name,
                    'attachment_path': invoice_data.attachment_path,
                    'ocr_confidence': invoice_data.ocr_confidence,
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save extracted data: {e}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate processing summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'success_rate': (self.stats['successful'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0,
            'results': self.results,
        }

        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total invoices:     {self.stats['total']}")
        logger.info(f"Successful:         {self.stats['successful']}")
        logger.info(f"Failed:             {self.stats['failed']}")
        logger.info(f"Flagged for review: {self.stats['flagged']}")
        logger.info(f"Ledgers created:    {self.stats['ledgers_created']}")
        logger.info(f"Items created:      {self.stats['items_created']}")
        logger.info(f"Projects created:   {self.stats['projects_created']}")
        logger.info(f"Success rate:       {summary['success_rate']:.1f}%")
        logger.info(f"{'='*60}\n")

        return summary

    def _save_report(self):
        """Save detailed report to JSON file"""
        report_path = os.path.join(self.invoice_folder, f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.stats,
                    'results': self.results,
                }, f, indent=2)
            logger.info(f"Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Could not save report: {e}")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("TALLY INVOICE AUTOMATION")
    print("Automated Purchase Voucher Entry from Scanned Invoices")
    print("="*60 + "\n")

    # Determine input
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

        if os.path.isfile(input_path):
            # Process single file
            processor = InvoiceProcessor(os.path.dirname(input_path))
            if processor.initialize():
                result = processor.process_single_invoice(input_path)
                print(f"\nResult: {result['status']}")
                print(f"Message: {result['message']}")
                if result['items_count'] > 0:
                    print(f"Vendor: {result['vendor']}")
                    print(f"Date: {result['date']}")
                    print(f"Items: {result['items_count']}")
                    print(f"Total: {result['total']}")

        elif os.path.isdir(input_path):
            # Process folder
            processor = InvoiceProcessor(input_path)
            processor.process_all()

        else:
            print(f"Error: Path not found: {input_path}")
            sys.exit(1)
    else:
        # Use default folder from config
        processor = InvoiceProcessor()
        processor.process_all()


if __name__ == "__main__":
    main()
