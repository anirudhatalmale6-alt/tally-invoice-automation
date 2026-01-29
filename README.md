# Tally Invoice Automation

Automatically extract data from scanned invoices (PDF/images) and create purchase vouchers in Tally ERP 9.

## Features

- **OCR Extraction**: Handles both printed and handwritten invoices using EasyOCR and Tesseract
- **Auto-create Masters**: Automatically creates vendor ledgers, stock items, and projects if they don't exist
- **VAT Support**: Handles UAE 5% VAT calculation
- **Project Allocation**: Assigns all items to a project (from stamped project name)
- **Attachment Linking**: Stores invoice file path in Tally voucher
- **Flagging System**: Low-confidence OCR results are flagged for manual review
- **Batch Processing**: Process entire folders of invoices at once

## Requirements

### Software
- Python 3.8 or higher
- Tally ERP 9 version 6.6.3 (with ODBC server enabled)
- Tesseract OCR (optional, for better printed text recognition)
- Poppler (for PDF processing)

### Installation

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR (Windows):**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install and add to PATH
   - Or set path in code: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

3. **Install Poppler (Windows):**
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract and add `bin` folder to PATH

4. **Enable Tally ODBC Server:**
   - Open Tally ERP 9
   - Go to: Gateway of Tally > F12: Configure > Advanced Configuration
   - Set "Enable ODBC Server" to Yes
   - Default port is 9000

## Configuration

Edit `config.py` to match your setup:

```python
# Tally Connection
TALLY_HOST = "localhost"
TALLY_PORT = 9000

# Your company name in Tally (must match exactly)
COMPANY_NAME = "MOHD WASEEM BUILDING CONTRACTING (L.L.C)"

# Folder where scanned invoices are stored
INVOICE_FOLDER = r"C:\Invoices"
```

## Usage

### Process all invoices in configured folder:
```bash
python main.py
```

### Process a single invoice:
```bash
python main.py "C:\Invoices\1037.pdf"
```

### Process a specific folder:
```bash
python main.py "D:\MyInvoices"
```

## Invoice Requirements

For best OCR results:
1. **Voucher number**: File name should be the voucher number (e.g., `1037.pdf`)
2. **Project name**: Stamp the project name on the invoice (same stamp location for all)
3. **Quality**: Scan at 300 DPI minimum for clear text
4. **Format**: PDF, JPG, PNG, TIFF supported

## Output

- **Processed invoices**: Moved to `processed/` subfolder
- **Flagged invoices**: Copied to `flagged/` subfolder with extracted data JSON
- **Processing report**: JSON file with full results

## Data Extracted

From each invoice:
- Vendor/Company name
- Invoice date
- TRN (Tax Registration Number)
- Line items (code, description, quantity, unit, rate, amount)
- VAT amount
- Discount
- Total amount
- Project name (from stamp)

## Troubleshooting

### "Cannot connect to Tally"
- Ensure Tally is running
- Check ODBC server is enabled (F12 > Configure > Advanced)
- Verify port number (default 9000)
- Check firewall settings

### Low OCR confidence
- Use higher scan resolution (300+ DPI)
- Ensure good lighting when scanning
- Keep invoices flat and aligned
- Handwritten invoices may need manual review

### Items not matching
- Check if unit of measure exists in Tally
- Verify stock item names don't have special characters

## Support

For issues or customization requests, contact the developer.
