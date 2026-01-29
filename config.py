"""
Configuration settings for Tally Invoice Automation
"""

# Tally ERP 9 Connection Settings
TALLY_HOST = "localhost"
TALLY_PORT = 9000  # Default Tally port

# Invoice Processing Settings
INVOICE_FOLDER = r"C:\Invoices"  # Folder where scanned invoices are stored
SUPPORTED_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']

# Tally Company Settings (Update these based on your Tally setup)
COMPANY_NAME = "MOHD WASEEM BUILDING CONTRACTING (L.L.C)"

# VAT Settings (UAE)
VAT_RATE = 5.0  # 5% VAT in UAE
VAT_LEDGER_NAME = "VAT"

# Default Ledger Groups for auto-creation
SUNDRY_CREDITOR_GROUP = "Sundry Creditors"
STOCK_ITEM_GROUP = "Primary"
STOCK_CATEGORY = "Primary"
STOCK_UNIT = "Nos"  # Default unit

# OCR Settings
OCR_CONFIDENCE_THRESHOLD = 60  # Minimum confidence % to accept OCR result
FLAG_LOW_CONFIDENCE = True  # Flag invoices with low OCR confidence

# Logging
LOG_FILE = "tally_automation.log"
LOG_LEVEL = "INFO"

# Output Settings
OUTPUT_FOLDER = "processed"
FLAGGED_FOLDER = "flagged"  # Invoices that need manual review
