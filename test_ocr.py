import sys

# Try table extractor first
try:
    from table_extractor import process_invoice
    print("Using TABLE DETECTION OCR")
except ImportError:
    from ocr_extractor import process_invoice
    print("Using standard OCR")

# Update this path to your invoice location
invoice_path = r"C:\Users\ziad halabieh\Desktop\sample of scanned invoice with their voucher number as their file name\1037.pdf"

# Allow command line override
if len(sys.argv) > 1:
    invoice_path = sys.argv[1]

print(f"Processing: {invoice_path}")
print("="*50)

result = process_invoice(invoice_path)

print("EXTRACTION RESULTS:")
print("="*50)
print(f"Vendor: {result.vendor_name}")
print(f"Date: {result.invoice_date}")
print(f"TRN: {result.vendor_trn}")
print(f"Project: {result.project_name}")
print(f"Confidence: {result.ocr_confidence:.1f}%")
print("")
print(f"Items found: {len(result.items)}")
print("-"*50)
for i, item in enumerate(result.items, 1):
    print(f"{i}. {item.item_code} | {item.description}")
    print(f"   Qty: {item.quantity} {item.unit} | Rate: {item.rate} | Amount: {item.amount}")
print("-"*50)
print(f"Subtotal: {result.subtotal}")
print(f"VAT: {result.vat_amount}")
print(f"Discount: {result.discount}")
print(f"Total: {result.total_amount}")
print("="*50)

if result.needs_review:
    print("WARNING: Manual review recommended")
    for note in result.extraction_notes:
        print(f"  - {note}")
