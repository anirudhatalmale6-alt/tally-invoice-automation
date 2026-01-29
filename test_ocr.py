from ocr_extractor import process_invoice

invoice_path = r"C:\Users\ziad halabieh\Desktop\sample of scanned invoice with their voucher number as their file name\1037.pdf"

result = process_invoice(invoice_path)

print("RAW OCR TEXT:")
print(result.raw_text[:3000])
print("Vendor:", result.vendor_name)
print("Date:", result.invoice_date)
print("Items found:", len(result.items))
for item in result.items:
    print("  -", item.item_code, item.description, item.quantity, item.rate, item.amount)
print("Total:", result.total_amount)
