"""
Purchase Voucher Creator Module
Creates purchase vouchers in Tally ERP 9 from extracted invoice data
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

from config import (
    COMPANY_NAME, VAT_LEDGER_NAME, VAT_RATE,
    SUNDRY_CREDITOR_GROUP, STOCK_ITEM_GROUP, STOCK_UNIT,
    INVOICE_FOLDER
)
from ocr_extractor import InvoiceData, InvoiceItem
from tally_api import TallyAPI, TallyConnectionError, TallyAPIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VoucherResult:
    """Result of voucher creation attempt"""
    success: bool
    voucher_number: str
    message: str
    created_ledgers: List[str]
    created_items: List[str]
    created_projects: List[str]


class PurchaseVoucherCreator:
    """
    Creates purchase vouchers in Tally from invoice data
    Handles auto-creation of ledgers, stock items, and projects
    """

    def __init__(self, tally_api: TallyAPI = None):
        self.api = tally_api or TallyAPI()
        self.created_ledgers = []
        self.created_items = []
        self.created_projects = []

    def create_voucher(self, invoice: InvoiceData) -> VoucherResult:
        """
        Create a purchase voucher from invoice data
        """
        self.created_ledgers = []
        self.created_items = []
        self.created_projects = []

        try:
            # 1. Ensure vendor ledger exists
            if invoice.vendor_name:
                self._ensure_ledger_exists(
                    invoice.vendor_name,
                    SUNDRY_CREDITOR_GROUP,
                    invoice.vendor_address,
                    invoice.vendor_trn
                )

            # 2. Ensure VAT ledger exists
            self._ensure_vat_ledger_exists()

            # 3. Ensure all stock items exist
            for item in invoice.items:
                if item.description:
                    item_name = self._format_item_name(item)
                    self._ensure_stock_item_exists(item_name, item.unit)

            # 4. Ensure project exists
            if invoice.project_name:
                self._ensure_project_exists(invoice.project_name)

            # 5. Create the purchase voucher
            xml = self._build_voucher_xml(invoice)
            response = self.api._send_request(xml)

            # Check response for success
            if 'CREATED' in response.upper() or ('LINEERROR' not in response.upper() and 'ERROR' not in response.upper()):
                return VoucherResult(
                    success=True,
                    voucher_number=invoice.voucher_number,
                    message=f"Purchase voucher {invoice.voucher_number} created successfully",
                    created_ledgers=self.created_ledgers.copy(),
                    created_items=self.created_items.copy(),
                    created_projects=self.created_projects.copy()
                )
            else:
                return VoucherResult(
                    success=False,
                    voucher_number=invoice.voucher_number,
                    message=f"Failed to create voucher: {response[:500]}",
                    created_ledgers=self.created_ledgers.copy(),
                    created_items=self.created_items.copy(),
                    created_projects=self.created_projects.copy()
                )

        except (TallyConnectionError, TallyAPIError) as e:
            return VoucherResult(
                success=False,
                voucher_number=invoice.voucher_number,
                message=str(e),
                created_ledgers=self.created_ledgers.copy(),
                created_items=self.created_items.copy(),
                created_projects=self.created_projects.copy()
            )

    def _ensure_ledger_exists(self, name: str, group: str,
                               address: str = "", trn: str = "") -> bool:
        """Ensure a ledger exists, create if not"""
        if not self.api.ledger_exists(name):
            if self.api.create_ledger(name, group, address, trn):
                self.created_ledgers.append(name)
                logger.info(f"Auto-created ledger: {name}")
                return True
            else:
                logger.warning(f"Failed to create ledger: {name}")
                return False
        return True

    def _ensure_vat_ledger_exists(self) -> bool:
        """Ensure VAT ledger exists"""
        if not self.api.ledger_exists(VAT_LEDGER_NAME):
            xml = f"""
            <ENVELOPE>
                <HEADER>
                    <TALLYREQUEST>Import Data</TALLYREQUEST>
                </HEADER>
                <BODY>
                    <IMPORTDATA>
                        <REQUESTDESC>
                            <REPORTNAME>All Masters</REPORTNAME>
                            <STATICVARIABLES>
                                <SVCURRENTCOMPANY>{COMPANY_NAME}</SVCURRENTCOMPANY>
                            </STATICVARIABLES>
                        </REQUESTDESC>
                        <REQUESTDATA>
                            <TALLYMESSAGE xmlns:UDF="TallyUDF">
                                <LEDGER NAME="{VAT_LEDGER_NAME}" ACTION="Create">
                                    <NAME>{VAT_LEDGER_NAME}</NAME>
                                    <PARENT>Duties &amp; Taxes</PARENT>
                                    <TAXTYPE>VAT</TAXTYPE>
                                    <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
                                </LEDGER>
                            </TALLYMESSAGE>
                        </REQUESTDATA>
                    </IMPORTDATA>
                </BODY>
            </ENVELOPE>
            """
            try:
                self.api._send_request(xml)
                self.created_ledgers.append(VAT_LEDGER_NAME)
                return True
            except:
                pass
        return True

    def _ensure_stock_item_exists(self, name: str, unit: str = "Nos") -> bool:
        """Ensure a stock item exists, create if not"""
        # Normalize unit
        unit = unit.upper() if unit else STOCK_UNIT

        if not self.api.stock_item_exists(name):
            # First ensure the unit exists
            self.api.create_unit(unit)

            if self.api.create_stock_item(name, unit, STOCK_ITEM_GROUP):
                self.created_items.append(name)
                logger.info(f"Auto-created stock item: {name}")
                return True
            else:
                logger.warning(f"Failed to create stock item: {name}")
                return False
        return True

    def _ensure_project_exists(self, name: str) -> bool:
        """Ensure a project (cost centre) exists, create if not"""
        if not self.api.project_exists(name):
            if self.api.create_project(name):
                self.created_projects.append(name)
                logger.info(f"Auto-created project: {name}")
                return True
            else:
                logger.warning(f"Failed to create project: {name}")
                return False
        return True

    def _format_item_name(self, item: InvoiceItem) -> str:
        """Format item name for Tally"""
        name = item.description.strip()

        # If there's an item code, prepend it
        if item.item_code:
            name = f"{item.item_code} {name}"

        # Clean up the name
        name = name.replace('&', '&amp;')

        return name

    def _build_voucher_xml(self, invoice: InvoiceData) -> str:
        """Build the XML request for creating a purchase voucher"""

        # Build inventory entries for each item
        inventory_entries = ""
        for item in invoice.items:
            item_name = self._format_item_name(item)
            unit = item.unit.upper() if item.unit else STOCK_UNIT

            # Build cost centre allocation if project specified
            cost_centre_xml = ""
            if invoice.project_name:
                cost_centre_xml = f"""
                            <CATEGORYALLOCATIONS.LIST>
                                <CATEGORY></CATEGORY>
                                <COSTCENTREALLOCATIONS.LIST>
                                    <NAME>{invoice.project_name}</NAME>
                                    <AMOUNT>{item.amount}</AMOUNT>
                                </COSTCENTREALLOCATIONS.LIST>
                            </CATEGORYALLOCATIONS.LIST>
                """

            inventory_entries += f"""
                        <ALLINVENTORYENTRIES.LIST>
                            <STOCKITEMNAME>{item_name}</STOCKITEMNAME>
                            <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>
                            <ISLASTDEEMEDPOSITIVE>Yes</ISLASTDEEMEDPOSITIVE>
                            <ISAUTONEGATE>No</ISAUTONEGATE>
                            <RATE>{item.rate}/{unit}</RATE>
                            <AMOUNT>-{item.amount}</AMOUNT>
                            <ACTUALQTY>{item.quantity} {unit}</ACTUALQTY>
                            <BILLEDQTY>{item.quantity} {unit}</BILLEDQTY>
                            <BATCHALLOCATIONS.LIST>
                                <GODOWNNAME>Main Location</GODOWNNAME>
                                <BATCHNAME>Primary Batch</BATCHNAME>
                                <AMOUNT>-{item.amount}</AMOUNT>
                                <ACTUALQTY>{item.quantity} {unit}</ACTUALQTY>
                                <BILLEDQTY>{item.quantity} {unit}</BILLEDQTY>
                            </BATCHALLOCATIONS.LIST>
                            {cost_centre_xml}
                            <ACCOUNTINGALLOCATIONS.LIST>
                                <LEDGERNAME>Purchase</LEDGERNAME>
                                <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>
                                <AMOUNT>-{item.amount}</AMOUNT>
                            </ACCOUNTINGALLOCATIONS.LIST>
                        </ALLINVENTORYENTRIES.LIST>
            """

        # Build ledger entries
        # Vendor credit entry
        vendor_entry = f"""
                        <ALLLEDGERENTRIES.LIST>
                            <LEDGERNAME>{invoice.vendor_name}</LEDGERNAME>
                            <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
                            <AMOUNT>{invoice.total_amount}</AMOUNT>
                        </ALLLEDGERENTRIES.LIST>
        """

        # VAT entry if applicable
        vat_entry = ""
        if invoice.vat_amount > 0:
            vat_entry = f"""
                        <ALLLEDGERENTRIES.LIST>
                            <LEDGERNAME>{VAT_LEDGER_NAME}</LEDGERNAME>
                            <ISDEEMEDPOSITIVE>Yes</ISDEEMEDPOSITIVE>
                            <AMOUNT>-{invoice.vat_amount}</AMOUNT>
                        </ALLLEDGERENTRIES.LIST>
            """

        # Discount entry if applicable
        discount_entry = ""
        if invoice.discount > 0:
            discount_entry = f"""
                        <ALLLEDGERENTRIES.LIST>
                            <LEDGERNAME>Discount Received</LEDGERNAME>
                            <ISDEEMEDPOSITIVE>No</ISDEEMEDPOSITIVE>
                            <AMOUNT>{invoice.discount}</AMOUNT>
                        </ALLLEDGERENTRIES.LIST>
            """

        # Build attachment path
        attachment_path = invoice.attachment_path
        if not attachment_path.startswith(INVOICE_FOLDER):
            # Use the configured invoice folder + voucher number
            attachment_path = f"{INVOICE_FOLDER}\\{invoice.voucher_number}.pdf"

        # Build the complete voucher XML
        xml = f"""
        <ENVELOPE>
            <HEADER>
                <TALLYREQUEST>Import Data</TALLYREQUEST>
            </HEADER>
            <BODY>
                <IMPORTDATA>
                    <REQUESTDESC>
                        <REPORTNAME>Vouchers</REPORTNAME>
                        <STATICVARIABLES>
                            <SVCURRENTCOMPANY>{COMPANY_NAME}</SVCURRENTCOMPANY>
                        </STATICVARIABLES>
                    </REQUESTDESC>
                    <REQUESTDATA>
                        <TALLYMESSAGE xmlns:UDF="TallyUDF">
                            <VOUCHER VCHTYPE="Purchase" ACTION="Create">
                                <DATE>{invoice.invoice_date.replace('-', '')}</DATE>
                                <VOUCHERTYPENAME>Purchase</VOUCHERTYPENAME>
                                <VOUCHERNUMBER>{invoice.voucher_number}</VOUCHERNUMBER>
                                <PARTYLEDGERNAME>{invoice.vendor_name}</PARTYLEDGERNAME>
                                <REFERENCE>{invoice.voucher_number}</REFERENCE>
                                <NARRATION>Tax Invoice - Auto imported from {invoice.voucher_number}.pdf</NARRATION>
                                <ISINVOICE>Yes</ISINVOICE>
                                <PERSISTEDVIEW>Invoice Voucher View</PERSISTEDVIEW>
                                <EFFECTIVEDATE>{invoice.invoice_date.replace('-', '')}</EFFECTIVEDATE>
                                <BASICBASEPARTYNAME>{invoice.vendor_name}</BASICBASEPARTYNAME>
                                <FBTPAYMENTTYPE>Default</FBTPAYMENTTYPE>
                                <VCHGSTCLASS></VCHGSTCLASS>
                                <DIFFACTUALQTY>No</DIFFACTUALQTY>
                                <ISMSTFROMSYNC>No</ISMSTFROMSYNC>
                                <ASORIGINAL>Yes</ASORIGINAL>
                                <AUDESSION>No</AUDESSION>
                                <ISVOUCHERDELETED>No</ISVOUCHERDELETED>
                                <ISREALLEDGER>No</ISREALLEDGER>
                                <VATPAYMENTSTATUS>Unpaid</VATPAYMENTSTATUS>
                                <CHANGEVCHMODE>No</CHANGEVCHMODE>
                                <USEFORJOBCOSTING>No</USEFORJOBCOSTING>
                                <ALTERID>1</ALTERID>
                                <USEDFORINTEREST>No</USEDFORINTEREST>
                                <ISDEFAULTCATEGORYSTORED>Yes</ISDEFAULTCATEGORYSTORED>
                                <INVOICEDETAILS.LIST>
                                    <INVOICENO>{invoice.voucher_number}</INVOICENO>
                                    <INVOICEDATE>{invoice.invoice_date.replace('-', '')}</INVOICEDATE>
                                </INVOICEDETAILS.LIST>
                                <ATTACHMENTSLIST.LIST>
                                    <ATTACHMENTSLIST.LIST>
                                        <ATTACHMENTNAME>Tax Invoice</ATTACHMENTNAME>
                                        <ATTACHMENTSRCPATH>{attachment_path}</ATTACHMENTSRCPATH>
                                    </ATTACHMENTSLIST.LIST>
                                </ATTACHMENTSLIST.LIST>
                                {inventory_entries}
                                {vendor_entry}
                                {vat_entry}
                                {discount_entry}
                            </VOUCHER>
                        </TALLYMESSAGE>
                    </REQUESTDATA>
                </IMPORTDATA>
            </BODY>
        </ENVELOPE>
        """

        return xml


def create_voucher_from_invoice(invoice: InvoiceData, tally_api: TallyAPI = None) -> VoucherResult:
    """
    Convenience function to create a voucher from invoice data
    """
    creator = PurchaseVoucherCreator(tally_api)
    return creator.create_voucher(invoice)


if __name__ == "__main__":
    # Test with sample data
    from ocr_extractor import InvoiceItem

    sample_invoice = InvoiceData(
        voucher_number="1037",
        vendor_name="Media General Trading LLC",
        vendor_trn="100507591400003",
        invoice_date="27-Mar-2024",
        items=[
            InvoiceItem(
                item_code="14504",
                description="Bitumen Polycoat WB 200ltr Henkel",
                quantity=1.0,
                unit="DRUM",
                rate=330.0,
                amount=330.0
            ),
            InvoiceItem(
                item_code="19417",
                description="NOORA BRUSH H/D",
                quantity=2.0,
                unit="NOS",
                rate=3.25,
                amount=6.50
            ),
        ],
        subtotal=476.10,
        vat_amount=23.81,
        discount=1.40,
        total_amount=499.91,
        project_name="Villa Project",
        attachment_path=r"C:\Invoices\1037.pdf"
    )

    print("Testing voucher creation...")
    api = TallyAPI()

    if api.test_connection():
        result = create_voucher_from_invoice(sample_invoice, api)
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Created Ledgers: {result.created_ledgers}")
        print(f"Created Items: {result.created_items}")
        print(f"Created Projects: {result.created_projects}")
    else:
        print("Cannot connect to Tally")
