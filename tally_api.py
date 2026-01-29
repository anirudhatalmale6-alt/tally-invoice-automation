"""
Tally ERP 9 XML API Integration Module
Handles communication with Tally ERP 9 via XML requests
"""

import requests
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass

from config import TALLY_HOST, TALLY_PORT, COMPANY_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TallyConnectionError(Exception):
    """Raised when unable to connect to Tally"""
    pass


class TallyAPIError(Exception):
    """Raised when Tally returns an error"""
    pass


class TallyAPI:
    """
    Tally ERP 9 XML API Client
    Handles all communication with Tally
    """

    def __init__(self, host: str = TALLY_HOST, port: int = TALLY_PORT, company: str = COMPANY_NAME):
        self.host = host
        self.port = port
        self.company = company
        self.url = f"http://{host}:{port}"

    def _send_request(self, xml_data: str) -> str:
        """
        Send XML request to Tally and return response
        """
        headers = {'Content-Type': 'application/xml'}

        try:
            response = requests.post(self.url, data=xml_data, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.ConnectionError:
            raise TallyConnectionError(
                f"Cannot connect to Tally at {self.url}. "
                "Make sure Tally is running and ODBC server is enabled."
            )
        except requests.exceptions.Timeout:
            raise TallyConnectionError("Tally request timed out")
        except requests.exceptions.RequestException as e:
            raise TallyAPIError(f"Tally request failed: {e}")

    def _prettify_xml(self, xml_string: str) -> str:
        """Return pretty-printed XML string"""
        try:
            parsed = minidom.parseString(xml_string)
            return parsed.toprettyxml(indent="  ")
        except:
            return xml_string

    def test_connection(self) -> bool:
        """Test connection to Tally"""
        xml = """
        <ENVELOPE>
            <HEADER>
                <VERSION>1</VERSION>
                <TALLYREQUEST>Export</TALLYREQUEST>
                <TYPE>Data</TYPE>
                <ID>List of Companies</ID>
            </HEADER>
            <BODY>
                <DESC>
                    <STATICVARIABLES>
                        <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                    </STATICVARIABLES>
                </DESC>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            return 'COMPANY' in response.upper() or 'ENVELOPE' in response.upper()
        except (TallyConnectionError, TallyAPIError):
            return False

    def get_company_list(self) -> List[str]:
        """Get list of companies in Tally"""
        xml = """
        <ENVELOPE>
            <HEADER>
                <VERSION>1</VERSION>
                <TALLYREQUEST>Export</TALLYREQUEST>
                <TYPE>Collection</TYPE>
                <ID>List of Companies</ID>
            </HEADER>
            <BODY>
                <DESC>
                    <STATICVARIABLES>
                        <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                    </STATICVARIABLES>
                </DESC>
            </BODY>
        </ENVELOPE>
        """

        response = self._send_request(xml)
        companies = []

        try:
            root = ET.fromstring(response)
            for company in root.iter('COMPANY'):
                name = company.find('NAME')
                if name is not None and name.text:
                    companies.append(name.text)
        except ET.ParseError:
            pass

        return companies

    def ledger_exists(self, ledger_name: str) -> bool:
        """Check if a ledger exists in Tally"""
        xml = f"""
        <ENVELOPE>
            <HEADER>
                <VERSION>1</VERSION>
                <TALLYREQUEST>Export</TALLYREQUEST>
                <TYPE>Data</TYPE>
                <ID>Ledger</ID>
            </HEADER>
            <BODY>
                <DESC>
                    <STATICVARIABLES>
                        <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                        <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                    </STATICVARIABLES>
                    <TDL>
                        <TDLMESSAGE>
                            <REPORT NAME="LedgerCheck">
                                <FORMS>LedgerCheckForm</FORMS>
                            </REPORT>
                            <FORM NAME="LedgerCheckForm">
                                <PARTS>LedgerCheckPart</PARTS>
                            </FORM>
                            <PART NAME="LedgerCheckPart">
                                <LINES>LedgerCheckLine</LINES>
                                <REPEAT>LedgerCheckLine : LedgerCollection</REPEAT>
                            </PART>
                            <LINE NAME="LedgerCheckLine">
                                <FIELDS>LedgerName</FIELDS>
                            </LINE>
                            <FIELD NAME="LedgerName">
                                <SET>$Name</SET>
                            </FIELD>
                            <COLLECTION NAME="LedgerCollection">
                                <TYPE>Ledger</TYPE>
                                <FILTER>NameFilter</FILTER>
                            </COLLECTION>
                            <SYSTEM TYPE="Formulae" NAME="NameFilter">$Name = "{ledger_name}"</SYSTEM>
                        </TDLMESSAGE>
                    </TDL>
                </DESC>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            return ledger_name.upper() in response.upper()
        except (TallyConnectionError, TallyAPIError):
            return False

    def create_ledger(self, name: str, group: str = "Sundry Creditors",
                      address: str = "", trn: str = "") -> bool:
        """
        Create a new ledger in Tally
        """
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
                            <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                        </STATICVARIABLES>
                    </REQUESTDESC>
                    <REQUESTDATA>
                        <TALLYMESSAGE xmlns:UDF="TallyUDF">
                            <LEDGER NAME="{name}" ACTION="Create">
                                <NAME>{name}</NAME>
                                <PARENT>{group}</PARENT>
                                <ADDRESS.LIST>
                                    <ADDRESS>{address}</ADDRESS>
                                </ADDRESS.LIST>
                                <LEDSTATENAME></LEDSTATENAME>
                                <COUNTRYNAME>United Arab Emirates</COUNTRYNAME>
                                <GSTREGISTRATIONTYPE>Regular</GSTREGISTRATIONTYPE>
                                <PARTYGSTIN>{trn}</PARTYGSTIN>
                            </LEDGER>
                        </TALLYMESSAGE>
                    </REQUESTDATA>
                </IMPORTDATA>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            success = 'CREATED' in response.upper() or 'LINEERROR' not in response.upper()
            if success:
                logger.info(f"Created ledger: {name}")
            else:
                logger.error(f"Failed to create ledger {name}: {response}")
            return success
        except (TallyConnectionError, TallyAPIError) as e:
            logger.error(f"Failed to create ledger {name}: {e}")
            return False

    def stock_item_exists(self, item_name: str) -> bool:
        """Check if a stock item exists in Tally"""
        xml = f"""
        <ENVELOPE>
            <HEADER>
                <VERSION>1</VERSION>
                <TALLYREQUEST>Export</TALLYREQUEST>
                <TYPE>Collection</TYPE>
                <ID>StockItemCollection</ID>
            </HEADER>
            <BODY>
                <DESC>
                    <STATICVARIABLES>
                        <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                        <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                    </STATICVARIABLES>
                    <TDL>
                        <TDLMESSAGE>
                            <COLLECTION NAME="StockItemCollection">
                                <TYPE>StockItem</TYPE>
                                <FILTER>ItemNameFilter</FILTER>
                                <FETCH>NAME</FETCH>
                            </COLLECTION>
                            <SYSTEM TYPE="Formulae" NAME="ItemNameFilter">$Name = "{item_name}"</SYSTEM>
                        </TDLMESSAGE>
                    </TDL>
                </DESC>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            return item_name.upper() in response.upper()
        except (TallyConnectionError, TallyAPIError):
            return False

    def create_stock_item(self, name: str, unit: str = "Nos",
                          group: str = "Primary", category: str = "") -> bool:
        """
        Create a new stock item in Tally
        """
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
                            <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                        </STATICVARIABLES>
                    </REQUESTDESC>
                    <REQUESTDATA>
                        <TALLYMESSAGE xmlns:UDF="TallyUDF">
                            <STOCKITEM NAME="{name}" ACTION="Create">
                                <NAME>{name}</NAME>
                                <PARENT>{group}</PARENT>
                                <CATEGORY>{category}</CATEGORY>
                                <BASEUNITS>{unit}</BASEUNITS>
                                <GSTAPPLICABLE>Applicable</GSTAPPLICABLE>
                                <GSTTYPEOFSUPPLY>Goods</GSTTYPEOFSUPPLY>
                            </STOCKITEM>
                        </TALLYMESSAGE>
                    </REQUESTDATA>
                </IMPORTDATA>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            success = 'CREATED' in response.upper() or 'LINEERROR' not in response.upper()
            if success:
                logger.info(f"Created stock item: {name}")
            else:
                logger.error(f"Failed to create stock item {name}: {response}")
            return success
        except (TallyConnectionError, TallyAPIError) as e:
            logger.error(f"Failed to create stock item {name}: {e}")
            return False

    def create_unit(self, unit_name: str, symbol: str = None) -> bool:
        """Create a unit of measure in Tally"""
        if symbol is None:
            symbol = unit_name

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
                            <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                        </STATICVARIABLES>
                    </REQUESTDESC>
                    <REQUESTDATA>
                        <TALLYMESSAGE xmlns:UDF="TallyUDF">
                            <UNIT NAME="{unit_name}" ACTION="Create">
                                <NAME>{unit_name}</NAME>
                                <ORIGINALNAME>{symbol}</ORIGINALNAME>
                                <ISSIMPLEUNIT>Yes</ISSIMPLEUNIT>
                            </UNIT>
                        </TALLYMESSAGE>
                    </REQUESTDATA>
                </IMPORTDATA>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            return 'CREATED' in response.upper() or 'LINEERROR' not in response.upper()
        except (TallyConnectionError, TallyAPIError):
            return False

    def project_exists(self, project_name: str) -> bool:
        """Check if a cost centre (project) exists"""
        xml = f"""
        <ENVELOPE>
            <HEADER>
                <VERSION>1</VERSION>
                <TALLYREQUEST>Export</TALLYREQUEST>
                <TYPE>Collection</TYPE>
                <ID>CostCentreCollection</ID>
            </HEADER>
            <BODY>
                <DESC>
                    <STATICVARIABLES>
                        <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                        <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                    </STATICVARIABLES>
                    <TDL>
                        <TDLMESSAGE>
                            <COLLECTION NAME="CostCentreCollection">
                                <TYPE>CostCentre</TYPE>
                                <FILTER>ProjectFilter</FILTER>
                                <FETCH>NAME</FETCH>
                            </COLLECTION>
                            <SYSTEM TYPE="Formulae" NAME="ProjectFilter">$Name = "{project_name}"</SYSTEM>
                        </TDLMESSAGE>
                    </TDL>
                </DESC>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            return project_name.upper() in response.upper()
        except (TallyConnectionError, TallyAPIError):
            return False

    def create_project(self, name: str, parent: str = "") -> bool:
        """Create a cost centre (project) in Tally"""
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
                            <SVCURRENTCOMPANY>{self.company}</SVCURRENTCOMPANY>
                        </STATICVARIABLES>
                    </REQUESTDESC>
                    <REQUESTDATA>
                        <TALLYMESSAGE xmlns:UDF="TallyUDF">
                            <COSTCENTRE NAME="{name}" ACTION="Create">
                                <NAME>{name}</NAME>
                                <PARENT>{parent}</PARENT>
                            </COSTCENTRE>
                        </TALLYMESSAGE>
                    </REQUESTDATA>
                </IMPORTDATA>
            </BODY>
        </ENVELOPE>
        """

        try:
            response = self._send_request(xml)
            success = 'CREATED' in response.upper() or 'LINEERROR' not in response.upper()
            if success:
                logger.info(f"Created project/cost centre: {name}")
            return success
        except (TallyConnectionError, TallyAPIError) as e:
            logger.error(f"Failed to create project {name}: {e}")
            return False


if __name__ == "__main__":
    # Test connection
    api = TallyAPI()

    print("Testing Tally connection...")
    if api.test_connection():
        print("Connected to Tally successfully!")
        companies = api.get_company_list()
        print(f"Companies: {companies}")
    else:
        print("Failed to connect to Tally. Make sure:")
        print("1. Tally is running")
        print("2. Go to Gateway of Tally > F12: Configure > Advanced Configuration")
        print("3. Set 'Enable ODBC Server' to Yes")
        print(f"4. Tally should be listening on port {TALLY_PORT}")
