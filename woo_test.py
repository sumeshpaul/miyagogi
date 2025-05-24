import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

WC_URL = os.getenv("WC_URL") + "/wp-json/wc/v3/products"
WC_KEY = os.getenv("WC_KEY")
WC_SECRET = os.getenv("WC_SECRET")

query = "noemi dye gel"

response = requests.get(
    WC_URL,
    params={"search": query, "per_page": 1},
    auth=HTTPBasicAuth(WC_KEY, WC_SECRET),
)

if response.status_code == 200 and response.json():
    product = response.json()[0]
    name = product["name"]
    price = product.get("price", "N/A")
    stock_status = "in stock" if product.get("stock_status") == "instock" else "out of stock"
    print(f"✅ {name} — AED {price} ({stock_status})")
else:
    print(f"❌ No product found or API error: {response.status_code}")
