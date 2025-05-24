import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from woocommerce import API

# Load .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# WooCommerce Auth
wcapi = API(
    url=os.environ["WC_URL"],
    consumer_key=os.environ["WC_KEY"],
    consumer_secret=os.environ["WC_SECRET"],
    version="wc/v3",
    timeout=30,
)


# Fetch and flatten products
def fetch_all_products():
    all_products = []
    page = 1
    while True:
        response = wcapi.get(
            "products",
            params={"per_page": 100, "page": page},
        )
        batch = response.json()
        if not batch:
            break
        all_products.extend(batch)
        page += 1
    return all_products


# Serialize nested fields
def serialize_product(product):
    return {
        k: json.dumps(v) if isinstance(v, (dict, list)) else v
        for k, v in product.items()
    }


# Main export function
def export_to_csv(products, output_file="woo_products_full.csv"):
    if not products:
        print("No products found.")
        return
    fieldnames = list(products[0].keys())
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in products:
            writer.writerow(serialize_product(p))
    print(f"✅ Exported {len(products)} products to {output_file}")


# Run
if __name__ == "__main__":
    products = fetch_all_products()
    export_to_csv(products)
