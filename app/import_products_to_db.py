import sqlite3
import pandas as pd
import os

csv_path = "/app/woo_products_full.csv"
db_path = "/app/data/products.db"

# Load CSV
df = pd.read_csv(csv_path)

# Connect to SQLite
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Drop and recreate table
c.execute("DROP TABLE IF EXISTS products")

# Create table based on CSV columns (sample schema - adjust as needed)
c.execute(
    """
CREATE TABLE products (
    id INTEGER,
    name TEXT,
    slug TEXT,
    permalink TEXT,
    price TEXT,
    regular_price TEXT,
    sale_price TEXT,
    stock_status TEXT,
    categories TEXT,
    tags TEXT,
    description TEXT,
    short_description TEXT,
    attributes TEXT,
    variations TEXT
)
"""
)

# Fill nulls and insert
df = df.fillna("")
rows = df.to_dict(orient="records")

for row in rows:
    c.execute(
        """
    INSERT INTO products (
        id, name, slug, permalink, price, regular_price, sale_price,
        stock_status, categories, tags, description, short_description,
        attributes, variations
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            row.get("id"),
            row.get("name"),
            row.get("slug"),
            row.get("permalink"),
            row.get("price"),
            row.get("regular_price"),
            row.get("sale_price"),
            row.get("stock_status"),
            row.get("categories"),
            row.get("tags"),
            row.get("description"),
            row.get("short_description"),
            row.get("attributes"),
            row.get("variations"),
        ),
    )

conn.commit()
conn.close()
print("✅ Products imported to products.db successfully.")
