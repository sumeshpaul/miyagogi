import pandas as pd

# Load the CSV
df = pd.read_csv("../woo_products_full.csv")

# Show basic stats
print("\n📊 Columns:")
print(df.columns.tolist())

print("\n🧮 Total Products:", len(df))

print("\n🔍 Sample Rows:")
print(df.head(5).to_markdown())
