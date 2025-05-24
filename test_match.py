import sqlite3, difflib, re

def normalize(text):
    text = text.lower()
    text = re.sub(r'\(.*?\)', '', text)  # remove (...) parts
    text = re.sub(r'\b(ml|g|pairs|kit|pcs|pack|sachet|10ml|20ml|150ml|15ml|30ml|8 pairs)\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    return text.strip()

def match_query(query, db_path="data/products.db"):
    query_norm = normalize(query)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price, stock_status FROM products")
    products = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = 0.0
    for name, price, stock in products:
        name_norm = normalize(name)
        score = difflib.SequenceMatcher(None, query_norm, name_norm).ratio()
        if query_norm in name_norm:
            score += 0.4  # reward substring match
        if score > best_score:
            best_score = score
            best_match = (name, price, stock)

    return best_match, best_score

# Test queries
print(match_query("Thuya Cleanser"))
print(match_query("Noemi dye kit"))
print(match_query("Lash brow brush"))
print(match_query("Thuya lash filler"))
