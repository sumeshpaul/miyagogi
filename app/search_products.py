# app/search_products.py

import sqlite3
import re
from typing import List, Dict
from bs4 import BeautifulSoup
import json
import html
import difflib
from collections import defaultdict

def clean_html(text):
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def extract_brand(tags: str, fallback_name: str) -> str:
    try:
        parsed = json.loads(tags)
        if isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], dict) and "name" in parsed[0]:
                return parsed[0]["name"]
            elif isinstance(parsed[0], str):
                return parsed[0]
    except (json.JSONDecodeError, TypeError):
        pass

    if isinstance(tags, str):
        return re.split(r"[;,]", tags)[0].strip()

    return fallback_name.split()[0] if fallback_name else "Unknown"

def search_products_by_keywords(
    keywords: List[str], db_path: str
) -> Dict[str, List[Dict]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Expand WHERE clause to include more fields
    clauses = []
    params = []
    for kw in keywords:
        like = f"%{kw.lower()}%"
        for field in ["name", "tags", "categories", "attributes", "short_description"]:
            clauses.append(f"LOWER({field}) LIKE ?")
            params.append(like)

    query = f"""
    SELECT name, price, stock_status, short_description, permalink, tags
    FROM products
    WHERE {" OR ".join(clauses)}
    LIMIT 20
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results_by_brand = defaultdict(list)
    for name, price, stock, desc, link, tags in rows:
        brand = extract_brand(tags, name).upper()
        results_by_brand[brand].append({
            "name": name,
            "price": price,
            "stock": stock,
            "summary": clean_html(desc),
            "link": link,
        })

    if results_by_brand:
        return dict(results_by_brand)

    # ✅ Fuzzy fallback if no results found
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, price, stock_status FROM products")
        all_products = cursor.fetchall()
        conn.close()

        scored = sorted(
            all_products,
            key=lambda row: max(
                difflib.SequenceMatcher(None, kw.lower(), row[0].lower()).ratio()
                for kw in keywords
            ),
            reverse=True
        )[:3]

        return {
            "Suggested matches": [{
                "name": name,
                "price": price,
                "stock": stock,
                "summary": "",
                "link": "#"
            } for name, price, stock in scored]
        }
    except Exception as e:
        print(f"❌ Fuzzy fallback error: {e}")
        return {}
