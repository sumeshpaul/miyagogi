# app/search_products.py

import sqlite3
import re
from typing import List, Dict
from bs4 import BeautifulSoup
import json
import html
import difflib

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

    # Build WHERE clause dynamically
    clauses = []
    params = []
    for kw in keywords:
        like = f"%{kw.lower()}%"
        for field in ["name", "tags", "categories", "attributes"]:
            clauses.append(f"LOWER({field}) LIKE ?")
            params.append(like)

    query = f"""
    SELECT name, price, stock_status, short_description, permalink, tags
    FROM products
    WHERE {" OR ".join(clauses)}
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Determine primary brand to prioritize
    primary_brand = None
    for kw in keywords:
        if kw.lower() in (
            "thuya", "lashgo", "enigma", "sculptor", "viktoria", "revitabrow"
        ):
            primary_brand = kw.upper()
            break

    main_brand_results = {}
    other_brand_results = {}

    for name, price, stock, summary, link, tags in rows:
        brand = extract_brand(tags, name).upper()
        item = {
            "name": name,
            "price": price,
            "stock": stock,
            "summary": clean_html(summary),
            "link": link,
        }
        if primary_brand and brand == primary_brand:
            main_brand_results.setdefault(brand, []).append(item)
        else:
            other_brand_results.setdefault(brand or "Other", []).append(item)

    conn.close()

    results = {}
    if main_brand_results:
        results.update(main_brand_results)
    if other_brand_results:
        results["Suggested from other brands"] = [
            item for sublist in other_brand_results.values() for item in sublist[:3]
        ]

    # ✅ Fuzzy fallback if no results found
    if not results:
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

            results["Suggested matches"] = [{
                "name": name,
                "price": price,
                "stock": stock,
                "summary": "",
                "link": "#"
            } for name, price, stock in scored]
        except Exception as e:
            print(f"❌ Fuzzy fallback error: {e}")

    return results
