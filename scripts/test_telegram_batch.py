# scripts/test_telegram_batch.py

import aiohttp
import asyncio
import csv
import time
from datetime import datetime
import html

BOT_TOKEN = "7987599734:AAGJPPAwNo6lzlUxB6PenofWCPXKZ_u6t_0"
CHAT_ID = "715037900"  # Your Telegram chat ID
SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
LOGS_URL = "http://localhost:8001/recent-logs?limit=100"


timestamp_sent = datetime.utcnow().isoformat()
print("🚀 Sending queries...")

queries = [
"Tell me about Noemi Redness Neutralizer 20g",
"Tell me about Noemi Brow Gel Dye, 10ml",
"Do you have LB Brow Therapy Caring Oil?",
"Tell me about Sculptor Eyelashes Cream Dye (3 colors), 15ml",
"Do you have Noemi Premium silicone pads 2.0 for lash lift (8 pairs)?",
"Price of Glue holder pallets 50 pcs",
"Price of Lash&go gel remover",
"Do you have Sculptor Eyelashes Cream Dye (3 colors), 15ml?",
"Price of Lash Box 5 Palettes",
"Tell me about Lash Box 5 Palettes",
"Do you have Noemi PreDyed Thread Black, 20m?",
"Do you have Noemi Lash And Brow Hybrid Dye Sachet Kit?",
"Price of Sculptor Eyelashes Cream Dye (3 colors), 15ml",
"Price of LB Brow Therapy Caring Oil",
"Tell me about LB Brow Therapy Caring Oil",
"Price of Noemi PreDyed Thread Black, 20m",
"Price of Noemi Lash And Brow Hybrid Dye Sachet Kit",
"Price of Noemi Premium silicone pads 2.0 for lash lift (8 pairs)",
"Price of Noemi Redness Neutralizer 20g",
"Tell me about Noemi Premium silicone pads 2.0 for lash lift (8 pairs)",
"Tell me about Noemi PreDyed Thread Black, 20m",
"Do you have Glue holder pallets 50 pcs?",
"Do you have Lash Box 5 Palettes?",
"Tell me about Noemi Lash And Brow Hybrid Dye Sachet Kit",
"Tell me about Glue holder pallets 50 pcs",
"Do you have Noemi Redness Neutralizer 20g?",
"Tell me about Lash&go gel remover",
"Do you have Noemi Brow Gel Dye, 10ml?",
"Price of Noemi Brow Gel Dye, 10ml",
"Do you have Lash&go gel remover?",
]

async def send_query(session, query):
    payload = {
        "chat_id": CHAT_ID,
        "text": query
    }
    async with session.post(SEND_URL, json=payload) as resp:
        result = await resp.json()
        if not result.get("ok"):
            print(f"❌ Failed: {query[:30]}... → {result}")
        else:
            print(f"✅ Sent: {query[:30]}...")

async def send_all():
    async with aiohttp.ClientSession() as session:
        tasks = [send_query(session, q) for q in queries]
        await asyncio.gather(*tasks)

asyncio.run(send_all())

print("⏳ Waiting for bot replies...")
time.sleep(8)  # wait for bot to respond

print("📥 Fetching replies from /recent-logs...")
async def fetch_logs():
    async with aiohttp.ClientSession() as session:
        async with session.get(LOGS_URL) as resp:
            return await resp.json()

logs = asyncio.run(fetch_logs())

print("🧾 Logging results to CSV...")
rows = []

# Prepare a reply lookup dictionary
log_lookup = {}
for entry in logs:
    if entry["timestamp"] > timestamp_sent and entry["user_id"] == CHAT_ID:
        query = entry["query"]
        response = html.unescape(entry["response"])
        log_lookup[query.strip().lower()] = response.strip()

for q in queries:
    q_normalized = q.strip().lower()
    response = log_lookup.get(q_normalized, "— Not found —")
    rows.append({
        "Timestamp": timestamp_sent,
        "Query": q,
        "Status": "✅ Sent",
        "Bot Reply": response
    })

with open("batch_test_log.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Timestamp", "Query", "Status", "Bot Reply"])
    writer.writeheader()
    writer.writerows(rows)

print("✅ Done. Log saved to batch_test_log.csv")
