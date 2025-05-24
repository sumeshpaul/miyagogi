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
    "Thuya Brow Lamination Kit",
    "Thuya Perming Kit",
    "Thuya Cleanser",
    "Thuya White Contour Paste",
    "Thuya Silicone Shields",
    "Thuya Eyebrow Henna Color",
    "Thuya Eyelash Dye Oxidant",
    "Thuya Tint Remover",
    "Thuya Vegan Lash Lift Kit",
    "Thuya Neutralizer Cream",
    "Sculptor Eyelash Glue",
    "Sculptor Brow Scrub",
    "Sculptor Lash Lifting Cream",
    "Noemi Eyelash Lamination Pads",
    "Noemi Vegan Lash Dye",
    "Noemi Lifting Tool",
    "Noemi Cleanser Pads",
    "Noemi Keratin Booster",
    "Lashgo Lash Lift Glue",
    "Lashgo Neutralizing Lotion",
    "Lashgo Silicone Pads",
    "Lashgo Perming Solution",
    "Enigma Lash Foam Cleanser",
    "Enigma Lash Lift Kit",
    "Enigma Oil-Free Remover",
    "Viktoria Brow Sculpting Paste",
    "Viktoria Lamination Cream",
    "Viktoria Brow Conditioner",
    "Revitabrow Eyebrow Serum",
    "Revitabrow Conditioner"
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
