# scripts/test_telegram_batch.py

import aiohttp
import asyncio
import csv
import time
from datetime import datetime

BOT_TOKEN = "7987599734:AAGJPPAwNo6lzlUxB6PenofWCPXKZ_u6t_0"
CHAT_ID = "715037900"  # ✅ Replace with your actual chat ID
URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

queries = [
    "Thuya cleanser",
    "Thuya Brow Scrub 15ml",
    "Sculptor Dye Oxidant",
    "lash foam cleaner",
    "thuya shampoo",
    "lash lift glue",
    "Noemi pads",
    "brow lamination Thuya vegan",
    "serum for men",
    "dishwashing cream"
]

log_filename = "batch_test_log.csv"

# Initialize CSV log
with open(log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Query", "Status", "Description"])

async def send_query(session, query):
    timestamp = datetime.utcnow().isoformat()
    payload = {
        "chat_id": CHAT_ID,
        "text": query
    }

    try:
        async with session.post(URL, json=payload) as resp:
            result = await resp.json()
            if result.get("ok"):
                print(f"✅ Sent: {query}")
                log_row = [timestamp, query, "✅ Sent", ""]
            else:
                error_msg = result.get("description", "Unknown error")
                print(f"❌ Failed: {query[:30]}... → {error_msg}")
                log_row = [timestamp, query, "❌ Failed", error_msg]
    except Exception as e:
        print(f"❌ Exception: {query[:30]}... → {e}")
        log_row = [timestamp, query, "❌ Exception", str(e)]

    # Append to CSV
    with open(log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_row)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_query(session, q) for q in queries]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
