# scripts/test_telegram_batch.py

import aiohttp
import asyncio
import csv
from datetime import datetime
import time

BOT_TOKEN = "7987599734:AAGJPPAwNo6lzlUxB6PenofWCPXKZ_u6t_0"
CHAT_ID = "715037900"
SEND_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
GET_UPDATES_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
LOG_FILE = "batch_test_log.csv"

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

# Helper: Get all recent messages from Telegram
async def fetch_replies():
    async with aiohttp.ClientSession() as session:
        async with session.get(GET_UPDATES_URL) as resp:
            return await resp.json()

# Helper: Send message and wait briefly
async def send_and_wait(session, query):
    payload = {"chat_id": CHAT_ID, "text": query}
    async with session.post(SEND_URL, json=payload) as resp:
        result = await resp.json()
        status = "✅ Sent" if result.get("ok") else "❌ Failed"
        desc = "" if result.get("ok") else result.get("description", "Unknown error")
        print(f"{status}: {query[:30]}... {desc}")
        return query, status, desc

# Main orchestration
async def main():
    print("🚀 Sending queries...")
    async with aiohttp.ClientSession() as session:
        tasks = [send_and_wait(session, q) for q in queries]
        results = await asyncio.gather(*tasks)

    print("⏳ Waiting for bot replies...")
    await asyncio.sleep(10)  # ⏱ Wait for replies to be generated

    replies_raw = await fetch_replies()
    all_messages = replies_raw.get("result", [])

    # Match each query to a reply
    latest_replies = {msg["message"]["text"]: msg["message"].get("reply_to_message", {}).get("text", "") for msg in all_messages if "message" in msg and "text" in msg["message"]}

    print("🧾 Logging results to CSV...")
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Query", "Status", "Bot Reply"])
        for query, status, desc in results:
            timestamp = datetime.utcnow().isoformat()
            reply = next((m["message"]["text"] for m in all_messages if m["message"].get("reply_to_message", {}).get("text", "") == query), "— Not found —") if status == "✅ Sent" else desc
            writer.writerow([timestamp, query, status, reply])

    print(f"✅ Done. Log saved to {LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
