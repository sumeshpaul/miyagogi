# scripts/test_telegram_batch_v2.py

import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TEST_CHAT_ID", "")  # Optional: Add your test chat ID to .env

if not BOT_TOKEN or not CHAT_ID:
    raise Exception("❌ TELEGRAM_TOKEN or TEST_CHAT_ID missing in .env")

# 🔁 Define test prompts for all categories
test_queries = [
    # Exact Match
    "Thuya Brow Scrub 15ml",
    "Sculptor Dye Oxidant",
    "Katya Vinog Lash Lift Pads Silicone Shields",

    # Brand-only
    "Thuya",
    "Sculptor",
    "Lash Lift",
    "Eyebrow scrub",

    # Vague or misspelled
    "thuya cleanser",
    "thya cleenzer",
    "brow laminating shampoo",
    "lash foam cleaner",

    # Unrelated
    "dishwasher cream",
    "vitamin supplement",
    "serum for men",

    # Long
    "Tell me about all Thuya vegan products for lamination and brow coloring."
]

async def send_message(session, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    async with session.post(url, json=payload) as resp:
        result = await resp.json()
        if resp.status == 200 and result.get("ok"):
            print(f"✅ Sent: {text[:40]}... → Success")
        else:
            print(f"❌ Failed: {text[:40]}... → {result}")

async def main():
    async with aiohttp.ClientSession() as session:
        for query in test_queries:
            await send_message(session, query)
            await asyncio.sleep(2)  # small delay to avoid rate limits

if __name__ == "__main__":
    asyncio.run(main())
