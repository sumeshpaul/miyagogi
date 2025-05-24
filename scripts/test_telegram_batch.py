# scripts/test_telegram_batch.py

import asyncio
import os
import aiohttp
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

TEST_QUERIES = [
    "Is Thuya Brow Lamination in stock?",
    "Tell me about Thuya Cleanser",
    "Show me lash extension glue",
    "Compare Thuya Cleanser and Noemi Cleanser",
    "Thuya Brow Tint vs Noemi Henna",
    "Tell me about bleach for eyebrows",
    "What is the price of Noemi Lash Lift Kit?"
]

async def get_latest_chat_id():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/getUpdates") as resp:
            data = await resp.json()
            results = data.get("result", [])
            if results:
                return results[-1]["message"]["chat"]["id"]
            return None

async def send_message(session, text, chat_id):
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    async with session.post(f"{BASE_URL}/sendMessage", json=payload) as resp:
        data = await resp.json()
        ok = data.get("ok", False)
        print(f"✅ Sent: {text[:40]}... → Response: {ok}")

async def batch_send():
    chat_id = await get_latest_chat_id()
    if not chat_id:
        print("❌ Could not determine chat_id. Make sure you've messaged the bot.")
        return
    async with aiohttp.ClientSession() as session:
        tasks = [send_message(session, query, chat_id) for query in TEST_QUERIES]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(batch_send())
