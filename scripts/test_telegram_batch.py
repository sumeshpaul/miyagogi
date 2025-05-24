# scripts/test_telegram_batch.py
import asyncio
import aiohttp
import time

BOT_TOKEN = "7987599734:AAGJPPAwNo6lzlUxB6PenofWCPXKZ_u6t_0"
CHAT_ID = "<your_user_id>"  # ← Replace with your real Telegram user_id
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

TEST_QUERIES = [
    "thuya shampoo",
    "Thuya Brow Scrub 15ml",
    "brow lamination Thuya vegan",
    "Thuya cleanser",
    "Noemi pads",
    "lash lift glue",
    "Sculptor Dye Oxidant",
    "lash foam cleaner",
    "serum for men",
    "dishwashing cream",
]

async def send_message(session, text):
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    async with session.post(f"{API_URL}/sendMessage", json=payload) as resp:
        result = await resp.json()
        return result

async def fetch_updates(session):
    async with session.get(f"{API_URL}/getUpdates") as resp:
        return await resp.json()

async def run_tests():
    async with aiohttp.ClientSession() as session:
        for query in TEST_QUERIES:
            response = await send_message(session, query)
            if response.get("ok"):
                print(f"✅ Sent: {query}")
            else:
                print(f"❌ Failed: {query} → {response}")

            await asyncio.sleep(2)  # Give bot time to reply

        # Fetch latest messages
        print("\n🔍 Fetching recent replies from Telegram:")
        updates = await fetch_updates(session)
        messages = updates.get("result", [])[-len(TEST_QUERIES):]
        for msg in messages:
            text = msg["message"]["text"]
            print(f"📨 {text}")

asyncio.run(run_tests())
