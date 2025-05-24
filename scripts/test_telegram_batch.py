# scripts/test_telegram_batch.py
import aiohttp
import asyncio
import json

BOT_TOKEN = "7987599734:AAGJPPAwNo6lzlUxB6PenofWCPXKZ_u6t_0"
CHAT_ID = "715037900"  # Replace with your Telegram chat ID
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

async def send_query(session, query):
    payload = {
        "chat_id": CHAT_ID,
        "text": query
    }
    async with session.post(URL, json=payload) as resp:
        result = await resp.json()
        if not result.get("ok"):
            print(f"❌ Failed: {query[:30]}... → {result}")
        else:
            print(f"✅ Sent: {query}")

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_query(session, q) for q in queries]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
