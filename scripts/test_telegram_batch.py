import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()  # Load TELEGRAM_TOKEN and TEST_TELEGRAM_USER_ID from .env

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
USER_ID = os.getenv("TEST_TELEGRAM_USER_ID")

if not TELEGRAM_TOKEN or not USER_ID:
    raise ValueError("TELEGRAM_TOKEN and TEST_TELEGRAM_USER_ID must be set in .env")

TEST_MESSAGES = [
    "Tell me about Thuya Silicone Shields",
    "Is it in stock?",
    "Do you have anything for lash cleaning?",
    "Do you sell electric kettles?",
    "Show me all Thuya products",
    "Which Thuya product is best for brows?",
    "What’s the price of Noemi brow gel?",
    "Compare Thuya Laminator vs Noemi Brow Fix",
]

async def send_messages():
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    async with aiohttp.ClientSession() as session:
        for message in TEST_MESSAGES:
            payload = {
                "chat_id": USER_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            async with session.post(base_url, json=payload) as resp:
                result = await resp.json()
                print(f"✅ Sent: {message} | Response OK: {result.get('ok', False)}")
            await asyncio.sleep(3)  # Pause between messages to avoid spam

if __name__ == "__main__":
    asyncio.run(send_messages())
