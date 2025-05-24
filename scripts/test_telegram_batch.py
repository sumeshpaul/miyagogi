import asyncio
import os
from dotenv import load_dotenv
import aiohttp

# Load the .env file from the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TEST_CHAT_ID", "your_telegram_user_id")  # Optionally load or hardcode your user ID

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Sample test messages
TEST_QUERIES = [
    "Tell me about Thuya Cleanser",
    "What is the price of Noemi Lash Lift Kit?",
    "Is Thuya Brow Lamination in stock?",
    "Compare Thuya Cleanser and Noemi Cleanser",
    "Show me lash extension glue",
    "Tell me about bleach for eyebrows",
    "Thuya Brow Tint vs Noemi Henna"
]

async def send_message(session, text):
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    async with session.post(f"{BASE_URL}/sendMessage", json=payload) as resp:
        response = await resp.json()
        print("✅ Sent:", text[:50], "→ Response:", response.get("ok"))

async def batch_send_messages():
    async with aiohttp.ClientSession() as session:
        tasks = [send_message(session, query) for query in TEST_QUERIES]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(batch_send_messages())
