import os
import asyncio
import aiohttp
import csv
from pathlib import Path
from dotenv import load_dotenv

# Load .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

TEST_PROMPTS = [
    "Is Thuya Brow Lamination in stock?",
    "Tell me about Thuya Cleanser",
    "Show me lash extension glue",
    "Compare Thuya Cleanser and Noemi Cleanser",
    "Thuya Brow Tint vs Noemi Henna",
    "Tell me about bleach for eyebrows",
    "What is the price of Noemi Lash Lift Kit?"
]

CSV_LOG = "telegram_batch_test_log.csv"

async def get_chat_id(session):
    async with session.get(f"{BOT_API}/getUpdates") as resp:
        data = await resp.json()
        if "result" in data and len(data["result"]) > 0:
            for entry in data["result"]:
                try:
                    return entry["message"]["chat"]["id"]
                except KeyError:
                    continue
        return None

async def send_and_log(session, chat_id, query, writer):
    async with session.post(f"{BOT_API}/sendMessage", json={"chat_id": chat_id, "text": query}) as r1:
        await asyncio.sleep(2)  # wait for response
        async with session.get(f"{BOT_API}/getUpdates") as r2:
            updates = await r2.json()
            replies = [m for m in updates["result"] if m.get("message", {}).get("text") != query]
            latest = replies[-1]["message"]["text"] if replies else "❌ No reply"
            status = "✅" if latest and latest.lower() not in ["false", "❌ no reply"] else "❌"
            print(f"{status} Sent: {query} → Response: {latest[:50]}")
            writer.writerow([query, latest.strip(), status])

async def main():
    async with aiohttp.ClientSession() as session:
        chat_id = await get_chat_id(session)
        if not chat_id:
            print("❌ Could not determine chat_id. Please message the bot first.")
            return

        with open(CSV_LOG, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Query", "Response", "Status"])
            for query in TEST_PROMPTS:
                await send_and_log(session, chat_id, query, writer)

if __name__ == "__main__":
    asyncio.run(main())
