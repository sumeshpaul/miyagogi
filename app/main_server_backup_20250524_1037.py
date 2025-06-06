# app/main_server.py

import os
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Query, Body
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from woocommerce import API
import sqlite3
import textwrap
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import difflib
from telegram.constants import ParseMode
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict
from pydantic import BaseModel
import aiohttp
from search_products import search_products_by_keywords
from peft import PeftModel
import html
import csv
from collections import defaultdict
import time
import gradio as gr
import uvicorn
import random

# ──────────────────────────── Logging & Environment ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WC_URL = os.environ["WC_URL"]
WC_KEY = os.environ["WC_KEY"]
WC_SECRET = os.environ["WC_SECRET"]

app = FastAPI()
DB_PATH = os.getenv("PRODUCT_DB", "data/products.db")

# ────────────────────────────── WooCommerce Client ──────────────────────────────
wcapi = API(url=WC_URL, consumer_key=WC_KEY, consumer_secret=WC_SECRET, version="wc/v3")

# ───────────── Hermes + LoRA Model Load (CPU Mode) ─────────────
try:
    # 1. Safe GPU-friendly quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 2. Load base model using device_map and memory cap
    base_model = AutoModelForCausalLM.from_pretrained(
        "/app/base_model",
        quantization_config=quant_config,
        device_map={"": 0},
        max_memory={0: "14GiB"},
        local_files_only=True
    )

    # 3. Load LoRA adapter on top
    model = PeftModel.from_pretrained(
        base_model,
        "/app/final_lora_model_v2",
        torch_dtype=torch.float16,
        local_files_only=True
    )

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/app/final_lora_model_v2",
        use_fast=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("✅ Hermes (LoRA) model and tokenizer loaded successfully with GPU 4bit config.")
except Exception as e:
    import traceback
    logger.error("\u274C Failed to load Hermes model with GPU.")
    traceback.print_exc()
    model = None
    tokenizer = None


# ──────────────────────── Per-user memory store ─────────────────────────
user_chat_memory = defaultdict(lambda: [])
MEMORY_MAX_TURNS = 3

# ──────────────────────────────── FastAPI Init ──────────────────────────────────
app = FastAPI()

# ───────────────────────────── Pydantic Response Model ──────────────────────────
class LogEntry(BaseModel):
    timestamp: str
    user_id: str
    query: str
    response: str

class ProductQuery(BaseModel):
    query: str

# ──────────────────────────────── SQLite Logger ─────────────────────────────────
def log_query(user_id, query, response):
    conn = sqlite3.connect("/app/data/query_logs.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT,
            query TEXT,
            response TEXT
        )
    """)
    c.execute("INSERT INTO logs (user_id, query, response) VALUES (?, ?, ?)", (user_id, query, response))
    conn.commit()
    conn.close()

# ────────────────────────────── Telegram Formatter ─────────────────────────────
def format_telegram_table(results: Dict[str, List[Dict]], max_per_brand: int = 10) -> str:
    sections = []
    for brand, items in results.items():
        sections.append(f"<b>Brand: {html.escape(brand)}</b>")
        for item in items[:max_per_brand]:
            name = html.escape(item["name"])
            price = f"AED {item['price']}"
            stock = "✅ In stock" if item["stock"] == "instock" else "❌ Out"
            summary = html.escape(item.get("summary", ""))
            link = html.escape(item.get("link", "#"))
            row = f'<a href="{link}">{name}</a>\n{summary}\n<b>Price:</b> {price} | <b>Stock:</b> {stock}'
            sections.append(row)
    return "\n\n".join(sections)

# ──────────────────────────────── Telegram Handler ─────────────────────────────
FASTAPI_URL = "http://localhost:8000/search-products"
MAX_MESSAGE_LENGTH = 4000

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    query = update.message.text.strip()
    reply_text = ""

    async with aiohttp.ClientSession() as session:
        try:
            # 🔍 Step 1: Interpret user intent
            async with session.post("http://localhost:8000/interpret", json={"query": query}) as resp:
                interpreted = await resp.json()
                keywords = interpreted.get("keywords", query)

            # 📦 Step 2: Try structured product search
            async with session.post("http://localhost:8000/search-products", json={"query": keywords}) as resp:
                result = await resp.json()
                search_text = result.get("response", "")

            # 💬 Step 3: Always fetch LLM-enhanced response
            async with session.post(f"http://localhost:8000/ask?user_id={user_id}", json={"query": query}) as chat_resp:
                chat_result = await chat_resp.json()
                chat_text = chat_result.get("response", "")

            # ✅ FINAL RESPONSE PRIORITY
            if chat_text:
                reply_text = chat_text
            elif search_text and "No matching products" not in search_text:
                reply_text = search_text
            else:
                reply_text = "❌ No relevant products or insights found."

        except Exception as e:
            reply_text = f"❌ Error: {str(e)}"

    # 🧾 Chunk large responses
    if len(reply_text) > MAX_MESSAGE_LENGTH:
        chunks = [reply_text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(reply_text), MAX_MESSAGE_LENGTH)]
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)

    # 📊 Log interaction
    log_query(user_id, query, reply_text)

# ──────────────────────────────── API Endpoints ─────────────────────────────────
@app.get("/recent-logs", response_model=List[LogEntry])
def recent_logs(limit: int = Query(10)):
    try:
        conn = sqlite3.connect("/app/data/query_logs.db")
        c = conn.cursor()
        c.execute(
            "SELECT timestamp, user_id, query, response FROM logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()
        conn.close()
        return [
            LogEntry(timestamp=ts, user_id=uid, query=q, response=r)
            for ts, uid, q, r in rows
        ]
    except Exception as e:
        logger.error(f"❌ Failed to fetch recent logs: {e}")
        return JSONResponse(
            status_code=500, content={"error": "Could not retrieve logs"}
        )


@app.get("/admin/logs/search", response_model=List[LogEntry])
def search_logs(
    user_id: str = Query(None), keyword: str = Query(None), since: str = Query(None)
):
    try:
        conn = sqlite3.connect("/app/data/query_logs.db")
        c = conn.cursor()
        clauses = []
        params = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if keyword:
            clauses.append("(query LIKE ? OR response LIKE ?)")
            params += [f"%{keyword}%"] * 2
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where_clause = "WHERE " + " AND ".join(clauses) if clauses else ""
        c.execute(
            f"SELECT timestamp, user_id, query, response FROM logs {where_clause} ORDER BY timestamp DESC LIMIT 100",
            params,
        )
        rows = c.fetchall()
        conn.close()
        return [
            LogEntry(timestamp=r[0], user_id=r[1], query=r[2], response=r[3])
            for r in rows
        ]
    except Exception as e:
        logger.error(f"❌ Log search failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Search failed"})


@app.delete("/admin/logs/delete")
def delete_logs(user_id: str = Query(None), before: str = Query(None)):
    try:
        conn = sqlite3.connect("/app/data/query_logs.db")
        c = conn.cursor()
        clauses = []
        params = []
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        if before:
            clauses.append("timestamp < ?")
            params.append(before)
        if not clauses:
            return {"error": "Specify at least one filter to delete."}
        where_clause = "WHERE " + " AND ".join(clauses)
        c.execute(f"DELETE FROM logs {where_clause}", params)
        deleted = c.rowcount
        conn.commit()
        conn.close()
        return {"deleted": deleted}
    except Exception as e:
        logger.error(f"❌ Delete failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Delete failed"})


@app.get("/admin/logs/export")
def export_logs():
    try:
        conn = sqlite3.connect("/app/data/query_logs.db")
        c = conn.cursor()
        c.execute("SELECT timestamp, user_id, query, response FROM logs")
        rows = c.fetchall()
        conn.close()

        def iter_csv():
            yield "timestamp,user_id,query,response\n"
            for row in rows:
                yield ",".join(
                    ['"' + str(field).replace('"', '""') + '"' for field in row]
                ) + "\n"

        return StreamingResponse(
            iter_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=logs.csv"},
        )
    except Exception as e:
        logger.error(f"❌ CSV export failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Export failed"})


@app.get("/")
def root():
    return {"message": "Miyagogi Bot v1 with Hermes is running"}


@app.post("/search-products")
async def search_products_endpoint(data: ProductQuery):
    keywords = data.query.lower().split()
    results = search_products_by_keywords(keywords, DB_PATH)

    if not results:
        if model and tokenizer:
            try:
                prompt = f"### Instruction:\nThe user searched: '{data.query}'. Suggest what they might want.\n\n### Suggestion:"
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                with torch.no_grad():
                    start = time.time()

                    output = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=300,
                        do_sample=False,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                    logger.info(f"🧠 Hermes response time: {time.time() - start:.2f}s")
                reply = (
                    tokenizer.decode(output[0], skip_special_tokens=True)
                    .split("###")[-1]
                    .strip()
                )
                return {
                    "response": f"❌ No matching products found.\n\n🤖 Suggestion: {reply}"
                }
            except Exception as e:
                logger.error(f"LLM fallback failed: {e}")
                return {
                    "response": "❌ No matching products found. (Model failed to suggest alternative)"
                }

        return {"response": "❌ No matching products found."}

    return {"response": format_telegram_table(results)}

@app.post("/interpret")
async def interpret_query(data: ProductQuery):
    if not model or not tokenizer:
        return {"error": "Hermes model not available."}

    try:
        prompt = f"### Instruction:\nConvert this user input into 3-5 keywords related to beauty or product search:\n\n'{data.query}'\n\n### Keywords:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ← fixed device

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )


        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        keywords = decoded.split("### Keywords:")[-1].strip()
        return {"keywords": keywords}

    except Exception as e:
        logger.error(f"/interpret failed: {e}")
        return {"error": "Interpretation failed."}

# ──────────────────────────────── /ask Endpoint for Hermes Chat (Final Fix) ─────────────────────────────
from collections import defaultdict

user_last_product_match = defaultdict(str)

@app.post("/ask", response_model=Dict[str, str])
async def ask_hermes(
    data: ProductQuery, user_id: str = Query(..., description="Telegram user ID")
):
    if not model or not tokenizer:
        return {"error": "Hermes model not available."}

    query_text = data.query.strip().lower()
    messages = user_chat_memory[user_id][-MEMORY_MAX_TURNS:]
    messages.append({"role": "user", "content": data.query})

    price_related_keywords = ["price", "cost", "availability", "stock", "how much", "in stock"]
    is_price_question = any(word in query_text for word in price_related_keywords)

    brand_keywords = ["thuya", "noemi", "fairy", "lash"]
    brand_match = next((b for b in brand_keywords if b in query_text), None)

    db_context = ""
    best_match = None
    injected_product_name = None

    try:
        recent_product_text = query_text
        if is_price_question and user_last_product_match[user_id]:
            recent_product_text = user_last_product_match[user_id]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if brand_match and not is_price_question:
            cursor.execute("""
                SELECT name, price, stock_status FROM products
                WHERE LOWER(name) LIKE ?
                ORDER BY LENGTH(name) ASC LIMIT 5
            """, (f"%{brand_match}%",))
            rows = cursor.fetchall()
            if rows:
                db_context = f"Here are some popular {brand_match.capitalize()} products:\n"
                for name, price, stock in rows:
                    stock_status = "✅ In stock" if stock == "instock" else "❌ Out of stock"
                    db_context += f"- {name} — AED {price} ({stock_status})\n"

        else:
            cursor.execute("SELECT DISTINCT LOWER(name) FROM products")
            product_names = [row[0] for row in cursor.fetchall()]
            best_score = 0
            for pname in product_names:
                score = difflib.SequenceMatcher(None, recent_product_text, pname).ratio()
                if score > best_score:
                    best_match = pname
                    best_score = score

            if best_score > 0.5:
                user_last_product_match[user_id] = best_match
                cursor.execute("""
                    SELECT name, price, stock_status FROM products 
                    WHERE LOWER(name) LIKE ?
                    ORDER BY LENGTH(name) ASC LIMIT 1
                """, (f"%{best_match}%",))
                row = cursor.fetchone()
                if row:
                    name, price, stock = row
                    injected_product_name = name
                    stock_status = "✅ In stock" if stock == "instock" else "❌ Out of stock"
                    db_context = (
                        f"<b>{html.escape(name)}</b> is priced at <b>AED {price}</b>.\n"
                        f"Stock status: <b>{stock_status}</b>."
                    )
                    logger.info(f"✅ Injected product info for: {name}")

        conn.close()

    except Exception as e:
        logger.error(f"❌ Product DB fetch failed: {e}")

    # Compose prompt
    prompt = (
        "You are a friendly and knowledgeable beauty and skincare assistant named Aura. "
        "Your responses should be helpful, conversational, and gently persuasive."
        + ("\nProduct Info:\n" + db_context if db_context else "")
    )
    for msg in messages:
        role = msg["role"].capitalize()
        prompt += f"\n{role}: {msg['content']}"
    prompt += "\nAssistant:"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=300,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Assistant:")[-1].strip()

        final_response = f"{db_context}\n\n{html.escape(answer)}" if db_context else html.escape(answer)
        final_response += "\n\nFeel free to ask for comparisons or recommendations anytime ✨"

        user_chat_memory[user_id].append({"role": "user", "content": data.query})
        user_chat_memory[user_id].append({"role": "assistant", "content": answer})

        return {"response": final_response}

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.error("❌ CUDA OOM in Hermes")
        return {"error": "Hermes out of memory. Try again."}
    except Exception as e:
        logger.error(f"Hermes failed: {e}")
        return {"error": "Hermes failed."}

# ─────────────────────────────── Bot Lifecycle ──────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        app.bot_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.bot_app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )
        await app.bot_app.initialize()
        await app.bot_app.start()
        await app.bot_app.updater.start_polling()
        logger.info("✅ Telegram bot started successfully.")
    except Exception as e:
        logger.error("❌ Telegram bot failed to start.")
        import traceback

        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown():
    await app.bot_app.updater.stop()
    await app.bot_app.stop()
    await app.bot_app.shutdown()

def gradio_ask(user_input: str):
    user_id = "gradio-user"
    query = {"query": user_input}
    try:
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.post(f"/ask?user_id={user_id}", json=query)
        raw_response = response.json().get("response", "No response.")
        return html.unescape(raw_response)
    except Exception as e:
        return f"❌ Error: {str(e)}"

demo = gr.Interface(
    fn=gradio_ask,
    inputs=gr.Textbox(label="Ask about any Miyagogi product"),
    outputs=gr.Textbox(label="Hermes Response"),
    title="💬 Miyagogi Virtual Assistant",
    description="Powered by Hermes (LoRA) and product database",
    allow_flagging="never"
)

# Optional: Launch Gradio automatically if this script is executed directly
if __name__ == "__main__":
    import threading
    threading.Thread(target=lambda: demo.launch(server_name="0.0.0.0", server_port=7863, share=False)).start()
    uvicorn.run("main_server:app", host="0.0.0.0", port=8000)
