# app/main_server.py
# Standard Library
import os
import csv
import time
import html
import sqlite3
import random
import logging
import textwrap
from pathlib import Path
from collections import defaultdict
user_last_product_match = defaultdict(str)

# ───────────────────────────────────────────────
# Third-Party Packages
import torch
import aiohttp
import difflib
import gradio as gr
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict
from fastapi import FastAPI, Request, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ───────────────────────────────────────────────
# Telegram Bot Framework
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from logic_handler import prioritize_reply_logic
import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup

# ───────────────────────────────────────────────
# External APIs and Business Logic
from woocommerce import API
from search_products import search_products_by_keywords

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
            # 🧠 Step 1: Ask Hermes for natural language + Woo fallback
            async with session.post(f"http://localhost:8000/ask?user_id={user_id}", json={"query": query}) as chat_resp:
                chat_result = await chat_resp.json()
                reply_text = html.unescape(chat_result.get("response", ""))

        except Exception as e:
            reply_text = f"❌ Error: {str(e)}"

    # 📤 Chunk response if too long
    if len(reply_text) > MAX_MESSAGE_LENGTH:
        chunks = [reply_text[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(reply_text), MAX_MESSAGE_LENGTH)]
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text(reply_text, parse_mode=ParseMode.HTML)

    # 🧾 Log interaction
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
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name, price, stock_status, brand FROM products")
        all_products = cursor.fetchall()
        conn.close()

        results_by_brand = defaultdict(list)
        for name, price, stock, brand in all_products:
            name_lower = name.lower()
            match_score = max(difflib.SequenceMatcher(None, word, name_lower).ratio() for word in keywords)
            if match_score > 0.5:
                results_by_brand[brand].append({
                    "name": name,
                    "price": price,
                    "stock": stock,
                    "summary": "",
                    "link": "#"
                })

        if results_by_brand:
            return JSONResponse(content={"response": format_telegram_table(results_by_brand)})
        else:
            return JSONResponse(content={"response": ""})
    except Exception as e:
        logger.error(f"/search-products failed: {e}")
        return JSONResponse(content={"response": "❌ Product search failed."})

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

# ─────────────── Woo API Helper ────────────────
def fetch_woo_products(query_text, wc_url, wc_key, wc_secret, fallback_terms=None):
    all_results = []
    seen_ids = set()
    terms_to_try = [query_text] + (fallback_terms or [])
    
    for term in terms_to_try:
        resp = requests.get(
            wc_url + "/wp-json/wc/v3/products",
            params={"search": term, "per_page": 20},
            auth=HTTPBasicAuth(wc_key, wc_secret),
        )
        if resp.status_code != 200:
            continue
        for p in resp.json():
            if p["id"] not in seen_ids:
                all_results.append(p)
                seen_ids.add(p["id"])
        if all_results:
            break  # stop once we find relevant results
    
    return all_results

# ──────────────────────────────── /ask Endpoint for Hermes Chat (Improved) ─────────────────────────────
# PATCHED /ask ENDPOINT (API-Aware + Humanized, Final Fix)

@app.post("/ask", response_model=Dict[str, str])
async def ask_hermes(data: ProductQuery, user_id: str = Query(...)):
    query_text = data.query.strip()
    messages = user_chat_memory[user_id][-MEMORY_MAX_TURNS:]
    messages.append({"role": "user", "content": query_text})

    # 🧠 Smart vague follow-up memory
    vague_keywords = ["how much", "price", "is it in stock", "availability", "is it good", "stock", "how many", "safe", "compare"]
    query_lower = query_text.lower()
    is_vague = any(kw in query_lower for kw in vague_keywords)
    word_count = len(query_lower.split())

    # WooCommerce credentials
    WC_URL = os.getenv("WC_URL") + "/wp-json/wc/v3/products"
    WC_KEY = os.getenv("WC_KEY")
    WC_SECRET = os.getenv("WC_SECRET")
    product = None

    if is_vague and word_count <= 5:
        last_product = user_last_product_match.get(user_id)
        if isinstance(last_product, dict):
            product = last_product
        elif isinstance(last_product, str):
            response = requests.get(WC_URL, params={"search": last_product, "per_page": 1}, auth=HTTPBasicAuth(WC_KEY, WC_SECRET))
            if response.status_code == 200 and response.json():
                product = response.json()[0]
    else:
        response = requests.get(WC_URL, params={"search": query_text, "per_page": 1}, auth=HTTPBasicAuth(WC_KEY, WC_SECRET))
        if response.status_code == 200 and response.json():
            product = response.json()[0]
            user_last_product_match[user_id] = product
        else:
            return {"response": "I couldn’t find a match for that. Could you rephrase or try a brand like Thuya or Noemi?"}

    if product:
        name = product.get("name")
        price = product.get("price", "N/A")
        stock = "in stock" if product.get("stock_status") == "instock" else "out of stock"
        summary_raw = product.get("short_description", "") or product.get("description", "")
        summary = BeautifulSoup(summary_raw, "html.parser").get_text(" ", strip=True)
        short_summary = summary[:300].rsplit(".", 1)[0] + "." if len(summary) > 300 else summary
        product_sentence = f"The {name} is available for AED {price} and it's currently {stock}."
        if short_summary:
            product_sentence += f" {short_summary}"
    else:
        return {"response": "Sorry, I couldn’t find any product information."}

    # 🤖 LLM Prompt
    prompt = (
        "You are Aura, a kind and knowledgeable beauty consultant at Miyagogi.\n"
        "Always answer like a human, clearly and conversationally.\n"
        "Start with the product name, price, and stock. Do not repeat descriptions or sound robotic.\n"
        f"\nProduct Info: {product_sentence}\n"
    )
    for msg in messages:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += "Assistant:"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768, padding=True)
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

        final_response = f"{product_sentence}\n\n{answer}\n\nLet me know if you'd like help comparing this with other products or placing an order 😊"
        user_chat_memory[user_id].append({"role": "user", "content": query_text})
        user_chat_memory[user_id].append({"role": "assistant", "content": answer})
        return {"response": final_response}

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"error": "Hermes ran out of memory."}
    except Exception as e:
        logger.error(f"❌ Hermes error: {e}")
        return {"error": "Hermes model failed."}
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
