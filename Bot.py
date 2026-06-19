#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexFlow AI — Legal Intake Telegram Bot
========================================
Single-file, production-ready client intake assistant.
No external file dependencies. Everything self-contained.

Requirements:
    pip install python-telegram-bot>=21.0 python-dotenv>=1.0.0 requests>=2.31.0

Optional AI providers (install for full fallback chain):
    pip install google-generativeai>=0.7.0 groq>=0.9.0 openai>=1.35.0

Environment variables (export or place in .env):
    TELEGRAM_BOT_TOKEN
    GEMINI_API_KEY
    GROQ_API_KEY
    XAI_API_KEY
    OPENROUTER_API_KEY
    OLLAMA_URL        (default: http://localhost:11434)
    OLLAMA_MODEL      (default: llama3.2)
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# ───────────────────────────────────────────────
# Environment
# ───────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ───────────────────────────────────────────────
# Telegram
# ───────────────────────────────────────────────
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.request import HTTPXRequest

# ───────────────────────────────────────────────
# HTTP client for Ollama
# ───────────────────────────────────────────────
import requests

# ───────────────────────────────────────────────
# Optional AI providers (graceful degradation)
# ───────────────────────────────────────────────
_AI_AVAILABILITY: Dict[str, bool] = {}

try:
    import google.generativeai as genai
    _AI_AVAILABILITY["gemini"] = True
except ImportError:
    _AI_AVAILABILITY["gemini"] = False
    genai = None  # type: ignore

try:
    from groq import Groq
    _AI_AVAILABILITY["groq"] = True
except ImportError:
    _AI_AVAILABILITY["groq"] = False
    Groq = None  # type: ignore

try:
    from openai import AsyncOpenAI
    _AI_AVAILABILITY["openai"] = True
except ImportError:
    _AI_AVAILABILITY["openai"] = False
    AsyncOpenAI = None  # type: ignore

# ───────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

# ───────────────────────────────────────────────
# Logging (file + console, never sent to user)
# ───────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("LexFlowAI")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(DATA_DIR / "bot.log", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# ───────────────────────────────────────────────
# Conversation States
# ───────────────────────────────────────────────
(
    STATE_ISSUE_TYPE,
    STATE_FULL_NAME,
    STATE_PHONE,
    STATE_CASE_DESC,
    STATE_DOCUMENTS,
    STATE_CONFIRM,
) = range(6)

# ───────────────────────────────────────────────
# System Persona
# ───────────────────────────────────────────────
SYSTEM_PERSONA = (
    "You are LexFlow AI, the senior intake coordinator for a distinguished law firm. "
    "You are refined, composed, and highly professional. "
    "You speak with quiet confidence and precision. "
    "You never use robotic greetings like 'Hello! How can I assist you today?' "
    "You never claim to be a licensed attorney, never provide legal advice, and never predict outcomes. "
    "Your sole purpose is to gather relevant intake details, put the prospective client at ease, "
    "and assure them that their matter will be reviewed with care and discretion. "
    "You are concise. You do not repeat yourself. You do not use exclamation points excessively. "
    "When the conversation calls for it, you are warm — but always restrained and dignified."
)

# ───────────────────────────────────────────────
# Prompts for AI generation per state
# ───────────────────────────────────────────────
INTAKE_PROMPTS = {
    STATE_ISSUE_TYPE: (
        "Welcome the prospective client with composure and brevity. "
        "Introduce yourself as LexFlow AI, the firm's intake coordinator. "
        "Ask them to select the category that best describes their legal matter from the options provided. "
        "Do not be verbose."
    ),
    STATE_FULL_NAME: (
        "Acknowledge their selection naturally. Ask for their full legal name. "
        "Keep it brief and professional."
    ),
    STATE_PHONE: (
        "Acknowledge their name. Ask for the best phone number to reach them. "
        "Be brief."
    ),
    STATE_CASE_DESC: (
        "Ask them to describe their situation in a few sentences. "
        "Reassure them that all communications are confidential. "
        "Be brief — one or two sentences at most."
    ),
    STATE_DOCUMENTS: (
        "Ask what relevant documents or evidence they have — contracts, photos, reports, emails, etc. "
        "Tell them they may upload files directly here. Keep it concise."
    ),
    STATE_CONFIRM: (
        "Present a clean summary of the collected information: issue type, name, phone, description, and documents. "
        "Ask them to confirm everything is accurate or if they would like to make changes. "
        "Close with quiet confidence. Do not make promises about outcomes."
    ),
}

# ───────────────────────────────────────────────
# Fallback prompts (natural re-ask, zero technical exposure)
# ───────────────────────────────────────────────
FALLBACK_PROMPTS = {
    STATE_ISSUE_TYPE: (
        "Good day. I am LexFlow AI, the firm's intake coordinator. "
        "Please select the category that best describes your legal matter."
    ),
    STATE_FULL_NAME: "Thank you. May I have your full legal name, please?",
    STATE_PHONE: "Thank you. And the best phone number to reach you?",
    STATE_CASE_DESC: (
        "Please share a brief description of your situation. "
        "Everything you tell me is held in strict confidence."
    ),
    STATE_DOCUMENTS: (
        "Do you have any relevant documents or evidence — contracts, photos, reports, emails? "
        "You may upload them directly here. Type 'done' when finished."
    ),
    STATE_CONFIRM: (
        "Here is what I have recorded. Please confirm everything is accurate, or let me know what to change."
    ),
}

# ───────────────────────────────────────────────
# Pending question storage for silent error recovery
# ───────────────────────────────────────────────
PENDING_QUESTION: Dict[int, str] = {}

# ───────────────────────────────────────────────
# AI Fallback Chain
# ───────────────────────────────────────────────

async def _generate_ai_response(
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 800,
) -> Optional[str]:
    """
    Attempts AI providers in sequence:
    Gemini → Groq → xAI Grok → OpenRouter → Ollama (local).
    Each provider is retried once after a 2-second delay on failure.
    Returns None only if all providers fail.
    """

    # Provider 1: Google Gemini
    if _AI_AVAILABILITY.get("gemini") and GEMINI_API_KEY and genai:
        for attempt in range(2):
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                history = []
                system_text = ""
                for msg in messages:
                    if msg["role"] == "system":
                        system_text = msg["content"]
                    else:
                        role = "user" if msg["role"] == "user" else "model"
                        history.append({"role": role, "parts": [msg["content"]]})
                if history:
                    if system_text and history[0]["role"] == "user":
                        history[0]["parts"][0] = f"{system_text}\n\n{history[0]['parts'][0]}"
                    current = history.pop()
                    chat = model.start_chat(history=history)
                    response = await asyncio.to_thread(
                        chat.send_message,
                        current["parts"][0],
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        ),
                    )
                    text = response.text.strip()
                    if text:
                        logger.info("AI response via Gemini")
                        return text
            except Exception as e:
                logger.warning("Gemini attempt %d failed: %s", attempt + 1, e)
                if attempt == 0:
                    await asyncio.sleep(2)

    # Provider 2: Groq
    if _AI_AVAILABILITY.get("groq") and GROQ_API_KEY and Groq:
        for attempt in range(2):
            try:
                client = Groq(api_key=GROQ_API_KEY)
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="llama-3.1-70b-versatile",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    logger.info("AI response via Groq")
                    return text
            except Exception as e:
                logger.warning("Groq attempt %d failed: %s", attempt + 1, e)
                if attempt == 0:
                    await asyncio.sleep(2)

    # Provider 3: xAI Grok
    if _AI_AVAILABILITY.get("openai") and XAI_API_KEY and AsyncOpenAI:
        for attempt in range(2):
            try:
                client = AsyncOpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
                response = await client.chat.completions.create(
                    model="grok-2-latest",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    logger.info("AI response via xAI Grok")
                    return text
            except Exception as e:
                logger.warning("xAI Grok attempt %d failed: %s", attempt + 1, e)
                if attempt == 0:
                    await asyncio.sleep(2)

    # Provider 4: OpenRouter
    if _AI_AVAILABILITY.get("openai") and OPENROUTER_API_KEY and AsyncOpenAI:
        for attempt in range(2):
            try:
                client = AsyncOpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                )
                response = await client.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    logger.info("AI response via OpenRouter")
                    return text
            except Exception as e:
                logger.warning("OpenRouter attempt %d failed: %s", attempt + 1, e)
                if attempt == 0:
                    await asyncio.sleep(2)

    # Provider 5: Ollama (local)
    for attempt in range(2):
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                text = data.get("message", {}).get("content", "").strip()
                if text:
                    logger.info("AI response via Ollama")
                    return text
        except Exception as e:
            logger.warning("Ollama attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                await asyncio.sleep(2)

    logger.error("All AI providers exhausted.")
    return None


# ───────────────────────────────────────────────
# Messaging helpers
# ───────────────────────────────────────────────

async def _ask_user(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup=None,
) -> None:
    """
    Send a message to the user and store it as the pending question.
    This allows the error handler to re-ask naturally if something breaks.
    """
    user_id = None
    if update.effective_user:
        user_id = update.effective_user.id
    elif update.callback_query and update.callback_query.from_user:
        user_id = update.callback_query.from_user.id

    if user_id:
        PENDING_QUESTION[user_id] = text

    if update.callback_query and update.effective_message:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text, reply_markup=reply_markup)
    elif update.effective_message:
        await update.effective_message.reply_text(text, reply_markup=reply_markup)
    else:
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id:
            await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


async def _generate_intake_message(state: int, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Generate an intake message using AI, with a natural fallback."""
    intake = context.user_data.get("intake", {})
    context_summary = "\n".join(
        f"{k}: {v}"
        for k, v in intake.items()
        if v is not None
        and k not in ("started_at", "completed_at", "user_id", "username", "file_ids", "awaiting_edit", "final_summary")
    )

    messages = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "user", "content": INTAKE_PROMPTS[state]},
    ]
    if context_summary:
        messages.append({"role": "user", "content": f"Context so far:\n{context_summary}"})

    ai_text = await _generate_ai_response(messages, temperature=0.6, max_tokens=400)
    if ai_text:
        return ai_text

    return FALLBACK_PROMPTS.get(state, "Could you share that detail with me once more?")


async def _generate_case_summary_text(intake: Dict[str, Any]) -> str:
    """Generate a polished case summary via AI, with a structured fallback."""
    prompt = f"""Format the following client intake into a professional case summary for an attorney.

Issue Type: {intake.get("issue_type", "N/A")}
Full Name: {intake.get("full_name", "N/A")}
Phone: {intake.get("phone", "N/A")}
Case Description: {intake.get("case_description", "N/A")}
Documents: {intake.get("documents", "N/A")}
File Uploads: {len(intake.get("file_ids", []))} file(s)

Rules:
- Use professional formatting with clear sections
- Do NOT add legal opinions, advice, or analysis
- Do NOT suggest strategies or outcomes
- Note that no legal advice was provided during intake
- Preserve all facts exactly as stated by the client"""

    messages = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "user", "content": prompt},
    ]

    ai_summary = await _generate_ai_response(messages, temperature=0.3, max_tokens=1200)
    if ai_summary:
        return ai_summary

    # Structured fallback (never fails)
    files_note = ""
    if intake.get("file_ids"):
        files_note = f"\n  Uploaded Files: {len(intake['file_ids'])} file(s) logged."
    return f"""╔══════════════════════════════════════════════════════════════════╗
║                    LEXFLOW AI — CASE INTAKE SUMMARY              ║
╠══════════════════════════════════════════════════════════════════╣
  Intake Date: {intake.get("started_at", "N/A")}
  Client Telegram ID: {intake.get("user_id", "N/A")}
  Client Username: @{intake.get("username") or "N/A"}

  ┌─ CLIENT INFORMATION ──────────────────────────────────────────┐
  │ Full Name:        {intake.get("full_name") or "Not provided"}
  │ Phone Number:     {intake.get("phone") or "Not provided"}
  └───────────────────────────────────────────────────────────────┘

  ┌─ CASE CLASSIFICATION ─────────────────────────────────────────┐
  │ Issue Type:       {intake.get("issue_type") or "Not specified"}
  └───────────────────────────────────────────────────────────────┘

  ┌─ CASE DESCRIPTION ──────────────────────────────────────────┐
  {intake.get("case_description") or "No description provided."}
  └───────────────────────────────────────────────────────────────┘

  ┌─ DOCUMENTS & EVIDENCE ────────────────────────────────────────┐
  {intake.get("documents") or "No documents listed."}{files_note}
  └───────────────────────────────────────────────────────────────┘

  ⚖️  INTAKE PROTOCOL COMPLIANCE
  • No legal advice was provided during this intake session
  • All information was collected by automated assistant LexFlow AI
  • Client awaits attorney review and direct contact
  • Recommended response time: 24–48 business hours

╚══════════════════════════════════════════════════════════════════╝"""


def _save_intake(intake: Dict[str, Any]) -> Optional[Path]:
    """Persist intake data as JSON for attorney review."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    user_id = intake.get("user_id", "unknown")
    filename = f"case_{user_id}_{timestamp}.json"
    filepath = DATA_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(intake, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Intake saved to %s", filepath)
    return filepath


# ───────────────────────────────────────────────
# Command Handlers
# ───────────────────────────────────────────────

async def cmd_start(update, context):
    """Initiates the intake conversation — full implementation in main.py."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from datetime import datetime, timezone

    user = update.effective_user
    logger.info("New intake started by user %d (%s)", user.id, user.username or "N/A")

    context.user_data["intake"] = {
        "user_id": user.id,
        "username": user.username,
        "issue_type": None,
        "full_name": None,
        "phone": None,
        "case_description": None,
        "documents": None,
        "file_ids": [],
        "additional_notes": None,
        "awaiting_edit": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    keyboard = [
        [
            InlineKeyboardButton("Criminal", callback_data="Criminal"),
            InlineKeyboardButton("Civil", callback_data="Civil"),
        ],
        [
            InlineKeyboardButton("Family", callback_data="Family"),
            InlineKeyboardButton("Property", callback_data="Property"),
        ],
        [
            InlineKeyboardButton("Corporate", callback_data="Corporate"),
            InlineKeyboardButton("Immigration", callback_data="Immigration"),
        ],
        [
            InlineKeyboardButton("Employment", callback_data="Employment"),
            InlineKeyboardButton("Other", callback_data="Other"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    text = await _generate_intake_message(STATE_ISSUE_TYPE, context)
    await _ask_user(update, context, text, reply_markup=reply_markup)
    return STATE_ISSUE_TYPE
