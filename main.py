#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexFlow AI — main entrypoint
Completes Bot.py (which is truncated) with full conversation handlers,
error handling, and a lightweight health-check HTTP server on port 8000.
"""

import os
import sys
import asyncio
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Telegram ──────────────────────────────────────────────────────────────────
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ── Import helpers from Bot.py ─────────────────────────────────────────────
from Bot import (
    TELEGRAM_BOT_TOKEN,
    DATA_DIR,
    PENDING_QUESTION,
    STATE_ISSUE_TYPE,
    STATE_FULL_NAME,
    STATE_PHONE,
    STATE_CASE_DESC,
    STATE_DOCUMENTS,
    STATE_CONFIRM,
    FALLBACK_PROMPTS,
    _generate_ai_response,
    _ask_user,
    _generate_intake_message,
    _generate_case_summary_text,
    _save_intake,
    cmd_start,
    SYSTEM_PERSONA,
    logger,
)

# ── Conversation States (re-exported for local use) ───────────────────────────
END = ConversationHandler.END

# ────────────────────────────────────────────────────────────────────────────
# Health-check HTTP server (runs in background thread)
# ────────────────────────────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK - LexFlow AI bot is running")

    def log_message(self, format, *args):  # silence access logs
        pass


def _start_health_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Health-check HTTP server started on port %d", port)


# ────────────────────────────────────────────────────────────────────────────
# Issue-type callback handler
# ────────────────────────────────────────────────────────────────────────────

async def cb_issue_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the inline keyboard button press for issue type."""
    query = update.callback_query
    await query.answer()
    issue_type = query.data
    context.user_data.setdefault("intake", {})["issue_type"] = issue_type
    logger.info("User %d selected issue type: %s", update.effective_user.id, issue_type)
    text = await _generate_intake_message(STATE_FULL_NAME, context)
    await _ask_user(update, context, text)
    return STATE_FULL_NAME


# ────────────────────────────────────────────────────────────────────────────
# Full name handler
# ────────────────────────────────────────────────────────────────────────────

async def handle_full_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    full_name = update.message.text.strip()
    context.user_data["intake"]["full_name"] = full_name
    text = await _generate_intake_message(STATE_PHONE, context)
    await _ask_user(update, context, text)
    return STATE_PHONE


# ────────────────────────────────────────────────────────────────────────────
# Phone handler
# ────────────────────────────────────────────────────────────────────────────

async def handle_phone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    phone = update.message.text.strip()
    context.user_data["intake"]["phone"] = phone
    text = await _generate_intake_message(STATE_CASE_DESC, context)
    await _ask_user(update, context, text)
    return STATE_CASE_DESC


# ────────────────────────────────────────────────────────────────────────────
# Case description handler
# ────────────────────────────────────────────────────────────────────────────

async def handle_case_desc(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    desc = update.message.text.strip()
    context.user_data["intake"]["case_description"] = desc
    text = await _generate_intake_message(STATE_DOCUMENTS, context)
    await _ask_user(update, context, text)
    return STATE_DOCUMENTS


# ────────────────────────────────────────────────────────────────────────────
# Documents handler (accepts text and file uploads)
# ────────────────────────────────────────────────────────────────────────────

async def handle_documents(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    intake = context.user_data["intake"]

    # Handle file uploads
    if update.message.document:
        file_id = update.message.document.file_id
        intake.setdefault("file_ids", []).append(file_id)
        await update.message.reply_text(
            "File received. Send more files or type 'done' to continue."
        )
        return STATE_DOCUMENTS

    if update.message.photo:
        file_id = update.message.photo[-1].file_id
        intake.setdefault("file_ids", []).append(file_id)
        await update.message.reply_text(
            "Photo received. Send more or type 'done' to continue."
        )
        return STATE_DOCUMENTS

    # Handle text ("done" or doc description)
    text = update.message.text.strip()
    if text.lower() == "done":
        if not intake.get("documents"):
            intake["documents"] = "None"
    else:
        intake["documents"] = text

    # Move to confirmation
    summary_text = await _generate_case_summary_text(intake)
    confirm_text = await _generate_intake_message(STATE_CONFIRM, context)

    keyboard = [
        [
            InlineKeyboardButton("✅ Confirm", callback_data="CONFIRM"),
            InlineKeyboardButton("✏️ Edit", callback_data="EDIT"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    full_msg = f"{summary_text}\n\n{confirm_text}"
    intake["final_summary"] = summary_text
    await _ask_user(update, context, full_msg, reply_markup=reply_markup)
    return STATE_CONFIRM


# ────────────────────────────────────────────────────────────────────────────
# Confirmation handler
# ────────────────────────────────────────────────────────────────────────────

async def cb_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    intake = context.user_data.get("intake", {})

    if query.data == "CONFIRM":
        intake["completed_at"] = datetime.now(timezone.utc).isoformat()
        filepath = _save_intake(intake)
        closing_messages = [
            {"role": "system", "content": SYSTEM_PERSONA},
            {"role": "user", "content": (
                "The client has confirmed their intake. "
                "Thank them with quiet dignity. "
                "Let them know their matter has been logged and an attorney will review it. "
                "Mention a 24-48 business-hour response time. "
                "Be brief — 2-3 sentences maximum."
            )},
        ]
        closing = await _generate_ai_response(closing_messages, temperature=0.5, max_tokens=200)
        if not closing:
            closing = (
                "Your information has been received and logged. "
                "An attorney will review your matter and reach out within 24–48 business hours. "
                "Thank you for your time."
            )
        await query.edit_message_text(closing)
        logger.info("Intake completed for user %d, saved to %s", update.effective_user.id, filepath)
        return END

    elif query.data == "EDIT":
        # Restart the flow
        await query.edit_message_text(
            "Of course. Let's start over. Please use /start to begin again."
        )
        return END

    return STATE_CONFIRM


# ────────────────────────────────────────────────────────────────────────────
# /cancel and /help commands
# ────────────────────────────────────────────────────────────────────────────

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Your intake session has been cancelled. "
        "You may restart at any time with /start."
    )
    context.user_data.clear()
    return END


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "LexFlow AI — Legal Intake Assistant\n\n"
        "Commands:\n"
        "  /start  — Begin a new intake session\n"
        "  /cancel — Cancel the current session\n"
        "  /help   — Show this message\n\n"
        "This assistant collects your details for attorney review. "
        "No legal advice is provided."
    )


# ────────────────────────────────────────────────────────────────────────────
# Error handler
# ────────────────────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Update caused error: %s", context.error, exc_info=True)

    if isinstance(update, Update) and update.effective_user:
        user_id = update.effective_user.id
        pending = PENDING_QUESTION.get(user_id)
        if pending and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "I apologise for the interruption. " + pending
                )
            except Exception:
                pass


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set. Exiting.")
        sys.exit(1)

    # Start health-check server in background
    _start_health_server(port=int(os.getenv("PORT", "8000")))

    # Build the Telegram application
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Conversation handler covering the full intake flow
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            STATE_ISSUE_TYPE: [
                CallbackQueryHandler(cb_issue_type),
            ],
            STATE_FULL_NAME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_full_name),
            ],
            STATE_PHONE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_phone),
            ],
            STATE_CASE_DESC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_case_desc),
            ],
            STATE_DOCUMENTS: [
                MessageHandler(
                    (filters.TEXT | filters.Document.ALL | filters.PHOTO) & ~filters.COMMAND,
                    handle_documents,
                ),
            ],
            STATE_CONFIRM: [
                CallbackQueryHandler(cb_confirm, pattern="^(CONFIRM|EDIT)$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_error_handler(error_handler)

    logger.info("LexFlow AI bot starting — polling Telegram API …")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
