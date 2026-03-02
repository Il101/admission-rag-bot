import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.decorators import with_db_session
from bot.db import delete_user_data

logger = logging.getLogger(__name__)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "Этот бот поддерживает следующие пользовательские команды:\n\n"
            "/start - Показать описание бота\n"
            "/help - Показать доступные команды\n"
            "/ans - Ответить на вопрос\n"
            "/ans_rep - Ответить на вопрос из отвеченного сообщения\n"
            "/docs - Предоставить релевантные ссылки\n"
            "/docs_rep - Предоставить релевантные ссылки для отвеченного сообщения\n"
            "/delete_my_data - Удалить все мои данные и историю обращений"
        ),
    )

@with_db_session()
async def delete_data(update: Update, context: ContextTypes.DEFAULT_TYPE, db_session):
    await delete_user_data(db_session, update.effective_user.id)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="✅ Все твои данные удалены.",
    )

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Извините, указанная команда не поддерживается.",
    )

async def reaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sometimes Telegram sends message_reaction updates even though the bot asks only
    for message updates. This handler silently ignores reactions.
    """
    pass


async def ignore(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sometimes Telegram sends edited_message updates on simple message reactions
    that cause running the whole RAG pipeline once again. This handler just ignores all
    edited_message updates
    """
    pass


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception", exc_info=context.error)
