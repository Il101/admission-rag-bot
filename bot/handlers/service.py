import csv
import io
import logging
import zipfile

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from bot.decorators import with_db_session, admin_only
from bot.db import delete_user_data, get_analytics_summary, get_feedback_stats, get_full_report_data, Feedback

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


@with_db_session()
@admin_only(should_can_add_admins=False, should_can_add_info=False)
async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, db_session):
    """Admin-only command: show bot analytics dashboard."""
    # Parse optional days argument: /stats 30
    parts = (update.effective_message.text or "").split()
    days = 7
    if len(parts) > 1:
        try:
            days = int(parts[1])
        except ValueError:
            pass

    try:
        data = await get_analytics_summary(db_session, days=days)
    except Exception as e:
        logger.exception("Failed to get analytics")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"⚠️ Ошибка при сборе аналитики: {e}",
        )
        return

    u = data["users"]
    q = data["queries"]
    fb = data["feedback"]
    m = data["messages"]

    satisfaction = "-"
    if fb["total"] > 0:
        satisfaction = f"{fb['positive'] / fb['total'] * 100:.0f}%"

    text = (
        f"📊 <b>Аналитика за {days} дней</b>\n"
        f"\n"
        f"👥 <b>Пользователи</b>\n"
        f"  Всего: {u['total']}\n"
        f"  Активных за период: {u['active_last_period']}\n"
        f"\n"
        f"⚡ <b>Пайплайн</b>\n"
        f"  Запросов: {q['total']}\n"
        f"  Кэш-хиты: {q['cache_hits']} ({q['cache_hit_rate']:.1%})\n"
        f"  Avg latency: {q['avg_latency_s']}s\n"
        f"  P95 latency: {q['p95_latency_s']}s\n"
        f"  Avg docs/query: {q['avg_docs_per_query']}\n"
        f"\n"
        f"👍 <b>Фидбэк</b>\n"
        f"  👍 {fb['positive']}  👎 {fb['negative']}  (всего {fb['total']})\n"
        f"  Удовлетворённость: {satisfaction}\n"
        f"\n"
        f"💬 <b>Сообщения</b>\n"
        f"  За период: {m['total_last_period']}"
    )

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=text,
        parse_mode=ParseMode.HTML,
    )

    # Build ZIP with detailed CSVs
    try:
        report = await get_full_report_data(db_session, days=days)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for sheet_name, rows in (
                ("users", report["users"]),
                ("feedback", report["feedback"]),
                ("pipeline_logs", report["pipeline_logs"]),
            ):
                if not rows:
                    continue
                csv_buf = io.StringIO()
                writer = csv.DictWriter(csv_buf, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
                zf.writestr(f"{sheet_name}.csv", csv_buf.getvalue())
        zip_buf.seek(0)
        from datetime import date
        filename = f"stats_{date.today().isoformat()}_last{days}d.zip"
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=zip_buf,
            filename=filename,
            caption=f"📎 Подробный отчёт за {days} дней",
        )
    except Exception as e:
        logger.exception("Failed to build report ZIP")
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"⚠️ Не удалось сформировать файл отчёта: {e}",
        )
