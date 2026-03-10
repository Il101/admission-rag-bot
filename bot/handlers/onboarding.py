from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler, CommandHandler, CallbackQueryHandler
from bot.keyboards import (
    document_keyboard,
    target_level_keyboard,
    german_level_keyboard,
    english_level_keyboard,
    suggested_questions_keyboard,
)
from bot.db import upsert_user, get_user, get_user_memory
from bot.memory import get_fallback_buttons, JOURNEY_STAGES

# Define states for ConversationHandler (country removed, english added)
DOCUMENT, TARGET_LEVEL, GERMAN_LEVEL, ENGLISH_LEVEL = range(4)

def build_cmd_start(db_session):
    async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Entry point for onboarding FSM. Checks if user already exists."""
        tg_id = update.effective_user.id
        
        async with db_session() as session:
            user = await get_user(session, tg_id)
            
        # If user already has a profile, give a contextual welcome
        if user and user.country:
            # Load memory for personalized greeting
            async with db_session() as session:
                memory = await get_user_memory(session, tg_id)

            journey_state = memory.get("journey_state")
            summary = memory.get("conversation_summary")

            # Count discussed stages
            discussed = []
            pending = []
            if journey_state:
                for stage_id, status in journey_state.items():
                    if stage_id.startswith("_"):
                        continue
                    stage_info = JOURNEY_STAGES.get(stage_id)
                    if not stage_info:
                        continue
                    if status == "discussed":
                        discussed.append(stage_info["label"])
                    else:
                        pending.append(stage_info["label"])

            # Build personalized greeting
            if summary and len(summary) > 20:
                text = (
                    f"👋 С возвращением! Рад тебя снова видеть.\n\n"
                    f"📝 <b>Где мы остановились:</b>\n"
                    f"{summary}\n\n"
                )
                if discussed:
                    text += f"✅ Обсудили: {len(discussed)} из {len(JOURNEY_STAGES)} этапов\n"
                if pending:
                    text += f"⬜ Осталось: {', '.join(pending[:3])}"
                    if len(pending) > 3:
                        text += f" и ещё {len(pending) - 3}"
                    text += "\n"
                text += "\nЧем могу помочь сегодня? 👇"
            else:
                text = (
                    f"👋 С возвращением! Рад тебя снова видеть.\n\n"
                    f"Я готов отвечать на твои вопросы о поступлении в Австрию! 👇"
                )

            # Suggest buttons based on undiscussed stages
            starter_questions = get_fallback_buttons(journey_state)
            
            if update.message:
                sent_msg = await update.message.reply_text(text, parse_mode='HTML')
            elif update.callback_query:
                sent_msg = await update.callback_query.message.reply_text(text, parse_mode='HTML')
            
            if sent_msg:
                from bot.handlers.rag import _save_suggested
                keyboard = suggested_questions_keyboard(starter_questions, msg_id=sent_msg.message_id)
                await sent_msg.edit_reply_markup(reply_markup=keyboard)
                _save_suggested(context, None, None, starter_questions, msg_id=sent_msg.message_id)
            
            return ConversationHandler.END

        # Reset user_data context on fresh /start
        context.user_data["onboarding"] = {"countryScope": "RU"}  # default Russia
        
        text = (
            "👋 Привет! Я — Агент по поступлению в вузы Австрии.\n\n"
            "Я работаю на базе проверенных документов австрийских министерств "
            "и помогу тебе собрать точный чеклист.\n\n"
            "Я также сохраняю историю твоих сообщений для лучшего понимания контекста разговора.\n"
            "Ты можешь запросить удаление всех своих данных командой /delete_my_data.\n\n"
            "📄 Какое у тебя сейчас есть образование (документ)?"
        )
        
        # Message vs Callback — go directly to DOCUMENT step
        if update.message:
            await update.message.reply_text(text, reply_markup=document_keyboard())
        elif update.callback_query:
            await update.callback_query.message.reply_text(text, reply_markup=document_keyboard())

        return DOCUMENT
    return cmd_start

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves document, asks for target level."""
    query = update.callback_query
    await query.answer()
    
    doc_val = query.data.split("_")[1]
    context.user_data["onboarding"]["document"] = doc_val
    
    text = "🎓 Куда хочешь поступить в Австрии?"
    await query.edit_message_text(text, reply_markup=target_level_keyboard())
    return TARGET_LEVEL

async def handle_target_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves target level, asks for german level."""
    query = update.callback_query
    await query.answer()
    
    target_val = query.data.split("_")[1]
    context.user_data["onboarding"]["targetLevel"] = target_val
    
    text = "🗣 Какой у тебя уровень немецкого языка?"
    await query.edit_message_text(text, reply_markup=german_level_keyboard())
    return GERMAN_LEVEL

async def handle_german_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves german level, asks for english level."""
    query = update.callback_query
    await query.answer()
    
    ger_val = query.data.split("_")[1]
    context.user_data["onboarding"]["germanLevel"] = ger_val
    
    text = "🇬🇧 Какой у тебя уровень английского языка?"
    await query.edit_message_text(text, reply_markup=english_level_keyboard())
    return ENGLISH_LEVEL

def build_english_level_handler(db_session):
    async def handle_english_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Saves english level, finishes onboarding."""
        query = update.callback_query
        await query.answer()
        
        eng_val = query.data.split("_")[1]
        context.user_data["onboarding"]["englishLevel"] = eng_val
        
        # Save to database
        onboarding_data = context.user_data.get("onboarding", {})
        user_data = {
            "country": onboarding_data.get("countryScope", "RU"),
            "document_type": onboarding_data.get("document"),
            "target_level": onboarding_data.get("targetLevel"),
            "german_level": onboarding_data.get("germanLevel"),
            "english_level": onboarding_data.get("englishLevel"),
        }
        
        # Save to PostgreSQL
        tg_id = update.effective_user.id
        async with db_session() as session:
            await upsert_user(session, tg_id, user_data)
        
        # Generate profile-aware starter questions
        target = onboarding_data.get("targetLevel", "bachelor")
        starter_questions = []
        if target == "bachelor":
            starter_questions = [
                "📋 С чего мне начать подготовку?",
                "📄 Нужна ли мне нострификация?",
                "🏫 Какие вузы мне подходят?",
            ]
        else:
            starter_questions = [
                "📋 С чего мне начать подготовку?",
                "🎓 Как признают мой диплом?",
                "🏫 Какие магистерские программы есть?",
            ]

        # Final message
        text = (
            "🎯 Отлично! Твой профиль сохранён.\n\n"
            "Теперь я — твой проводник по поступлению! "
            "Я помогу разобраться с каждым шагом: от документов до переезда.\n\n"
            "С чего начнём? 👇"
        )

        # Final message with message-specific suggestions
        sent_msg = await query.edit_message_text(text)
        
        from bot.handlers.rag import _save_suggested
        keyboard = suggested_questions_keyboard(starter_questions, msg_id=sent_msg.message_id)
        await sent_msg.edit_reply_markup(reply_markup=keyboard)
        _save_suggested(context, None, None, starter_questions, msg_id=sent_msg.message_id)
        
        # End conversation, returning to standard RAG chat loop
        return ConversationHandler.END
    return handle_english_level

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fallback if user cancels onboarding."""
    if update.message:
        await update.message.reply_text("Регистрация прервана. Напиши /start, чтобы начать заново.")
    return ConversationHandler.END

def build_onboarding_handler(db_session):
    return ConversationHandler(
        entry_points=[CommandHandler("start", build_cmd_start(db_session))],
        states={
            DOCUMENT: [CallbackQueryHandler(handle_document, pattern="^doc_")],
            TARGET_LEVEL: [CallbackQueryHandler(handle_target_level, pattern="^target_")],
            GERMAN_LEVEL: [CallbackQueryHandler(handle_german_level, pattern="^german_")],
            ENGLISH_LEVEL: [CallbackQueryHandler(build_english_level_handler(db_session), pattern="^english_")],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

