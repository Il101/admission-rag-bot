import logging
import os
from functools import partial

import bot.env
import hydra
from omegaconf import DictConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    MessageReactionHandler,
    filters,
)

from bot.db import get_db_sessionmaker
from bot.handlers.management import (
    add_admin,
    add_fact,
    add_fact_from_replied,
    add_facts_from_link,
    ban_user,
    delete_fact,
    unban_user,
)
from bot.handlers.rag import (
    answer,
    answer_to_replied,
    retrieve_docs,
    retrieve_docs_to_replied,
)
from bot.handlers.service import error, help_command, ignore, reaction, unknown, delete_data
from bot.handlers.onboarding import build_onboarding_handler
from bot.handlers.suggested import build_suggested_handler
from crag.simple_rag import SimpleRAG

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def prepare_rag_based_handlers(simple_rag: SimpleRAG, db_session: sessionmaker):
    answer_with_graph = partial(answer, simple_rag=simple_rag, db_session=db_session)
    answer_to_replied_with_graph = partial(
        answer_to_replied, simple_rag=simple_rag, db_session=db_session
    )
    retrieve_docs_with_graph = partial(retrieve_docs, simple_rag=simple_rag, db_session=db_session)
    retrieve_docs_to_replied_with_graph = partial(
        retrieve_docs_to_replied, simple_rag=simple_rag, db_session=db_session
    )

    return {
        "answer": answer_with_graph,
        "answer_to_replied": answer_to_replied_with_graph,
        "retrieve": retrieve_docs_with_graph,
        "retrieve_to_replied": retrieve_docs_to_replied_with_graph,
    }


def prepare_management_handlers(
    simple_rag: SimpleRAG,
    db_session: sessionmaker,
):
    handlers = {}
    handlers["add_fact"] = partial(
        add_fact,
        simple_rag=simple_rag,
        db_session=db_session,
    )
    handlers["add_fact_from_replied"] = partial(
        add_fact_from_replied,
        simple_rag=simple_rag,
        db_session=db_session,
    )
    handlers["add_facts_from_link"] = partial(
        add_facts_from_link,
        simple_rag=simple_rag,
        db_session=db_session,
    )
    handlers["delete_fact"] = partial(
        delete_fact, simple_rag=simple_rag, db_session=db_session
    )
    handlers["ban_user"] = partial(ban_user, db_session=db_session)
    handlers["unban_user"] = partial(unban_user, db_session=db_session)
    handlers["add_admin"] = partial(add_admin, db_session=db_session)

    return handlers


def prepare_handlers(config: DictConfig):
    # Parse DB connection template
    db_url_template = config.bot_db_connection
    db_url = db_url_template.replace("${oc.env:POSTGRES_USER}", os.environ.get("POSTGRES_USER", "")) \
                            .replace("${oc.env:POSTGRES_PASSWORD}", os.environ.get("POSTGRES_PASSWORD", "")) \
                            .replace("${oc.env:POSTGRES_HOST}", os.environ.get("POSTGRES_HOST", "")) \
                            .replace("${oc.env:POSTGRES_DB}", os.environ.get("POSTGRES_DB", ""))
                            
    db_session = get_db_sessionmaker(db_url)
    
    # Sync engine for retrieve logic
    sync_db_url = db_url.replace("+asyncpg", "+psycopg") if "+asyncpg" in db_url else db_url
    if sync_db_url.startswith("postgresql://"):
        sync_db_url = sync_db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    
    db_engine = create_engine(sync_db_url)

    # Instantiate our custom simple RAG
    # We still read the LLM templates from Hydra config for backward compatibility
    simple_rag = SimpleRAG(db_engine, config.get("prompts"))

    rag_handlers = prepare_rag_based_handlers(simple_rag, db_session)
    manag_handlers = prepare_management_handlers(simple_rag, db_session)
    suggested_handler = build_suggested_handler(simple_rag, db_session)

    return rag_handlers, manag_handlers, db_session, suggested_handler


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(config: DictConfig) -> None:
    rag_handlers, manag_handlers, db_session, suggested_callback = prepare_handlers(config)

    tgbot_token = os.getenv("TGBOT_TOKEN")
    application = ApplicationBuilder().token(tgbot_token).build()

    help_handler = CommandHandler("help", help_command)
    reaction_handler = MessageReactionHandler(reaction)
    edited_message_handler = MessageHandler(filters.UpdateType.EDITED_MESSAGE, ignore)
    
    onboarding_handler = build_onboarding_handler(db_session)
    suggested_question_handler = CallbackQueryHandler(suggested_callback, pattern="^suggest_")
    delete_data_handler = CommandHandler("delete_my_data", partial(delete_data, db_session=db_session))

    answer_handler = CommandHandler("ans", rag_handlers["answer"])
    answer_to_replied_handler = CommandHandler(
        "ans_rep", rag_handlers["answer_to_replied"], filters=filters.REPLY
    )
    retrieve_docs_handler = CommandHandler("docs", rag_handlers["retrieve"])
    retrieve_docs_to_replied_handler = CommandHandler(
        "docs_rep", rag_handlers["retrieve_to_replied"], filters=filters.REPLY
    )

    private_message_handler = MessageHandler(
        filters.TEXT & (~filters.COMMAND) & filters.ChatType.PRIVATE,
        rag_handlers["answer"],
    )

    add_fact_handler = CommandHandler("add", manag_handlers["add_fact"])
    delete_fact_handler = CommandHandler("del", manag_handlers["delete_fact"])
    add_fact_from_replied_handler = CommandHandler(
        "add_rep", manag_handlers["add_fact_from_replied"]
    )
    add_facts_from_link_handler = CommandHandler(
        "add_link", manag_handlers["add_facts_from_link"]
    )
    ban_handler = CommandHandler("ban", manag_handlers["ban_user"])
    unban_handler = CommandHandler("unban", manag_handlers["unban_user"])
    add_admin_handler = CommandHandler("add_admin", manag_handlers["add_admin"])

    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application.add_handler(edited_message_handler)
    application.add_handler(onboarding_handler)
    application.add_handler(help_handler)
    application.add_handler(reaction_handler)
    application.add_handler(delete_data_handler)
    application.add_handler(suggested_question_handler)
    application.add_handler(answer_handler)
    application.add_handler(answer_to_replied_handler)
    application.add_handler(retrieve_docs_handler)
    application.add_handler(retrieve_docs_to_replied_handler)
    application.add_handler(private_message_handler)
    application.add_handler(add_fact_handler)
    application.add_handler(add_fact_from_replied_handler)
    application.add_handler(delete_fact_handler)
    application.add_handler(add_facts_from_link_handler)
    application.add_handler(ban_handler)
    application.add_handler(unban_handler)
    application.add_handler(add_admin_handler)
    application.add_handler(unknown_handler)

    application.add_error_handler(error)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
