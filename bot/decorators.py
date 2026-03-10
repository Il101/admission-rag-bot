import inspect

from sqlalchemy.future import select
from telegram.constants import ChatType

from bot.db import Admin, BannedUserOrChat


def _strip_extra(func, kwargs: dict) -> dict:
    """Remove kwargs the target *func* doesn't accept (unless it has **kwargs)."""
    sig = inspect.signature(func)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # function accepts **kwargs — pass everything
    return {k: v for k, v in kwargs.items() if k in params}


def with_db_session(session_param_name="db_session"):
    def decorator(handler):
        async def wrapper(*args, **kwargs):
            db_session_factory = kwargs.pop(session_param_name)
            async with db_session_factory() as session:
                kwargs[session_param_name] = session
                kwargs["db_session_factory"] = db_session_factory
                await handler(*args, **_strip_extra(handler, kwargs))

        return wrapper

    return decorator


def admin_only(
    session_param_name="db_session",
    should_can_add_info=True,
    should_can_add_admins=True,
):
    def decorator(handler):
        async def wrapper(*args, **kwargs):
            update, context = args
            session = kwargs[session_param_name]

            admin = await session.get(Admin, update.effective_user.id)
            if (
                admin is None
                or (should_can_add_info and not admin.can_add_info)
                or (should_can_add_admins and not admin.can_add_new_admins)
            ):
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    reply_to_message_id=update.effective_message.id,
                    text="У вас недостаточно прав для этого действия.",
                )
                return
            else:
                await handler(*args, **_strip_extra(handler, kwargs))

        return wrapper

    return decorator


def filter_banned(session_param_name="db_session"):
    def decorator(handler):
        async def wrapper(*args, **kwargs):
            update, _ = args
            session = kwargs[session_param_name]

            banned_user = await session.get(BannedUserOrChat, update.effective_user.id)
            if banned_user is not None:
                return

            # check whether chat is banned iff chat is a group
            chat_type = update.effective_chat.type
            if chat_type == ChatType.GROUP or chat_type == ChatType.SUPERGROUP:
                banned_chat_query = (
                    select(BannedUserOrChat)
                    .where(BannedUserOrChat.tg_id == update.effective_chat.id)
                    .where(BannedUserOrChat.is_user == False)  # noqa
                )
                banned_chat = await session.scalar(banned_chat_query)
                if banned_chat is not None:
                    return

            await handler(*args, **_strip_extra(handler, kwargs))

        return wrapper

    return decorator
