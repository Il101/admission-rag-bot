import uuid
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from sqlalchemy import BigInteger, Column, ForeignKey, String, DateTime, Text, func, select, delete, update
from sqlalchemy.dialects.postgresql import UUID, JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


class Admin(Base):
    __tablename__ = "admin"

    tg_id = Column(
        BigInteger, primary_key=True, index=True, unique=True, nullable=False
    )
    tg_tag: Mapped[Optional[str]]
    can_add_info: Mapped[bool] = mapped_column(default=True)
    can_add_new_admins: Mapped[bool] = mapped_column(default=True)

    added_by_id = mapped_column(ForeignKey("admin.tg_id"))
    added_by: Mapped["Admin"] = relationship(foreign_keys=[added_by_id])

    banned_users: Mapped[List["BannedUserOrChat"]] = relationship(
        back_populates="banned_by", cascade="all, delete-orphan"
    )


class BannedUserOrChat(Base):
    __tablename__ = "banned"

    tg_id = Column(
        BigInteger, primary_key=True, index=True, unique=True, nullable=False
    )  # can be negative
    is_user: Mapped[bool]

    banned_by_id = mapped_column(ForeignKey("admin.tg_id"))
    banned_by: Mapped["Admin"] = relationship(back_populates="banned_users")


class User(Base):
    __tablename__ = "users"

    tg_id = Column(
        BigInteger, primary_key=True, index=True, unique=True, nullable=False
    )
    country: Mapped[Optional[str]]
    document_type: Mapped[Optional[str]]
    target_level: Mapped[Optional[str]]
    german_level: Mapped[Optional[str]]
    english_level: Mapped[Optional[str]]

    # Hybrid memory fields
    journey_state = Column(JSONB, nullable=True, default=None)
    conversation_summary = Column(Text, nullable=True, default=None)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tg_id: Mapped[int] = mapped_column(
        ForeignKey("users.tg_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20)) # "user" or "assistant"
    content: Mapped[str]
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="messages")


@lru_cache(maxsize=1)
def get_db_sessionmaker(conn_string: str) -> sessionmaker:
    engine = create_async_engine(conn_string)
    return sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# DB utility functions for Users and Messages

async def upsert_user(session: AsyncSession, tg_id: int, user_data: dict):
    stmt = insert(User).values(tg_id=tg_id, **user_data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["tg_id"],
        set_=user_data
    )
    await session.execute(stmt)
    await session.commit()

async def get_user(session: AsyncSession, tg_id: int) -> Optional[User]:
    stmt = select(User).where(User.tg_id == tg_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

async def get_context_messages(session: AsyncSession, tg_id: int, limit: int = 6) -> List[dict]:
    # Check if user exists first to not return errors
    user = await get_user(session, tg_id)
    if not user:
        return []
        
    stmt = (
        select(Message)
        .where(Message.tg_id == tg_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]

async def add_chat_messages(session: AsyncSession, tg_id: int, user_content: str, assistant_content: str):
    # Ensure user exists; if not, create a minimal profile
    user = await get_user(session, tg_id)
    if not user:
        await upsert_user(session, tg_id, {})
        
    user_msg = Message(tg_id=tg_id, role="user", content=user_content)
    assistant_msg = Message(tg_id=tg_id, role="assistant", content=assistant_content)
    session.add_all([user_msg, assistant_msg])
    
    stmt = (
        update(User)
        .where(User.tg_id == tg_id)
        .values(last_active=func.now())
    )
    await session.execute(stmt)
    await session.commit()

async def delete_user_data(session: AsyncSession, tg_id: int):
    stmt = delete(User).where(User.tg_id == tg_id)
    await session.execute(stmt)
    await session.commit()


async def get_user_memory(session: AsyncSession, tg_id: int) -> dict:
    """Get journey_state and conversation_summary for a user."""
    user = await get_user(session, tg_id)
    if not user:
        return {"journey_state": None, "conversation_summary": None}
    return {
        "journey_state": user.journey_state,
        "conversation_summary": user.conversation_summary,
    }


async def update_user_memory(
    session: AsyncSession, tg_id: int,
    journey_state: dict = None, conversation_summary: str = None
):
    """Update journey_state and/or conversation_summary for a user."""
    values = {"last_active": func.now()}
    if journey_state is not None:
        values["journey_state"] = journey_state
    if conversation_summary is not None:
        values["conversation_summary"] = conversation_summary
    stmt = update(User).where(User.tg_id == tg_id).values(**values)
    await session.execute(stmt)
    await session.commit()
