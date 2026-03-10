import uuid
from datetime import datetime
from functools import lru_cache
from typing import List, Optional

from sqlalchemy import BigInteger, Column, Float, ForeignKey, Integer, String, DateTime, Text, func, select, delete, update
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


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tg_id: Mapped[int] = mapped_column(
        ForeignKey("users.tg_id", ondelete="CASCADE"), index=True
    )
    message_id = Column(BigInteger, nullable=False)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    rating: Mapped[int]  # 1 = positive, -1 = negative
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PipelineLog(Base):
    __tablename__ = "pipeline_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tg_id: Mapped[int] = mapped_column(BigInteger, index=True)
    question: Mapped[str] = mapped_column(Text)
    rewritten_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(default=False)
    docs_retrieved: Mapped[int] = mapped_column(Integer, default=0)
    docs_after_grading: Mapped[int] = mapped_column(Integer, default=0)
    t_rewrite: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_retrieve: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_grade: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_generate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    t_total: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


@lru_cache(maxsize=1)
def get_db_sessionmaker(conn_string: str) -> sessionmaker:
    engine = create_async_engine(
        conn_string, pool_size=10, max_overflow=20, pool_recycle=1800
    )
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


async def add_feedback(
    session: AsyncSession, tg_id: int, message_id: int,
    question: str, answer_text: str, rating: int,
):
    """Store user feedback (👍/👎) for a bot response."""
    fb = Feedback(
        tg_id=tg_id, message_id=message_id,
        question=question, answer=answer_text, rating=rating,
    )
    session.add(fb)
    await session.commit()


async def get_feedback_stats(session: AsyncSession) -> dict:
    """Get aggregate feedback statistics."""
    from sqlalchemy import func as sa_func
    pos = await session.scalar(
        select(sa_func.count()).select_from(Feedback).where(Feedback.rating == 1)
    )
    neg = await session.scalar(
        select(sa_func.count()).select_from(Feedback).where(Feedback.rating == -1)
    )
    return {"positive": pos or 0, "negative": neg or 0, "total": (pos or 0) + (neg or 0)}


async def add_pipeline_log(
    session_factory,
    tg_id: int,
    question: str,
    rewritten_question: str | None,
    cache_hit: bool,
    docs_retrieved: int,
    docs_after_grading: int,
    timings: dict,
):
    """Persist pipeline execution metrics to DB.

    Opens its own session so it can safely run as a fire-and-forget task
    without conflicting with the caller's session lifecycle.
    """
    async with session_factory() as session:
        log = PipelineLog(
            tg_id=tg_id,
            question=question[:1000],
            rewritten_question=(rewritten_question or "")[:1000],
            cache_hit=cache_hit,
            docs_retrieved=docs_retrieved,
            docs_after_grading=docs_after_grading,
            t_rewrite=timings.get("rewrite"),
            t_retrieve=timings.get("retrieve"),
            t_grade=timings.get("grade"),
            t_generate=timings.get("generate"),
            t_total=timings.get("total", 0),
        )
        session.add(log)
        await session.commit()


async def get_analytics_summary(session: AsyncSession, days: int = 7) -> dict:
    """Get comprehensive analytics for the last N days."""
    from sqlalchemy import func as f
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=days)

    # -- Users --
    total_users = await session.scalar(
        select(f.count()).select_from(User)
    ) or 0
    active_users = await session.scalar(
        select(f.count()).select_from(User).where(User.last_active >= cutoff)
    ) or 0

    # -- Pipeline logs --
    total_queries = await session.scalar(
        select(f.count()).select_from(PipelineLog).where(PipelineLog.created_at >= cutoff)
    ) or 0
    cache_hits = await session.scalar(
        select(f.count()).select_from(PipelineLog).where(
            PipelineLog.created_at >= cutoff, PipelineLog.cache_hit == True  # noqa: E712
        )
    ) or 0
    avg_latency = await session.scalar(
        select(f.avg(PipelineLog.t_total)).where(PipelineLog.created_at >= cutoff)
    )
    avg_docs = await session.scalar(
        select(f.avg(PipelineLog.docs_after_grading)).where(
            PipelineLog.created_at >= cutoff, PipelineLog.cache_hit == False  # noqa: E712
        )
    )
    p95_latency = await session.scalar(
        select(f.percentile_cont(0.95).within_group(PipelineLog.t_total)).where(
            PipelineLog.created_at >= cutoff
        )
    )

    # -- Feedback --
    fb_stats = await get_feedback_stats(session)

    # -- Messages --
    total_messages = await session.scalar(
        select(f.count()).select_from(Message).where(Message.created_at >= cutoff)
    ) or 0

    return {
        "period_days": days,
        "users": {"total": total_users, "active_last_period": active_users},
        "queries": {
            "total": total_queries,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_queries, 3) if total_queries else 0,
            "avg_latency_s": round(avg_latency or 0, 2),
            "p95_latency_s": round(p95_latency or 0, 2),
            "avg_docs_per_query": round(avg_docs or 0, 1),
        },
        "feedback": fb_stats,
        "messages": {"total_last_period": total_messages},
    }
