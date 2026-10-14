"""
Database connection setup (SQLite via SQLAlchemy).

SQLite stores the whole database in ONE file (backend/fraud_detection.db),
so no separate database server has to be installed.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

import config

DATABASE_URL = f"sqlite:///{config.DATABASE_PATH}"

# check_same_thread=False lets FastAPI use the connection from different threads
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    """Parent class for all database table models (see models.py)."""


def init_db() -> None:
    """Create the tables if they do not exist yet."""
    import models  # noqa: F401  (import registers the tables on Base)

    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency: opens a DB session for one request and always closes it.
    Used in routes as:  db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
