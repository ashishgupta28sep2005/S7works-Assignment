"""Safely create missing forecast tables without altering existing data."""

from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError

from app.database.connection import (
    DatabaseConfigurationError,
    create_database_engine,
)
from app.database.models import Base

logger = logging.getLogger(__name__)


def initialize_database(engine=None) -> None:
    """Create tables that do not yet exist; this never drops or replaces objects."""
    active_engine = engine or create_database_engine()
    try:
        Base.metadata.create_all(bind=active_engine, checkfirst=True)
    finally:
        if engine is None:
            active_engine.dispose()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        engine = create_database_engine()
    except DatabaseConfigurationError as exc:
        print(f"Database configuration error: {exc}")
        return 2
    try:
        initialize_database(engine)
    except SQLAlchemyError as exc:
        logger.error("Database initialization failed (%s); connection details were withheld.", type(exc).__name__)
        print(f"Database initialization failed ({type(exc).__name__}); connection details were withheld.")
        return 1
    finally:
        engine.dispose()
    print("Database tables created or verified; no existing tables were dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
