"""Database connection and session management using SQLModel."""

from pathlib import Path

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

DEFAULT_DB_PATH = Path.home() / ".lmsys-query-analysis" / "queries.db"


class Database:
    """Database connection manager."""

    def __init__(self, db_path: str | Path | None = None, auto_create_tables: bool = True):
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Enable SQLite foreign key enforcement via PRAGMA
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False}
        )

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: D401
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Auto-create tables if enabled
        if auto_create_tables:
            self.create_tables()

    def create_tables(self):
        """Create all tables in the database."""
        SQLModel.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return Session(self.engine)

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        SQLModel.metadata.drop_all(self.engine)


def get_db(db_path: str | Path | None = None, auto_create_tables: bool = True) -> Database:
    """Get database instance."""
    return Database(db_path, auto_create_tables=auto_create_tables)
