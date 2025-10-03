"""Database connection and session management using SQLModel."""
from pathlib import Path
from sqlmodel import create_engine, SQLModel, Session

DEFAULT_DB_PATH = Path.home() / ".lmsys-query-analysis" / "queries.db"


class Database:
    """Database connection manager."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def create_tables(self):
        """Create all tables in the database."""
        SQLModel.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return Session(self.engine)

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        SQLModel.metadata.drop_all(self.engine)


def get_db(db_path: str | Path | None = None) -> Database:
    """Get database instance."""
    return Database(db_path)
