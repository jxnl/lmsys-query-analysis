"""Database connection and session management using SQLModel."""

from pathlib import Path
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import event, text
import logging

logger = logging.getLogger(__name__)

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
        # Run any pending migrations
        self._run_migrations()

    def get_session(self) -> Session:
        """Get a new database session."""
        return Session(self.engine)

    def drop_tables(self):
        """Drop all tables (use with caution)."""
        SQLModel.metadata.drop_all(self.engine)

    def _run_migrations(self):
        """Run pending database migrations."""
        with self.get_session() as session:
            # Migration 1: Add hierarchy_runs table (if not exists)
            try:
                    # Check if hierarchy_runs table exists
                    result = session.exec(text("SELECT name FROM sqlite_master WHERE type='table' AND name='hierarchy_runs'")).fetchall()
                    if not result:
                        logger.info("Creating hierarchy_runs table...")
                        # The table will be created by SQLModel.metadata.create_all() above
                        # This is just a placeholder for future migrations
                        pass
                    else:
                        logger.debug("hierarchy_runs table already exists")
                    
                    # Check if summary_runs table exists
                    result = session.exec(text("SELECT name FROM sqlite_master WHERE type='table' AND name='summary_runs'")).fetchall()
                    if not result:
                        logger.info("Creating summary_runs table...")
                        # The table will be created by SQLModel.metadata.create_all() above
                        # This is just a placeholder for future migrations
                        pass
                    else:
                        logger.debug("summary_runs table already exists")
            except Exception as e:
                logger.warning(f"Migration check failed: {e}")
                # Continue anyway - SQLModel will handle table creation


def get_db(db_path: str | Path | None = None, auto_create_tables: bool = True) -> Database:
    """Get database instance."""
    return Database(db_path, auto_create_tables=auto_create_tables)
