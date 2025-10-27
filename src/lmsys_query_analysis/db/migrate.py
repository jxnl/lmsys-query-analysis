"""Database migration utilities for schema changes."""

from sqlalchemy import text
from sqlmodel import Session, select

from .connection import Database
from .models import Dataset, Query


def migrate_to_multi_dataset(db: Database, default_dataset_name: str = "legacy-data") -> None:
    """Migrate existing database to support multiple datasets.

    This migration:
    1. Creates the datasets table
    2. Adds dataset_id column to queries table
    3. Creates a default dataset for existing queries
    4. Updates all existing queries to reference the default dataset

    Args:
        db: Database instance
        default_dataset_name: Name for the default dataset containing existing queries
    """
    session = db.get_session()

    try:
        # Check if migration is needed
        result = session.exec(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'")
        )
        if result.first():
            print("✓ Migration already applied (datasets table exists)")
            return

        print("Starting migration to multi-dataset support...")

        # Step 1: Create datasets table
        print("  1. Creating datasets table...")
        Dataset.metadata.create_all(db.engine, tables=[Dataset.__table__])

        # Step 2: Check if queries table has dataset_id column
        result = session.exec(text("PRAGMA table_info(queries)"))
        columns = [row[1] for row in result.all()]

        has_dataset_id = "dataset_id" in columns

        if not has_dataset_id:
            # Step 3: Create default dataset
            print(f"  2. Creating default dataset '{default_dataset_name}'...")

            # Count existing queries
            count_result = session.exec(select(Query)).all()
            query_count = len(count_result)

            default_dataset = Dataset(
                name=default_dataset_name,
                source="unknown",
                description="Legacy data migrated from single-dataset schema",
                query_count=query_count,
            )
            session.add(default_dataset)
            session.commit()
            session.refresh(default_dataset)

            print(f"     Created dataset (ID: {default_dataset.id}) with {query_count} queries")

            # Step 4: Add dataset_id column to queries table (SQLite doesn't support ADD COLUMN with FK directly)
            print("  3. Adding dataset_id column to queries table...")

            # Rename old table
            session.exec(text("ALTER TABLE queries RENAME TO queries_old"))

            # Create new table with updated schema
            Query.metadata.create_all(db.engine, tables=[Query.__table__])

            # Copy data with default dataset_id
            print(f"  4. Migrating {query_count} queries to new schema...")
            session.exec(
                text(f"""
                INSERT INTO queries
                (dataset_id, conversation_id, model, query_text, language, timestamp, extra_metadata, created_at)
                SELECT
                    {default_dataset.id},
                    conversation_id, model, query_text, language, timestamp, extra_metadata, created_at
                FROM queries_old
                """)
            )

            # Drop old table
            session.exec(text("DROP TABLE queries_old"))
            session.commit()

            print(f"✓ Migration complete! All queries assigned to dataset '{default_dataset_name}'")
        else:
            print("  ✓ Queries table already has dataset_id column")

    except Exception as e:
        session.rollback()
        print(f"✗ Migration failed: {e}")
        raise
    finally:
        session.close()


def check_migration_status(db: Database) -> dict:
    """Check if database needs migration to multi-dataset support.

    Returns:
        dict with migration status information
    """
    session = db.get_session()

    try:
        # Check for datasets table
        result = session.exec(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'")
        )
        has_datasets_table = result.first() is not None

        # Check for dataset_id column in queries
        result = session.exec(text("PRAGMA table_info(queries)"))
        columns = [row[1] for row in result.all()]
        has_dataset_id = "dataset_id" in columns

        needs_migration = not (has_datasets_table and has_dataset_id)

        return {
            "needs_migration": needs_migration,
            "has_datasets_table": has_datasets_table,
            "has_dataset_id_column": has_dataset_id,
        }
    finally:
        session.close()
