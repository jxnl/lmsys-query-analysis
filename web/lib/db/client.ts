import { drizzle } from 'drizzle-orm/better-sqlite3';
import Database from 'better-sqlite3';
import * as schema from './schema';
import { join } from 'path';
import { homedir } from 'os';

// Default path matches Python CLI default: ~/.lmsys-query-analysis/queries.db
const DEFAULT_DB_PATH = join(homedir(), '.lmsys-query-analysis', 'queries.db');

// Allow override via environment variable
const DB_PATH = process.env.DB_PATH || DEFAULT_DB_PATH;

// Create SQLite connection (read-only for safety)
const sqlite = new Database(DB_PATH, { readonly: true });

// Create Drizzle instance with schema
export const db = drizzle(sqlite, { schema });

// Export DB path for debugging
export const dbPath = DB_PATH;
