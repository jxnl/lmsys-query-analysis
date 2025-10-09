import { ChromaClient } from 'chromadb';
import { join } from 'path';
import { homedir } from 'os';

// Default path matches Python CLI default: ~/.lmsys-query-analysis/chroma
const DEFAULT_CHROMA_PATH = join(homedir(), '.lmsys-query-analysis', 'chroma');

// Allow override via environment variable
const CHROMA_PATH = process.env.CHROMA_PATH || DEFAULT_CHROMA_PATH;

// Create ChromaDB client
// Note: This connects to a local ChromaDB server, not embedded mode
// You need to run: chroma run --path ~/.lmsys-query-analysis/chroma
export const chroma = new ChromaClient({
  path: 'http://localhost:8000', // Default ChromaDB server port
});

/**
 * Get a ChromaDB collection by provider and model
 * Collection naming convention: {type}_{provider}_{model}
 * e.g., "queries_cohere_embed-v4.0" or "cluster_summaries_cohere_embed-v4.0"
 */
export async function getCollection(
  provider: string,
  model: string,
  type: 'queries' | 'cluster_summaries'
) {
  const collectionName = `${type}_${provider}_${model}`;
  return await chroma.getCollection({ name: collectionName });
}

// Export path for debugging
export const chromaPath = CHROMA_PATH;
