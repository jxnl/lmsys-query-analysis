#!/usr/bin/env bash
# Run linting for both Python and TypeScript

set -e

echo "🔍 Linting Python code with Ruff..."
uv run ruff check src/ tests/

echo ""
echo "🔍 Linting TypeScript code with ESLint..."
cd web && npm run lint

echo ""
echo "✅ All linting checks passed!"

