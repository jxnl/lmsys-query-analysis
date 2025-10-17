#!/usr/bin/env bash
# Check both linting and formatting without making changes

set -e

echo "🔍 Checking Python code..."
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

echo ""
echo "🔍 Checking TypeScript code..."
cd web && npm run lint
cd web && npm run format:check

echo ""
echo "✅ All checks passed!"

