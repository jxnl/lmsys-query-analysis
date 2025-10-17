#!/usr/bin/env bash

set -e

echo "✨ Formatting Python code with Ruff..."
uv run ruff format src/ tests/

echo ""
echo "✨ Formatting TypeScript code with Prettier..."
cd web && npm run format

echo ""
echo "✅ All code formatted!"

