#!/usr/bin/env bash
# Tail logs for both services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Tailing logs for FastAPI and Next.js...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop${NC}"
echo ""

# Tail both log files
tail -f logs/fastapi.log logs/nextjs.log
