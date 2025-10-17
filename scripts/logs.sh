#!/usr/bin/env bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Tailing logs for FastAPI and Next.js...${NC}"
echo -e "${GREEN}Press Ctrl+C to stop${NC}"
echo ""

tail -f logs/fastapi.log logs/nextjs.log
