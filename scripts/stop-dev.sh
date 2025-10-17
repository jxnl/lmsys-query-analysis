#!/usr/bin/env bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping development services...${NC}"

if [ -f "logs/fastapi.pid" ]; then
    FASTAPI_PID=$(cat logs/fastapi.pid)
    if ps -p $FASTAPI_PID > /dev/null 2>&1; then
        echo -e "Stopping FastAPI (PID: $FASTAPI_PID)..."
        kill $FASTAPI_PID
        rm logs/fastapi.pid
        echo -e "${GREEN}FastAPI stopped${NC}"
    else
        echo -e "${YELLOW}FastAPI not running${NC}"
        rm logs/fastapi.pid
    fi
else
    echo -e "${YELLOW}No FastAPI PID file found${NC}"
fi

if [ -f "logs/nextjs.pid" ]; then
    NEXTJS_PID=$(cat logs/nextjs.pid)
    if ps -p $NEXTJS_PID > /dev/null 2>&1; then
        echo -e "Stopping Next.js (PID: $NEXTJS_PID)..."
        kill $NEXTJS_PID
        rm logs/nextjs.pid
        echo -e "${GREEN}Next.js stopped${NC}"
    else
        echo -e "${YELLOW}Next.js not running${NC}"
        rm logs/nextjs.pid
    fi
else
    echo -e "${YELLOW}No Next.js PID file found${NC}"
fi

echo -e "${YELLOW}Cleaning up any remaining processes on ports 8000 and 3000...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo -e "${GREEN}All services stopped${NC}"
