#!/usr/bin/env bash
# Start both FastAPI backend and Next.js frontend for development

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LMSYS Query Analysis - Development Mode${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .env.local exists in web directory
if [ ! -f "web/.env.local" ]; then
    echo -e "${YELLOW}Creating web/.env.local from example...${NC}"
    cp web/.env.local.example web/.env.local
fi

# Create log directory
mkdir -p logs

# Start FastAPI backend in background
echo -e "${GREEN}Starting FastAPI backend on http://localhost:8000${NC}"
uv run uvicorn lmsys_query_analysis.api.app:app --reload --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Wait for FastAPI to be ready
echo -e "${YELLOW}Waiting for FastAPI to start...${NC}"
sleep 3

# Check if FastAPI is running
if ! curl -s http://localhost:8000/api/health > /dev/null; then
    echo -e "${YELLOW}Warning: FastAPI may not be ready yet. Check logs/fastapi.log${NC}"
fi

# Start Next.js frontend in background
echo -e "${GREEN}Starting Next.js frontend on http://localhost:3000${NC}"
cd web
npm run dev > ../logs/nextjs.log 2>&1 &
NEXTJS_PID=$!
cd ..
echo "Next.js PID: $NEXTJS_PID"

# Save PIDs to file for cleanup
echo "$FASTAPI_PID" > logs/fastapi.pid
echo "$NEXTJS_PID" > logs/nextjs.pid

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Both services started successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "FastAPI:  ${BLUE}http://localhost:8000${NC} (Docs: ${BLUE}http://localhost:8000/docs${NC})"
echo -e "Next.js:  ${BLUE}http://localhost:3000${NC}"
echo ""
echo -e "Logs:"
echo -e "  FastAPI: ${YELLOW}logs/fastapi.log${NC}"
echo -e "  Next.js: ${YELLOW}logs/nextjs.log${NC}"
echo ""
echo -e "To stop both services:"
echo -e "  ${YELLOW}./scripts/stop-dev.sh${NC}"
echo ""
echo -e "To view logs:"
echo -e "  ${YELLOW}tail -f logs/fastapi.log${NC}"
echo -e "  ${YELLOW}tail -f logs/nextjs.log${NC}"
echo ""
