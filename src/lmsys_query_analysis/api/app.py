"""FastAPI application with CORS and error handling."""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse, ErrorDetail

# Create FastAPI app
app = FastAPI(
    title="LMSYS Query Analysis API",
    description="REST API for clustering, search, and analysis of conversational queries",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS to allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:8000",  # FastAPI dev server (for testing)
        # Add production origins here when deployed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ===== Error Handlers =====


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "type": "ValidationError",
                "message": f"Request validation failed: {exc.errors()}",
            }
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError (e.g., invalid run_id)."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "type": "ValueError",
                "message": str(exc),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "InternalServerError",
                "message": f"An unexpected error occurred: {str(exc)}",
            }
        },
    )


# ===== Health Check =====


@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "lmsys-query-analysis"}


# ===== Register Routers =====

# Import routers (will be created next)
from .routers import clustering, analysis, hierarchy, summaries, search, curation

app.include_router(clustering.router, prefix="/api/clustering", tags=["clustering"])
app.include_router(analysis.router, prefix="/api", tags=["analysis"])
app.include_router(hierarchy.router, prefix="/api/hierarchy", tags=["hierarchy"])
app.include_router(summaries.router, prefix="/api/summaries", tags=["summaries"])
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(curation.router, prefix="/api/curation", tags=["curation"])


# ===== Root Endpoint =====


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "LMSYS Query Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


# ===== CLI Entry Point =====


def main():
    """Entry point for lmsys-api command."""
    import uvicorn
    uvicorn.run(
        "lmsys_query_analysis.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
