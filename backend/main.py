"""
backend/main.py
Main entry point for FedGraph-Diabetes FastAPI application.
"""

import sys
import os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from config.config import BACKEND_HOST, BACKEND_PORT, PLOTS_DIR, RESULTS_DIR
from backend.api.routes import router

app = FastAPI(
    title="FedGraph-Diabetes API",
    description="Privacy-Aware Explainable Federated Graph Learning for Diabetes Risk Prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS setup
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for plots
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

# Include API routes
app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Welcome to FedGraph-Diabetes API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
