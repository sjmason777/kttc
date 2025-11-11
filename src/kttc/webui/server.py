# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI WebUI server for KTTC Dashboard.

Provides REST API and web interface for translation QA.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, TypedDict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kttc.agents.orchestrator import AgentOrchestrator
from kttc.core.models import TranslationTask
from kttc.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


# Type definitions
class StatsDict(TypedDict):
    """Server statistics dictionary."""

    total_evaluations: int
    total_mqm_score: float
    start_time: float | None


class AppState(TypedDict):
    """Application state dictionary."""

    orchestrator: AgentOrchestrator | None
    stats: StatsDict
    active_websockets: list[WebSocket]


# Request/Response models
class EvaluateRequest(BaseModel):
    """Request model for translation evaluation."""

    source_text: str = Field(..., description="Source text", min_length=1)
    translation: str = Field(..., description="Translation to evaluate", min_length=1)
    source_lang: str = Field(..., description="Source language code", pattern=r"^[a-z]{2}$")
    target_lang: str = Field(..., description="Target language code", pattern=r"^[a-z]{2}$")
    reference: str | None = Field(default=None, description="Optional reference translation")
    context: dict[str, Any] | None = Field(default=None, description="Optional context")
    domain: str | None = Field(default=None, description="Optional domain (legal, medical, etc.)")


class EvaluateResponse(BaseModel):
    """Response model for translation evaluation."""

    mqm_score: float = Field(description="MQM quality score (0-100)")
    status: str = Field(description="Quality status (pass/fail)")
    errors_count: int = Field(description="Total number of errors")
    errors_by_severity: dict[str, int] = Field(description="Error counts by severity")
    errors: list[dict[str, Any]] = Field(description="Detailed error annotations")
    processing_time: float = Field(description="Evaluation time in seconds")


class BatchEvaluateRequest(BaseModel):
    """Request model for batch evaluation."""

    tasks: list[EvaluateRequest] = Field(..., description="List of translation tasks")


class ServerStats(BaseModel):
    """Server statistics."""

    total_evaluations: int = Field(default=0, description="Total evaluations performed")
    average_mqm_score: float = Field(default=0.0, description="Average MQM score")
    uptime_seconds: float = Field(default=0.0, description="Server uptime")


# Global state
app_state: AppState = {
    "orchestrator": None,
    "stats": {"total_evaluations": 0, "total_mqm_score": 0.0, "start_time": None},
    "active_websockets": [],
}


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="KTTC WebUI",
        description="Web Dashboard for Translation Quality Assurance",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Initialize orchestrator on startup
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize KTTC orchestrator on server startup."""
        logger.info("Initializing KTTC orchestrator...")

        # Get API key from environment
        api_key = os.getenv("KTTC_OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("KTTC_OPENAI_API_KEY not set. WebUI will not work properly.")
            api_key = "dummy-key-for-testing"

        # Initialize LLM provider
        llm = OpenAIProvider(api_key=api_key, model="gpt-4")

        # Create orchestrator
        app_state["orchestrator"] = AgentOrchestrator(llm)
        app_state["stats"]["start_time"] = time.time()

        logger.info("✅ KTTC WebUI server ready")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Cleanup on server shutdown."""
        logger.info("Shutting down KTTC WebUI server...")

        # Close WebSocket connections
        for ws in app_state["active_websockets"]:
            await ws.close()

        app_state["active_websockets"].clear()

    # Routes
    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        """Serve main dashboard page."""
        html_file = Path(__file__).parent / "templates" / "index.html"

        if html_file.exists():
            return html_file.read_text()

        # Fallback simple HTML
        return """
<!DOCTYPE html>
<html>
<head>
    <title>KTTC Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }
        input, textarea, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        textarea {
            min-height: 100px;
            font-family: monospace;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
        .error {
            border-left-color: #f44336;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .status-pass { color: #4CAF50; }
        .status-fail { color: #f44336; }
        .error-list {
            margin-top: 20px;
        }
        .error-item {
            padding: 10px;
            margin: 10px 0;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #ff9800;
        }
        .error-critical { border-left-color: #f44336; }
        .error-major { border-left-color: #ff9800; }
        .error-minor { border-left-color: #ffc107; }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 KTTC Dashboard</h1>
        <p class="subtitle">Translation Quality Assurance Platform</p>

        <form id="evaluateForm">
            <div class="form-group">
                <label>Source Language:</label>
                <select id="sourceLang" required>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="ru">Russian</option>
                    <option value="zh">Chinese</option>
                </select>
            </div>

            <div class="form-group">
                <label>Source Text:</label>
                <textarea id="sourceText" placeholder="Enter source text..." required></textarea>
            </div>

            <div class="form-group">
                <label>Target Language:</label>
                <select id="targetLang" required>
                    <option value="es">Spanish</option>
                    <option value="en">English</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="ru">Russian</option>
                    <option value="zh">Chinese</option>
                </select>
            </div>

            <div class="form-group">
                <label>Translation:</label>
                <textarea id="translation" placeholder="Enter translation to evaluate..." required></textarea>
            </div>

            <button type="submit" id="submitBtn">Evaluate Quality</button>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Evaluating translation...</p>
        </div>

        <div id="result" style="display: none;"></div>
    </div>

    <script>
        const form = document.getElementById('evaluateForm');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const submitBtn = document.getElementById('submitBtn');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Show loading
            loading.style.display = 'block';
            result.style.display = 'none';
            submitBtn.disabled = true;

            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        source_text: document.getElementById('sourceText').value,
                        translation: document.getElementById('translation').value,
                        source_lang: document.getElementById('sourceLang').value,
                        target_lang: document.getElementById('targetLang').value,
                    }),
                });

                const data = await response.json();

                // Display results
                let html = `
                    <div class="result">
                        <h2>Quality Assessment</h2>
                        <div class="metric">
                            <div class="metric-label">MQM Score</div>
                            <div class="metric-value status-${data.status}">${data.mqm_score.toFixed(2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Status</div>
                            <div class="metric-value status-${data.status}">${data.status.toUpperCase()}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Total Errors</div>
                            <div class="metric-value">${data.errors_count}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Processing Time</div>
                            <div class="metric-value">${data.processing_time.toFixed(2)}s</div>
                        </div>
                `;

                if (data.errors && data.errors.length > 0) {
                    html += '<div class="error-list"><h3>Detected Errors:</h3>';
                    data.errors.forEach(error => {
                        html += `
                            <div class="error-item error-${error.severity}">
                                <strong>${error.category}</strong> / ${error.subcategory}
                                <span style="float:right; text-transform:uppercase; font-weight:bold;">${error.severity}</span>
                                <p>${error.description}</p>
                                ${error.suggestion ? '<p><em>Suggestion: ' + error.suggestion + '</em></p>' : ''}
                            </div>
                        `;
                    });
                    html += '</div>';
                }

                html += '</div>';
                result.innerHTML = html;
                result.style.display = 'block';

            } catch (error) {
                result.innerHTML = `
                    <div class="result error">
                        <h2>Error</h2>
                        <p>${error.message}</p>
                    </div>
                `;
                result.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

    @app.post("/api/evaluate", response_model=EvaluateResponse)
    async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
        """Evaluate single translation."""
        orchestrator = app_state["orchestrator"]

        if orchestrator is None:
            raise HTTPException(status_code=503, detail="Orchestrator not initialized")

        # Create translation task (note: reference not supported in TranslationTask)
        task = TranslationTask(
            source_text=request.source_text,
            translation=request.translation,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            context=request.context,
        )

        # Evaluate
        start_time = time.time()
        report = await orchestrator.evaluate(task)
        processing_time = time.time() - start_time

        # Update stats
        app_state["stats"]["total_evaluations"] += 1
        app_state["stats"]["total_mqm_score"] += report.mqm_score

        # Prepare response
        errors_by_severity = {"critical": 0, "major": 0, "minor": 0}
        for error in report.errors:
            errors_by_severity[error.severity.value] += 1

        return EvaluateResponse(
            mqm_score=report.mqm_score,
            status="pass" if report.mqm_score >= 85 else "fail",
            errors_count=len(report.errors),
            errors_by_severity=errors_by_severity,
            errors=[
                {
                    "category": e.category,
                    "subcategory": e.subcategory,
                    "severity": e.severity.value,
                    "description": e.description,
                    "suggestion": e.suggestion,
                    "location": e.location,
                }
                for e in report.errors
            ],
            processing_time=processing_time,
        )

    @app.post("/api/batch-evaluate")
    async def batch_evaluate(request: BatchEvaluateRequest) -> JSONResponse:
        """Evaluate multiple translations in batch."""
        results = []

        for task_request in request.tasks:
            try:
                result = await evaluate(task_request)
                results.append({"status": "success", "result": result.model_dump()})
            except Exception as e:
                results.append({"status": "error", "error": str(e)})

        return JSONResponse(content={"results": results, "total": len(results)})

    @app.get("/api/stats", response_model=ServerStats)
    async def get_stats() -> ServerStats:
        """Get server statistics."""
        stats = app_state["stats"]
        uptime = time.time() - stats["start_time"] if stats["start_time"] else 0

        avg_score = 0.0
        if stats["total_evaluations"] > 0:
            avg_score = stats["total_mqm_score"] / stats["total_evaluations"]

        return ServerStats(
            total_evaluations=stats["total_evaluations"],
            average_mqm_score=avg_score,
            uptime_seconds=uptime,
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        app_state["active_websockets"].append(websocket)

        try:
            while True:
                # Wait for messages
                data = await websocket.receive_json()

                # Echo back for now
                await websocket.send_json({"message": "received", "data": data})
        except WebSocketDisconnect:
            app_state["active_websockets"].remove(websocket)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run WebUI server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    logger.info(f"Starting KTTC WebUI on http://{host}:{port}")

    uvicorn.run(
        "kttc.webui.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    run_server(reload=True)
