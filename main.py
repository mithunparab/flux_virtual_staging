import threading
from queue import Queue
from pathlib import Path
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
import gradio as gr

from model_pipeline import StagingModel
from worker import processing_worker
from api import router as api_router, set_queues as api_set_queues
from ui import create_ui, set_queues as ui_set_queues

os.environ["HOME"] = "/app"
Path.home = lambda: Path("/app")

job_queue = Queue()
results_store = {}
model = None
worker_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the application.
    """
    global model, worker_thread
    print("ASGI App is starting up...")

    try:
        print("Initializing StagingModel...")
        model = StagingModel()
        print("StagingModel initialized successfully.")
    except Exception as e:
        print(f"FATAL: Failed to initialize StagingModel during startup. Error: {e}")
        raise e

    worker_thread = threading.Thread(
        target=processing_worker,
        args=(model, job_queue, results_store),
        daemon=True
    )
    worker_thread.start()
    print("Processing worker thread started.")
    
    yield
    
    print("ASGI App is shutting down...")

app = FastAPI(
    title="Virtual Staging API",
    description="An API for virtually staging empty rooms using the FLUX.1 model.",
    lifespan=lifespan
)

api_set_queues(job_queue, results_store)
ui_set_queues(job_queue, results_store)

app.include_router(api_router, prefix="/api")

gradio_app = create_ui()

app = gr.mount_gradio_app(app, gradio_app, path="/ui")

@app.get("/health", status_code=200, include_in_schema=False)
async def health_check():
    return {"status": "ok", "model_ready": model is not None}

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Welcome to the Virtual Staging Service",
        "ui": "/ui",
        "api_docs": "/docs"
    }