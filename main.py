import threading
from queue import Queue
from fastapi import FastAPI
import gradio as gr
import uvicorn
from fastapi.responses import RedirectResponse
from model_pipeline import StagingModel
from worker import processing_worker
from api import router as api_router, set_queues as api_set_queues
from ui import create_ui, set_queues as ui_set_queues
from config import API_HOST, API_PORT

if __name__ == "__main__":
    job_queue = Queue()
    results_store = {} 

    model = StagingModel()

    worker_thread = threading.Thread(
        target=processing_worker,
        args=(model, job_queue, results_store),
        daemon=True 
    )
    worker_thread.start()
    
    app = FastAPI(
        title="Virtual Staging API",
        description="An API for virtually staging empty rooms using the FLUX.1 model."
    )
    
    api_set_queues(job_queue, results_store)
    app.include_router(api_router)
    
    ui_set_queues(job_queue, results_store)
    gradio_app = create_ui()
    
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/ui")

    print(f"Server starting. API Docs at http://{API_HOST}:{API_PORT}/docs")
    print(f"Gradio UI at http://{API_HOST}:{API_PORT}/ui")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)