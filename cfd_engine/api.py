import sys
from pathlib import Path
from typing import Any, Dict
import uuid
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from main import run_simulation  # noqa: E402

app = FastAPI(
    title="SciML CFD Engine API",
    description="API for running 3D PINN pipe flow simulations and returning visual results.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

OUTPUT_ROOT = root_dir / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

jobs: Dict[str, Dict[str, Any]] = {}


class SimulationRequest(BaseModel):
    reynolds_number: float = Field(100.0, gt=0)
    inlet_velocity: float = Field(1.0, gt=0)
    radius: float = Field(0.5, gt=0)
    length: float = Field(3.0, gt=0)
    epochs: int = Field(1000, gt=0)
    batch_interior: int = Field(1000, gt=1)
    batch_boundary: int = Field(200, gt=0)
    lbfgs_epochs: int = Field(100, ge=0)


class SimulationStatus(BaseModel):
    task_id: str
    status: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    warning: str | None = None
    image_path: str | None = None
    model_path: str | None = None


def _run_background_simulation(task_id: str, request: SimulationRequest) -> None:
    job = jobs.get(task_id)
    if job is None:
        return

    job["status"] = "running"
    job["started_at"] = time.time()

    try:
        result = run_simulation(
            reynolds_number=request.reynolds_number,
            inlet_velocity=request.inlet_velocity,
            radius=request.radius,
            length=request.length,
            epochs=request.epochs,
            output_dir=str(job["output_dir"]),
            batch_interior=request.batch_interior,
            batch_boundary=request.batch_boundary,
            lbfgs_epochs=request.lbfgs_epochs,
            run_id=task_id,
        )

        job["status"] = "completed"
        job["finished_at"] = time.time()
        job["image_path"] = result["image_path"]
        job["model_path"] = result["model_state_path"]
    except Exception as exc:
        job["status"] = "failed"
        job["finished_at"] = time.time()
        job["error"] = str(exc)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "SciML CFD API"}


@app.post("/simulate")
def create_simulation(request: SimulationRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    task_id = uuid.uuid4().hex
    output_dir = OUTPUT_ROOT / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    warning = None
    if request.reynolds_number > 2000:
        warning = (
            "Reynolds number exceeds 2000. "
            "This backend is optimized for laminar or transitional training, and high Re may be unstable."
        )

    jobs[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "warning": warning,
        "output_dir": output_dir,
        "image_path": None,
        "model_path": None,
    }

    background_tasks.add_task(_run_background_simulation, task_id, request)
    return {"task_id": task_id, "status": "pending", "warning": warning}


@app.get("/status/{task_id}", response_model=SimulationStatus)
def get_status(task_id: str) -> SimulationStatus:
    job = jobs.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return SimulationStatus(**job)


@app.get("/result/{task_id}")
def get_result(task_id: str):
    job = jobs.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Result not ready")

    image_path = job.get("image_path")
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Result image not found")

    return FileResponse(image_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
