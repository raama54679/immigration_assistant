import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from graph_builder import build_graph, ResearchState
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Immigration Assistant API", version="0.1.0")


class IntakeRequest(BaseModel):
    user_location: str
    desired_outcome: str
    credentials: List[str]
    radius_km: Optional[int] = 50
    max_iterations: Optional[int] = 5


class RunResponse(BaseModel):
    success: bool
    iterations: int
    nearest_option: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]]
    missing_requirements: List[str]
    trace: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer", response_model=RunResponse)
def infer(req: IntakeRequest):
    if not os.environ.get("TAVILY_API_KEY"):
        raise HTTPException(status_code=500, detail="TAVILY_API_KEY not set")

    graph = build_graph(max_iterations=req.max_iterations or 5)

    # Initial state
    init_state: ResearchState = {
        "user_location": req.user_location,
        "desired_outcome": req.desired_outcome,
        "credentials": req.credentials,
        "radius_km": req.radius_km or 50,
        "iteration": 0,
        "candidates": [],
        "nearest_option": None,
        "missing_requirements": [],
        "trace": []
    }

    try:
        final_state = graph.invoke(init_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RunResponse(
        success=final_state.get("nearest_option") is not None,
        iterations=final_state.get("iteration", 0),
        nearest_option=final_state.get("nearest_option"),
        candidates=final_state.get("candidates", []),
        missing_requirements=final_state.get("missing_requirements", []),
        trace=final_state.get("trace", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
