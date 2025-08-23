# graph_builder.py
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from credential_rules import check_sufficiency, normalize_desired_outcome

# ---- State ----
class ResearchState(TypedDict, total=False):
    user_location: str
    desired_outcome: str
    credentials: List[str]
    radius_km: int
    iteration: int
    candidates: List[Dict[str, Any]]
    nearest_option: Optional[Dict[str, Any]]
    missing_requirements: List[str]
    adequate: bool
    trace: List[str]
    _max_iterations: int  # injected by wrapper

# ---- Helpers ----
def _rank_key(x: Dict[str, Any]) -> Tuple[int, float]:
    title = (x.get("title") or "").lower()
    bonus = 2 if any(k in title for k in ["uscis", ".gov", "nonprofit", "clinic", "legal aid"]) else 0
    return (bonus, float(x.get("score") or 0.0))

def _search_places(state: ResearchState) -> List[Dict[str, Any]]:
    """
    Fast, resilient Tavily search:
    - basic depth (faster)
    - small result count
    - per-query try/except so one failure doesn't nuke everything
    - early stop once we’ve got enough candidates
    """
    client = TavilyClient()
    loc = state["user_location"]
    outcome = normalize_desired_outcome(state["desired_outcome"])

    queries = [
        f"immigration legal aid near {loc} {outcome}",
        f"accredited immigration services near {loc} {outcome}",
        f"nonprofit immigration clinic near {loc} {outcome}",
        f"uscis community outreach near {loc} {outcome}",
        f"immigration lawyer near {loc} {outcome} free consultation",
    ]

    found: List[Dict[str, Any]] = []
    seen = set()
    trace = state.get("trace", [])

    for q in queries:
        try:
            resp = client.search(q, search_depth="basic", max_results=3)
            for r in (resp.get("results") or []):
                url = r.get("url")
                if url and url not in seen:
                    seen.add(url)
                    found.append({
                        "title": r.get("title"),
                        "url": url,
                        "content": r.get("content"),
                        "score": r.get("score"),
                        "query": q,
                    })
            trace.append(f"Tavily OK: '{q}' -> {len(resp.get('results') or [])} results")
        except Exception as e:
            trace.append(f"Tavily ERROR on '{q}': {e}")

        if len(found) >= 8:  # early stop
            break

    state["trace"] = trace
    return sorted(found, key=_rank_key, reverse=True)

# ---- Nodes ----
def intake_node(state: ResearchState) -> ResearchState:
    trace = state.get("trace", [])
    trace.append(
        f"Intake: location={state.get('user_location')}, "
        f"desired_outcome={state.get('desired_outcome')}, "
        f"creds={state.get('credentials')}"
    )
    state["trace"] = trace
    state["iteration"] = int(state.get("iteration", 0))
    state["candidates"] = state.get("candidates", [])
    state["missing_requirements"] = state.get("missing_requirements", [])
    state["nearest_option"] = state.get("nearest_option")
    state["adequate"] = False
    return state

def search_node(state: ResearchState) -> ResearchState:
    trace = state.get("trace", [])
    trace.append(f"Search: iteration={state.get('iteration', 0)}")
    state["trace"] = trace

    new_results = _search_places(state)

    # merge + dedupe by URL, keep top 10
    existing = state.get("candidates", [])
    seen = {c.get("url") for c in existing if c.get("url")}
    for r in new_results:
        if r.get("url") not in seen:
            existing.append(r)
    state["candidates"] = sorted(existing, key=_rank_key, reverse=True)[:10]
    return state

def evaluate_node(state: ResearchState) -> ResearchState:
    """
    Check if we have enough to stop:
      - credentials sufficient
      - at least 1 candidate
    """
    desired_norm = normalize_desired_outcome(state["desired_outcome"])
    creds = [c.lower().strip() for c in state.get("credentials", [])]
    sufficient, missing = check_sufficiency(desired_norm, creds)
    state["missing_requirements"] = missing

    candidates = state.get("candidates", [])
    adequate = bool(sufficient and candidates)
    state["adequate"] = adequate

    if adequate:
        state["nearest_option"] = candidates[0]  # proxy for nearest; replace with geocoding later

    trace = state.get("trace", [])
    trace.append(
        "Evaluate: adequate (stop)." if adequate
        else f"Evaluate: not adequate (missing={missing}, candidates={len(candidates)})."
    )
    state["trace"] = trace
    return state

def decide_node(state: ResearchState) -> ResearchState:
    """
    If adequate → stop (nearest_option set).
    Else increment iteration; if < _max_iterations → continue; otherwise stop.
    """
    if state.get("adequate") and state.get("nearest_option"):
        state["trace"] = state.get("trace", []) + ["Decide: adequate → END."]
        return state

    it = int(state.get("iteration", 0)) + 1
    state["iteration"] = it
    max_iter = int(state.get("_max_iterations", 5))

    trace = state.get("trace", [])
    trace.append(f"Decide: inadequate, iteration={it}/{max_iter}.")
    if it >= max_iter:
        trace.append("Decide: reached max iterations → END.")
    else:
        trace.append("Decide: continue looping → Search.")
    state["trace"] = trace
    return state

# --- Build the LangGraph workflow (with wrapper) ---
def build_graph(max_iterations: int = 5):
    g = StateGraph(ResearchState)

    # Nodes
    g.add_node("intake", intake_node)
    g.add_node("search", search_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("decide", decide_node)

    # Edges: intake -> search -> evaluate -> decide -> (conditional)
    g.set_entry_point("intake")
    g.add_edge("intake", "search")
    g.add_edge("search", "evaluate")
    g.add_edge("evaluate", "decide")

    # Router: from decide either END or back to search
    def _router(state: ResearchState):
        if state.get("adequate") and state.get("nearest_option"):
            return END
        # if we've hit the cap, end; else loop
        if int(state.get("iteration", 0)) >= int(state.get("_max_iterations", 5)):
            return END
        return "search"

    g.add_conditional_edges("decide", _router, {"search": "search", END: END})

    app = g.compile()

    class _Wrapper:
        """Injects `_max_iterations` into state before invocation."""
        def __init__(self, app, max_iterations: int):
            self._app = app
            self._max_iterations = max_iterations

        def invoke(self, state: ResearchState):
            state["_max_iterations"] = self._max_iterations
            return self._app.invoke(state)

    return _Wrapper(app, max_iterations)
