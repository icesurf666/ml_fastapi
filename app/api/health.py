from fastapi import APIRouter

from app.core import state

router = APIRouter()


@router.get("/health")
def health():
    dataset_loaded = state.dataset_df is not None and not state.dataset_df.empty
    model_available = state.get_model_record() is not None

    return {
        "status": "ok" if dataset_loaded else "degraded",
        "dataset_loaded": dataset_loaded,
        "model_available": model_available,
    }
