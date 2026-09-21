from fastapi import FastAPI

from error_handlers import register_exception_handlers
from logging_config import logger
from routers import dataset, health, model, predict

app = FastAPI()
register_exception_handlers(app)

app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(model.router)
app.include_router(health.router)

logger.info("Churn service started")


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}
