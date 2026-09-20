from fastapi import FastAPI

from error_handlers import register_exception_handlers
from routers import dataset, model, predict

app = FastAPI()
register_exception_handlers(app)

app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(model.router)


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}
