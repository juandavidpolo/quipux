from typing import Union
from fastapi import FastAPI

from app.views import predict

app = FastAPI()

app.include_router(predict.router)