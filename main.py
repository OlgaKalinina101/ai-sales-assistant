from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from utils.logger import setup_logger

# Инициализация логгера ДО импорта роутера
logger = setup_logger("letter_pipeline")

from app.routes import router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
