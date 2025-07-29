# Настройка клиента
from openai import AsyncOpenAI

from data_ingestion.config import OPENAI_API_KEY

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)