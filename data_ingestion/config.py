import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Абсолютный путь к текущему файлу
CURRENT_FILE = Path(__file__).resolve()

# Поднимаемся на один уровень вверх от config.py → PROJECT_ROOT
PROJECT_ROOT = CURRENT_FILE.parents[1]

# Директории с данными
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Исходные файлы
ZIP_PATH = RAW_DATA_DIR / "Konsol_Pro_Articles.zip"
PDF_PATH = RAW_DATA_DIR / "Service_Console.pdf"

# Для Chroma
CHROMA_DB_PATH = PROJECT_ROOT / "vector_store"
CHROMA_COLLECTION_NAME = "sales_knowledge_base"
CHUNK_SIZE = 320
CHUNK_OVERLAP = 50

# Модель эмбеддингов
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#API KEY OPEN AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
