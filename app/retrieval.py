import os
import warnings
from typing import List

from chromadb.api.models import Collection
from sentence_transformers import SentenceTransformer

from data_ingestion.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_COLLECTION_NAME, ZIP_PATH
from data_ingestion.extractor import extract_nested_zip
from data_ingestion.ingestor import KnowledgeBaseBuilder
from utils.chroma_client import get_chroma_client
from utils.logger import setup_logger

# Игнорирование предупреждения torch
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="`encoder_attention_mask` is deprecated",
)

# Инициализация логгера
logger = setup_logger("chunks")

def find_relevant_chunks_by_segment(
    segment: str,
    collection: Collection,
    embedder: SentenceTransformer,
    top_k: int = 5,
) -> List[str]:
    """
    Семантический поиск релевантных чанков по описанию сегмента.

    Args:
        segment: Сегмент (например, "маркетинговое агентство").
        collection: Коллекция ChromaDB.
        embedder: Модель эмбеддингов (SentenceTransformer).
        top_k: Сколько самых похожих чанков вернуть.

    Returns:
        Список релевантных чанков.
    """
    # Проверка входных данных
    if not segment.strip():
        logger.warning("Пустой сегмент для поиска, возвращается пустой список.")
        return []
    if top_k <= 0:
        logger.warning(f"Недопустимое значение top_k ({top_k}), возвращается пустой список.")
        return []

    try:
        # Проверка: коллекция существует, но пуста
        if collection.count() == 0:
            logger.warning("🔄 Коллекция Chroma пуста. Запускаю пересборку базы...")

            # Шаг 1: Распаковка архива
            if os.path.exists(RAW_DATA_DIR):
                print("Начинаю распаковку архива")
                try:
                    extract_nested_zip(ZIP_PATH, PROCESSED_DATA_DIR)
                except Exception as e:
                    print(f"Не удалось распаковать архив {e}")
                logger.info("📦 Архив успешно распакован.")
            else:
                logger.error(f"❌ Архив не найден по пути: {RAW_DATA_DIR}")
                return []

            # Шаг 2: Построение базы знаний
            builder = KnowledgeBaseBuilder()
            builder.ingest()
            logger.info("✅ База знаний успешно создана.")

            # Пересоздаем collection, чтобы она увидела изменения
            collection = get_chroma_client().get_or_create_collection(CHROMA_COLLECTION_NAME)

        # Создание эмбеддинга и поиск
        query_embedding = embedder.encode(segment)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        chunks = results.get("documents", [[]])[0]

        logger.info(f"🔎 Найдено {len(chunks)} чанков по сегменту '{segment}' (семантический поиск).")
        return chunks

    except Exception as e:
        logger.error(f"❌ Ошибка при семантическом поиске: {e}")
        return []
