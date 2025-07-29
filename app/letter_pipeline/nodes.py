
import os
import time
from typing import Dict

import psutil
from sentence_transformers import SentenceTransformer

from app.helpers import extract_json
from app.letter_pipeline.openai_client import client
from app.letter_pipeline.types import LetterState
from app.retrieval import find_relevant_chunks_by_segment
from data_ingestion.config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from utils.chroma_client import get_chroma_client
from utils.logger import setup_logger

# Инициализация логгера
logger = setup_logger("letter_pipeline")

# Глобальная инициализация клиента ChromaDB и модели эмбеддингов
chroma_client = get_chroma_client()
chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
openai_client = client
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt_template.txt")

# Определение узлов конвейера
async def input_node(state: LetterState) -> LetterState:
    """
    Принимает начальное состояние и возвращает его без изменений.

    Args:
        state: Состояние конвейера с пользовательскими данными.

    Returns:
        То же состояние без изменений.
    """
    # Возврат входного состояния
    return state


async def search_chunks_node(state: LetterState) -> LetterState:
    """
    Выполняет семантический поиск релевантных чанков по сегменту.

    Args:
        state: Состояние конвейера с пользовательскими данными.

    Returns:
        Обновленное состояние с добавленным списком чанков.
    """
    # Проверка наличия и корректности сегмента
    if not isinstance(state.get("user_input"), Dict) or not state["user_input"].get("сегмент"):
        logger.error("Отсутствует или некорректен ключ 'сегмент' в user_input.")
        return {**state, "chunks": []}

    # Извлечение сегмента и поиск чанков
    segment = state["user_input"]["сегмент"]
    chunks = find_relevant_chunks_by_segment(segment, chroma_collection, embedder)

    # Логирование потребления памяти
    logger.info(
        f"Потребление памяти после поиска чанков: "
        f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
    )

    # Обновление состояния с найденными чанками
    return {**state, "chunks": chunks}

def load_prompt_template() -> str:
    """Загружает контекстный промпт по указанному пути"""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

async def build_prompt_node(state: LetterState) -> LetterState:
    """
    Формирует промпт для генерации письма на основе пользовательских данных и чанков.

    Args:
        state: Состояние конвейера с пользовательскими данными и чанками.

    Returns:
        Обновленное состояние с добавленным промптом.
    """
    # Проверка наличия необходимых данных
    if not isinstance(state.get("user_input"), Dict) or not state.get("chunks"):
        logger.error("Отсутствуют необходимые данные: user_input или chunks.")
        return {**state, "prompt": ""}

    user_input = state["user_input"]
    chunks = state["chunks"]

    required_keys = ["контакт", "должность", "название_компании", "сегмент"]
    missing_keys = [key for key in required_keys if key not in user_input]
    if missing_keys:
        logger.error(f"Отсутствуют ключи в user_input: {missing_keys}")
        return {**state, "prompt": ""}

    # Формирование контекста из чанков (ограничение до 5)
    context = "\n\n".join(chunks[:5])

    # Создание промпта для письма
    print("template keys:", user_input.keys())
    template = load_prompt_template()

    try:
        prompt = template.format(**user_input, context=context)
    except KeyError as e:
        logger.error(f"Ошибка форматирования шаблона: отсутствует ключ {e}")
        return {**state, "prompt": ""}

    # Обновление состояния с промптом
    return {**state, "prompt": prompt}


async def generate_letter_node(state: LetterState) -> LetterState:
    """
    Генерирует деловое письмо с помощью OpenAI API на основе промпта.

    Args:
        state: Состояние конвейера с промптом.

    Returns:
        Обновленное состояние с сгенерированным письмом.
    """
    # Проверка наличия промпта
    if not state.get("prompt"):
        logger.error("Отсутствует промпт для генерации письма.")
        return {**state, "letter": ""}

    # Генерация письма через асинхронный OpenAI API
    try:
        start_time = time.perf_counter()

        logger.info("Отправляем запрос в OpenAI API")
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Ты — AI-помощник, генерирующий персонализированные письма для клиентов.",
                },
                {"role": "user", "content": state["prompt"]},
            ],
            temperature=0.7,
        )
        letter_raw = response.choices[0].message.content
        elapsed = time.perf_counter() - start_time

        # Логгирование времени генерации
        logger.info(f"📨 Письмо успешно сгенерировано за {elapsed:.2f} секунд.")

        try:
            letter_json = extract_json(letter_raw)
            print(letter_json)
            subject = letter_json.get("subject", "")
            print(subject)
            body = letter_json.get("body", "")
            print(body)
        except Exception as e:
            logger.warning(f"Ошибка парсинга JSON-ответа: {e}")
            subject = ""
            body = letter_raw  # fallback

        # Логирование потребления памяти
        logger.info(
            f"Потребление памяти после генерации письма: "
            f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
        )

        # Обновление состояния с сгенерированным письмом
        return {**state, "subject": subject, "letter": body}

    except Exception as e:
        logger.error(f"Ошибка при генерации письма: {e}")
        return {**state, "subject": "", "letter": ""}


async def output_node(state: LetterState) -> LetterState:
    """
    Возвращает состояние с сгенерированным письмом.

    Args:
        state: Состояние конвейера с письмом.

    Returns:
        Состояние с письмом (для совместимости с LangGraph).
    """
    # Проверка наличия темы
    if not state.get("subject"):
        logger.warning("Тема письма отсутствует в состоянии.")
        return {**state, "subject": "", "letter": ""}

    # Проверка наличия письма
    if not state.get("letter"):
        logger.warning("Письмо отсутствует в состоянии.")
        return {**state, "subject": "", "letter": ""}

    # Возврат состояния для совместимости с LangGraph
    return state