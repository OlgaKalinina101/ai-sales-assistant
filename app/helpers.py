import json
import re
from typing import Dict
import psutil

from utils.logger import setup_logger

# Инициализация логгера
logger = setup_logger("letter_pipeline")


def extract_json(text: str) -> dict:
    """
    Извлекает JSON-объект из текста, возвращает словарь.

    Args:
        text: Входной текст, содержащий JSON (возможно, с оберткой).

    Returns:
        dict: Словарь с ответом модели, или пустой словарь в случае ошибки.
    """
    # Проверка входного текста
    if not text or not text.strip():
        logger.warning("Входной текст пустой, возвращается пустой словарь.")
        return {}

    # 1. Удаление Markdown-обертки, если она есть
    text = re.sub(r"^```json\s*|```$", "", text.strip(), flags=re.IGNORECASE).strip()

    # 2. Поиск JSON-подобного блока
    try:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            logger.warning("Не удалось найти JSON-структуру.")
            return {}

        json_block = match.group(0)
        parsed = json.loads(json_block)

        if not isinstance(parsed, dict):
            logger.warning("Извлеченный JSON не является словарём.")
            return {}

        return parsed

    except json.JSONDecodeError as e:
        logger.warning(f"Ошибка при разборе JSON: {str(e)[:100]}")
        return {}
    except Exception as e:
        logger.warning(f"Неожиданная ошибка при извлечении JSON: {str(e)[:100]}")
        return {}