import os
import shutil
from typing import Optional

from data_ingestion.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import setup_logger

# Инициализация логгера
logger = setup_logger("cleaner")


def clear_directory(path: str) -> None:
    """
    Удаляет все файлы и подкаталоги в указанной директории.

    Args:
        path: Путь к директории для очистки.

    Returns:
        None
    """
    # Проверка существования директории
    if not os.path.exists(path):
        logger.warning(f"Директория {path} не существует — нечего очищать.")
        return

    # Итерация по содержимому директории с использованием os.scandir
    with os.scandir(path) as entries:
        for entry in entries:
            item_path = entry.path
            try:
                # Удаление файла или символической ссылки
                if entry.is_file() or entry.is_symlink():
                    os.unlink(item_path)
                    logger.info(f"Удалён файл: {item_path}")
                # Удаление подкаталога
                elif entry.is_dir():
                    shutil.rmtree(item_path)
                    logger.info(f"Удалена директория: {item_path}")
            except Exception as e:
                logger.error(f"Ошибка при удалении {item_path}: {e}")

def clear_raw_data() -> None:
    """Очищает директорию с необработанными данными."""
    # Вызов функции очистки для RAW_DATA_DIR
    clear_directory(RAW_DATA_DIR)


def clear_processed_data() -> None:
    """Очищает директорию с обработанными данными."""
    # Вызов функции очистки для PROCESSED_DATA_DIR
    clear_directory(PROCESSED_DATA_DIR)


if __name__ == "__main__":
    clear_raw_data()
    clear_processed_data()