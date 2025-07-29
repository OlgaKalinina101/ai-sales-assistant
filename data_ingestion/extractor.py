import os
import shutil
import zipfile

def extract_nested_zip(zip_path: str, extract_to: str) -> None:
    """Распаковывает вложенные ZIP-архивы в плоскую структуру директории.

    Args:
        zip_path: Путь к исходному ZIP-файлу.
        extract_to: Путь к директории, куда будут распакованы файлы.

    Returns:
        None
    """
    # Создание целевой директории для распаковки, если она не существует
    os.makedirs(extract_to, exist_ok=True)

    # Запуск рекурсивной распаковки исходного ZIP-файла
    _extract(zip_path, extract_to)


def _extract(zip_file: str, path: str) -> None:
    """Внутренняя функция для рекурсивной распаковки ZIP-архивов.

    Args:
        zip_file: Путь к ZIP-файлу для распаковки.
        path: Путь к директории для распаковки содержимого.

    Returns:
        None
    """
    # Определение вспомогательной функции для генерации уникальных путей файлов
    def _get_unique_path(target: str) -> str:
        """Генерирует уникальный путь для файла, избегая перезаписи.

        Args:
            target: Предполагаемый путь к файлу.

        Returns:
            Уникальный путь для файла.
        """
        # Проверка существования файла и добавление суффикса при необходимости
        base, ext = os.path.splitext(target)
        counter = 1
        unique_path = target
        while os.path.exists(unique_path):
            unique_path = f"{base}_{counter}{ext}"
            counter += 1
        return unique_path

    # Открытие ZIP-архива и итерация по его содержимому
    with zipfile.ZipFile(zip_file, 'r') as z:
        for member in z.infolist():
            # Пропуск директорий
            if member.is_dir():
                continue

            # Извлечение имени файла без директорий и пропуск пустых имен
            filename = os.path.basename(member.filename)
            if not filename:
                continue

            # Формирование уникального пути для сохранения файла
            target_path = _get_unique_path(os.path.join(path, filename))

            # Потоковая распаковка файла в целевую директорию
            with z.open(member) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)

            # Проверка, является ли распакованный файл ZIP-архивом, и рекурсивная обработка
            if zipfile.is_zipfile(target_path):
                nested_path = f"{target_path}_extracted"
                os.makedirs(nested_path, exist_ok=True)
                _extract(target_path, nested_path)


if __name__ == "__main__":
    from config import ZIP_PATH, PROCESSED_DATA_DIR
    extract_nested_zip(ZIP_PATH, PROCESSED_DATA_DIR)

