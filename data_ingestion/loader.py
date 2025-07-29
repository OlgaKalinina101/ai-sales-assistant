import os
from typing import Iterator
import pdfplumber
from llama_index.readers.file import MarkdownReader
from llama_index.core.schema import Document

def read_md_documents(dir_path: str) -> Iterator[Document]:
    """
    Загружает все .md-файлы из указанной директории с помощью MarkdownReader.

    Args:
        dir_path: Путь к директории, где лежат .md файлы.

    Returns:
        Итератор, возвращающий Document объекты по одному.
    """
    # Инициализация читателя Markdown
    reader = MarkdownReader()

    # Формирование списка путей к .md файлам
    md_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".md")
    ]

    # Чтение и возврат документов по одному через генератор
    for file in md_files:
        try:
            docs = reader.load_data(file)
            for doc in docs:
                doc.metadata = {"source": os.path.basename(file)}
                yield doc
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file}: {e}", exc_info=True)


def read_pdf_document(pdf_path: str) -> Document:
    """
    Извлекает текст из PDF и оборачивает в Document.

    Args:
        pdf_path: Путь к PDF-файлу.

    Returns:
        Один Document с полным текстом.
    """
    # Открытие PDF-файла и инициализация буфера для текста
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Извлечение текста по одной странице для уменьшения памяти
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text.append(text)

        # Объединение текста всех страниц
        combined_text = "\n".join(full_text)

        # Создание объекта Document с метаданными
        return Document(text=combined_text, metadata={"source": os.path.basename(pdf_path)})

    except Exception as e:
        logger.error(f"Ошибка при чтении PDF {pdf_path}: {e}", exc_info=True)
        return Document(text="", metadata={"source": os.path.basename(pdf_path), "error": str(e)})