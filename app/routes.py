
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from app.letter_pipeline.graph import chain
import psutil

from utils.logger import setup_logger

# Инициализация логгера ДО импорта роутера
logger = setup_logger("letter_pipeline")

router = APIRouter()


# Определение модели для пользовательского ввода
class UserInput(BaseModel):
    """
    Модель для пользовательских данных, необходимых для генерации письма.

    Attributes:
        контакт: Имя контактного лица.
        должность: Должность контактного лица.
        название_компании: Название компании.
        сегмент: Сегмент рынка компании.
    """
    контакт: str = Field(..., max_length=100, description="Имя контактного лица")
    должность: str = Field(..., max_length=100, description="Должность контактного лица")
    название_компании: str = Field(..., max_length=200, description="Название компании")
    сегмент: str = Field(..., max_length=100, description="Сегмент рынка компании")


# Определение модели для тела запроса
class RequestBody(BaseModel):
    """
    Модель для тела запроса, содержащего пользовательский ввод.

    Attributes:
        user_input: Данные пользователя для генерации письма.
    """
    user_input: UserInput


# Определение эндпоинта для генерации письма
@router.post("/generate_email")
async def generate_letter(body: RequestBody) -> Dict[str, str]:
    """
    Генерирует персонализированное деловое письмо на основе пользовательских данных.

    Args:
        body: Тело запроса с пользовательскими данными.

    Returns:
        Словарь с сгенерированным письмом.

    Raises:
        HTTPException: Если произошла ошибка при генерации письма.
    """
    # Преобразование Pydantic модели в словарь
    logger.info("Получен запрос")
    user_input = body.user_input.dict()

    # Логирование пользовательского ввода (обрезка для экономии памяти)
    logger.debug(f"Получен user_input: {str(user_input)[:500]}")

    # Вызов конвейера для генерации письма
    try:
        result = await chain.ainvoke({"user_input": user_input})

        subject = result.get("subject", "").strip()
        body_text = result.get("letter", "").strip()

        # Проверка наличия письма в результате
        if not body_text:
            logger.error("Письмо не сгенерировано.")
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать письмо.")

        # Логирование потребления памяти
        logger.info(
            f"Потребление памяти после генерации письма: "
            f"{psutil.Process().memory_info().rss / 1024**2:.2f} МБ"
        )
        logger.debug(f"Тема: {subject}\nТекст письма: {body_text[:1000]}")

        # Формирование ответа
        return {"subject": subject, "letter": body_text}

    except Exception as e:
        # Логирование ошибки и возврат HTTP-ошибки
        logger.error(f"Ошибка при генерации письма: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации письма: {str(e)}")