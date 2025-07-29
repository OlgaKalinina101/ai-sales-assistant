from typing import TypedDict, List


class LetterState(TypedDict):
    """
    Типизированное состояние для конвейера генерации письма.

    Attributes:
        user_input: Словарь с пользовательскими данными (контакт, должность, компания, сегмент).
        chunks: Список релевантных чанков из базы знаний.
        prompt: Промпт для генерации письма.
        letter: Сгенерированное письмо."""
    user_input: dict
    chunks: List[str]
    prompt: str
    subject: str
    letter: str
