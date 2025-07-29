import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """Настраивает и возвращает логгер с заданным именем.

    Args:
        name: Имя логгера.

    Returns:
        Настроенный объект логгера.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger