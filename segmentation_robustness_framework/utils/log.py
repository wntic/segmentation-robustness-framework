import logging


def get_logger() -> logging.Logger:
    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Log formatter
    formatter = logging.Formatter("{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M:%S")

    # Handlers
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        filename="app.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Adding handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
