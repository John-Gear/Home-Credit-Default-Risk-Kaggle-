import logging

# инициализация глобального логгера
def get_logger(name: str = 'app'):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s') # формат
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger