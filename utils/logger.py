import logging
import os
from datetime import datetime
import sys
import traceback

class OzoneLogger:
    def __init__(self, name="OzoneForecasting", log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        if not os.path.exists('logs'):
            os.makedirs('logs')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        log_filename = f"logs/ozone_forecasting_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

    def info(self, message, *args):
        self.logger.info(message, *args)

    def error(self, message, *args):
        self.logger.error(message, *args)

    def warning(self, message, *args):
        self.logger.warning(message, *args)

    def debug(self, message, *args):
        self.logger.debug(message, *args)

    def critical(self, message, *args):
        if isinstance(message, Exception):
            self.logger.critical(f"{str(message)}\n{traceback.format_exc()}")
        else:
            self.logger.critical(message)

logger = OzoneLogger()

def log_function_call(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Вызов: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Успешно: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Ошибка в {func.__name__}: {e}\n{traceback.format_exc()}")
            raise
    return wrapper

def log_model_training(model_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Начало обучения: {model_name}")
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                logger.info(f"Модель {model_name} обучена за {datetime.now() - start}")
                return result
            except Exception as e:
                logger.error(f"Ошибка обучения {model_name}: {e}\n{traceback.format_exc()}")
                raise
        return wrapper
    return decorator

def log_data_operation(operation_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Операция: {operation_name}")
            try:
                result = func(*args, **kwargs)
                if hasattr(result, 'shape'):
                    logger.info(f"{operation_name} завершена. Размер: {result.shape}")
                else:
                    logger.info(f"{operation_name} завершена")
                return result
            except Exception as e:
                logger.error(f"Ошибка операции {operation_name}: {e}\n{traceback.format_exc()}")
                raise
        return wrapper
    return decorator