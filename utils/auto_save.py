# utils/auto_save.py
import os
import shutil
from datetime import datetime
import pandas as pd


def auto_save_predictions(predictions_df, max_history=50):
    """
    Автосохранение прогнозов с историей

    Args:
        predictions_df: DataFrame с прогнозами
        max_history: максимальное количество хранимых версий (по умолчанию 50)

    Returns:
        tuple: (основной_файл, архивный_файл_или_None)
    """
    base_dir = 'data/predictions'

    # Создаем директории если нет
    os.makedirs(base_dir, exist_ok=True)

    # Основной файл (перезаписывается)
    main_file = os.path.join(base_dir, 'ОСО_predict.dat')

    # 1. Сохраняем текущий прогноз в основной файл
    predictions_df.to_csv(main_file, index=False)

    # 2. Автосохранение в историю (если файл уже существовал)
    archive_file = None
    if os.path.exists(main_file):
        # Создаем директорию для истории если нет
        history_dir = os.path.join(base_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)

        # Генерируем имя файла с timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = os.path.join(history_dir, f'ОСО_predict_{timestamp}.dat')

        # Копируем в историю
        shutil.copy2(main_file, archive_file)

        # 3. Очищаем старые версии (оставляем только max_history)
        cleanup_old_predictions(history_dir, max_history)

    return main_file, archive_file


def cleanup_old_predictions(history_dir, max_history):
    """Удаляет старые файлы, оставляя только max_history последних"""
    try:
        # Получаем все файлы истории
        history_files = []
        for f in os.listdir(history_dir):
            if f.startswith('ОСО_predict_') and f.endswith('.dat'):
                filepath = os.path.join(history_dir, f)
                history_files.append((filepath, os.path.getmtime(filepath)))

        # Сортируем по времени модификации (новые в конце)
        history_files.sort(key=lambda x: x[1])

        # Удаляем старые, если их больше max_history
        if len(history_files) > max_history:
            files_to_delete = len(history_files) - max_history
            for i in range(files_to_delete):
                os.remove(history_files[i][0])

    except Exception as e:
        print(f"Ошибка при очистке истории: {e}")


def get_prediction_history(limit=10):
    """Получить последние прогнозы из истории"""
    history_dir = os.path.join('data/predictions', 'history')
    if not os.path.exists(history_dir):
        return []

    history_files = []
    for f in os.listdir(history_dir):
        if f.startswith('ОСО_predict_') and f.endswith('.dat'):
            filepath = os.path.join(history_dir, f)
            mtime = os.path.getmtime(filepath)
            history_files.append({
                'filename': f,
                'filepath': filepath,
                'timestamp': datetime.fromtimestamp(mtime),
                'size_kb': os.path.getsize(filepath) / 1024
            })

    # Сортируем по времени (новые сверху)
    history_files.sort(key=lambda x: x['timestamp'], reverse=True)

    return history_files[:limit]