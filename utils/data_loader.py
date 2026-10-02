import pandas as pd
import numpy as np
import os
import re
from utils.logger import log_function_call, log_data_operation, logger


class OzoneDataLoader:
    def __init__(self):
        self.data_path = "data/ram/"
        logger.info("Инициализирован OzoneDataLoader")

    def _parse_real_file(self, file_path, value_name):
        """Универсальный парсер: '1  01.01.2006  325' или '1\t01.01.2006\t325'"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_lines = f.readlines()

        logger.info(f"Прочитано строк: {len(raw_lines)}")

        rows = []
        skipped = 0
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            if len(parts) < 3:
                skipped += 1
                continue
            date_str = parts[1]
            value_str = parts[2].replace(',', '.')
            if date_str.count('.') != 2:
                skipped += 1
                continue
            try:
                value = float(value_str)
            except ValueError:
                skipped += 1
                continue
            rows.append((date_str, value))

        logger.info(f"Распознано строк: {len(rows)}, пропущено: {skipped}")

        if not rows:
            raise ValueError("Не удалось распарсить ни одной строки файла")

        df = pd.DataFrame(rows, columns=['date_str', value_name])
        df['date'] = pd.to_datetime(df['date_str'], format='%d.%m.%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        monthly = df.groupby(['year', 'month'])[value_name].mean().reset_index()
        monthly = monthly.sort_values(['year', 'month']).reset_index(drop=True)

        return monthly, len(df)

    @log_data_operation("Создание демонстрационных данных ОСО")
    def create_demo_oso_data(self):
        years = range(1960, 2025)
        months = range(1, 13)
        data = []
        for year in years:
            for month in months:
                base_oso = 300
                seasonal = 20 * np.sin((month - 3) * 2 * np.pi / 12)
                trend = -0.1 * (year - 1960)
                noise = np.random.normal(0, 5)
                anomaly = -8 if (year >= 2020 and month in [9, 10, 11, 12]) else 0
                oso_value = base_oso + seasonal + trend + noise + anomaly
                data.append({
                    'year': year,
                    'month': month,
                    'oso': max(250, min(350, oso_value)),
                    'latitude': 56.5,
                    'longitude': 84.95,
                    'temperature': 15 + 20 * np.sin((month - 1) * 2 * np.pi / 12) + np.random.normal(0, 3),
                    'pressure': 1013 + np.random.normal(0, 8)
                })
        df = pd.DataFrame(data)
        logger.info(f"Создано демо-данных: {len(df)} записей")
        return df

    @log_data_operation("Создание демонстрационных индексных данных")
    def create_demo_index_data(self):
        years = range(1960, 2025)
        data = []
        for year in years:
            for month in range(1, 13):
                data.append({
                    'year': year, 'month': month,
                    'oso_index': (month + year % 3) % 12 + 1,
                    'seasonal_factor': 0.8 + 0.3 * np.sin((month - 1) * 2 * np.pi / 12),
                    'trend_component': (year - 1960) * 0.02,
                    'anomaly_flag': 1 if (year >= 2020 and month in [9, 10, 11, 12]) else 0
                })
        df = pd.DataFrame(data)
        logger.info(f"Создано индексных данных: {len(df)} записей")
        return df

    def load_real_oso(self, file_path):
        try:
            monthly, n_days = self._parse_real_file(file_path, 'oso')
            logger.info(f"Загружено ОСО: {n_days} дней -> {len(monthly)} месяцев")
            return monthly
        except Exception as e:
            logger.error(f"Ошибка загрузки ОСО: {e}")
            raise

    def load_real_density(self, file_path):
        try:
            monthly, n_days = self._parse_real_file(file_path, 'density')
            logger.info(f"Загружена плотность: {n_days} дней -> {len(monthly)} месяцев")
            return monthly
        except Exception as e:
            logger.error(f"Ошибка загрузки плотности: {e}")
            raise

    def merge_oso_density(self, oso_df, density_df):
        try:
            merged = pd.merge(oso_df, density_df, on=['year', 'month'], how='inner')
            logger.info(f"Объединено: {len(merged)} месяцев с ОСО и плотностью")
            return merged
        except Exception as e:
            logger.error(f"Ошибка объединения: {e}")
            raise

    @log_function_call
    def analyze_data(self, data):
        analysis = {
            'total_records': len(data),
            'columns': list(data.columns),
            'date_range': None,
            'oso_stats': None
        }
        if 'year' in data.columns and 'month' in data.columns:
            years = data['year'].unique()
            analysis['date_range'] = f"{min(years)}-{max(years)}"
        if 'oso' in data.columns:
            analysis['oso_stats'] = {
                'mean': data['oso'].mean(),
                'min': data['oso'].min(),
                'max': data['oso'].max(),
                'std': data['oso'].std()
            }
        return analysis