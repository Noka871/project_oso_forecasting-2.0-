import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, GlobalAveragePooling1D, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time, json, os
from utils.data_loader import OzoneDataLoader
from utils.logger import logger

class ModelComparator:
    def __init__(self):
        self.data_loader = OzoneDataLoader()
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        logger.info("Инициализирован ModelComparator")

    def prepare_data(self, data=None, sequence_length=12):
        if data is None:
            data = self.data_loader.create_demo_oso_data()
        values = data['oso'].values
        X, y = [], []
        for i in range(len(values) - sequence_length):
            X.append(values[i:i + sequence_length])
            y.append(values[i + sequence_length])
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)
        split = int(len(X) * 0.8)
        self.X_train, self.X_val = X[:split], X[split:]
        self.y_train, self.y_val = y[:split], y[split:]
        return self.X_train, self.y_train, self.X_val, self.y_val

    def build_models(self):
        s = (self.X_train.shape[1], self.X_train.shape[2])
        m1 = Sequential([LSTM(64, input_shape=s), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
        m1.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['LSTM'] = m1
        m2 = Sequential([LSTM(128, return_sequences=True, input_shape=s), Dropout(0.3),
                         LSTM(64), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
        m2.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['Deep_LSTM'] = m2
        m3 = Sequential([Bidirectional(LSTM(64), input_shape=s), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
        m3.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['Bidirectional_LSTM'] = m3
        m4 = Sequential([GRU(64, input_shape=s), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
        m4.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['GRU'] = m4
        m5 = Sequential([Conv1D(64, 3, activation='relu', input_shape=s), Conv1D(32, 3, activation='relu'),
                         GlobalAveragePooling1D(), Dense(32, activation='relu'), Dropout(0.2), Dense(1)])
        m5.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['CNN'] = m5
        m6 = Sequential([Conv1D(64, 3, activation='relu', input_shape=s), LSTM(128),
                         Dense(64, activation='relu'), Dropout(0.3), Dense(32, activation='relu'), Dense(1)])
        m6.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.models['CNN_LSTM_Hybrid'] = m6
        return self.models

    def train_and_evaluate(self, epochs=30, batch_size=32):
        self.results = {}
        for name, model in self.models.items():
            t0 = time.time()
            h = model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val),
                          epochs=epochs, batch_size=batch_size, verbose=0)
            tt = time.time() - t0
            pred = model.predict(self.X_val, verbose=0)
            mae = mean_absolute_error(self.y_val, pred)
            rmse = np.sqrt(mean_squared_error(self.y_val, pred))
            r2 = r2_score(self.y_val, pred)
            self.results[name] = {
                'model': model,
                'history': h.history,
                'metrics': {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2), 'training_time': float(tt)},
                'predictions': pred.flatten().tolist()
            }
        return self.results

    def create_comparison_table(self):
        rows = []
        for name, r in self.results.items():
            m = r['metrics']
            rows.append({
                'Архитектура': name,
                'MAE': f"{m['MAE']:.3f}",
                'RMSE': f"{m['RMSE']:.3f}",
                'R2': f"{m['R2']:.3f}",
                'Время': f"{m['training_time']:.1f}",
                'Параметры': r['model'].count_params()
            })
        return pd.DataFrame(rows)

    def plot_comparison(self, save_path='experiments/results/'):
        os.makedirs(save_path, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(self.results.keys())
        mae = [self.results[n]['metrics']['MAE'] for n in names]
        ax.bar(names, mae, color='skyblue')
        ax.set_title('Сравнение MAE')
        ax.set_xticklabels(names, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'comparison.png'), dpi=150)
        plt.close()
        return fig

    def save_results(self, save_path='experiments/results/'):
        os.makedirs(save_path, exist_ok=True)
        self.create_comparison_table().to_csv(os.path.join(save_path, 'comparison_table.csv'),
                                               index=False, encoding='utf-8')
        mdict = {name: r['metrics'] for name, r in self.results.items()}
        with open(os.path.join(save_path, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(mdict, f, indent=4, ensure_ascii=False)