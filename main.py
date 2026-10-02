import customtkinter as ctk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ozone_model import OzoneHybridModel
import threading
import os
import json
import traceback
import shutil
from datetime import datetime
from utils.data_loader import OzoneDataLoader
from utils.logger import logger, log_function_call
from experiments.model_comparison import ModelComparator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BG_D, BG_L = '#2b2b2b', '#ffffff'
FG_D, FG_L = 'white', 'black'

ABOUT_FILE = "about_content.json"
RESULTS_DIR = "results"

DEFAULT_ABOUT = {
    "about": "OSO Forecasting v2.5\n\nПрограмма для прогнозирования общего содержания озона (ОСО).",
    "help": "РУКОВОДСТВО\n\n1. Загрузите данные (любой файл TXT/CSV/DAT)\n2. Обучите модель\n3. Сравните модели\n4. Выполните прогноз\n5. Визуализация",
    "license": "ЛИЦЕНЗИЯ\n\nУчебный проект ТУСУР, 2026."
}


def load_about_content():
    if os.path.exists(ABOUT_FILE):
        try:
            with open(ABOUT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_ABOUT.copy()


def save_about_content(data):
    with open(ABOUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_results_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    return RESULTS_DIR


class ModernOzoneApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        logger.info("Запуск приложения")
        ensure_results_dir()

        self.title_font = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.subtitle_font = ctk.CTkFont(family="Arial", size=14, weight="bold")
        self.normal_font = ctk.CTkFont(family="Arial", size=12)
        self.small_font = ctk.CTkFont(family="Arial", size=10)

        self.title("OSO Forecasting - Прогнозирование озонового слоя")
        self.geometry("1400x950")
        self.minsize(1200, 800)

        self.data_loader = OzoneDataLoader()
        self.model = OzoneHybridModel()
        self.comparator = None
        self.oso_data = None
        self.forecast = None
        self.comparison_results = None
        self.current_step = 0
        self.current_theme = "Dark"
        self.about_content = load_about_content()
        self.current_data_file = None
        self.current_density_file = None

        self.create_sidebar()
        self.create_main_content()
        self.create_status_bar()
        logger.info("Интерфейс инициализирован")

    def get_colors(self):
        return (BG_D, FG_D) if self.current_theme == "Dark" else (BG_L, FG_L)

    def apply_theme(self, ax):
        bg, fg = self.get_colors()
        self.figure.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for s in ax.spines.values():
            s.set_color(fg)

    def toggle_theme(self):
        if self.current_theme == "Dark":
            self.current_theme = "Light"
            ctk.set_appearance_mode("Light")
            self.theme_btn.configure(text="Тёмная тема")
        else:
            self.current_theme = "Dark"
            ctk.set_appearance_mode("Dark")
            self.theme_btn.configure(text="Светлая тема")
        try:
            self.show_historical()
        except Exception:
            pass

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        ctk.CTkLabel(self.sidebar, text="OSO Forecasting", font=self.title_font).pack(pady=(30, 10), padx=20)
        ctk.CTkLabel(self.sidebar, text="Прогнозирование и анализ", font=self.small_font, text_color="gray70").pack(pady=(0, 15))
        self.theme_btn = ctk.CTkButton(
            self.sidebar, text="Светлая тема", command=self.toggle_theme,
            font=self.normal_font, height=35,
            fg_color=("#e0e0e0", "#3a3a3a"),
            text_color=("#000000", "#ffffff"),
            hover_color=("#d0d0d0", "#4a4a4a")
        )
        self.theme_btn.pack(pady=(0, 20), padx=20, fill="x")
        steps_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        steps_frame.pack(fill="x", padx=20, pady=10)
        steps = [
            "1. Загрузить данные", "2. Обучить модель", "3. Сравнить модели",
            "4. Выполнить прогноз", "5. Визуализация"
        ]
        self.step_buttons = []
        for i, s in enumerate(steps):
            btn = ctk.CTkButton(
                steps_frame, text=s, font=self.normal_font, height=45, anchor="w",
                command=lambda idx=i: self.set_current_step(idx),
                state="disabled" if i > 0 else "normal"
            )
            btn.pack(fill="x", pady=3)
            self.step_buttons.append(btn)
        self.step_buttons[0].configure(fg_color="#2E8B57")
        self.about_btn = ctk.CTkButton(
            self.sidebar, text="О программе", command=self.show_about_window,
            font=self.normal_font, height=40,
            fg_color=("#c0c0c0", "#404040"),
            text_color=("#000000", "#ffffff"),
            hover_color=("#b0b0b0", "#505050")
        )
        self.about_btn.pack(side="bottom", padx=20, pady=(0, 5), fill="x")
        ctk.CTkLabel(self.sidebar, text="Версия 2.5", font=ctk.CTkFont(size=9), text_color="gray60").pack(side="bottom", pady=(0, 5))

    def create_main_content(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True)
        self.tab_data = self.tabview.add("Данные")
        self.tab_model = self.tabview.add("Модель")
        self.tab_experiments = self.tabview.add("Эксперименты")
        self.tab_forecast = self.tabview.add("Прогноз")
        self.tab_visualization = self.tabview.add("Визуализация")
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_experiments_tab()
        self.setup_forecast_tab()
        self.setup_visualization_tab()

    def setup_data_tab(self):
        ctk.CTkLabel(self.tab_data, text="Загрузка и анализ данных", font=self.title_font).pack(pady=20)
        bf = ctk.CTkFrame(self.tab_data, fg_color="transparent")
        bf.pack(pady=10)
        ctk.CTkButton(bf, text="Загрузить демо-данные", command=self.load_demo_data,
                      font=self.normal_font, height=40, width=200).pack(side="left", padx=5)
        ctk.CTkButton(bf, text="Загрузить из файла", command=self.load_from_file,
                      font=self.normal_font, height=40, width=200).pack(side="left", padx=5)
        hint = ("Поддерживаемые форматы:\n"
                "• TXT/DAT: строки вида «номер  дата  значение»\n"
                "• CSV: колонки year, month, oso\n"
                "Имя файла — любое")
        ctk.CTkLabel(self.tab_data, text=hint, font=self.small_font, justify="left", text_color="gray60").pack(pady=5)
        self.data_info_frame = ctk.CTkFrame(self.tab_data)
        self.data_info_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.data_info_text = ctk.CTkTextbox(self.data_info_frame, height=200)
        self.data_info_text.pack(fill="both", expand=True)
        self.data_info_text.insert("1.0", "Данные не загружены.")
        self.data_info_text.configure(state="disabled")

    def setup_model_tab(self):
        ctk.CTkLabel(self.tab_model, text="Обучение гибридной модели", font=self.title_font).pack(pady=20)
        af = ctk.CTkFrame(self.tab_model, corner_radius=8)
        af.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(af, text="Архитектура Conv1D + LSTM:", font=self.subtitle_font).pack(pady=(10, 5))
        txt = "Conv1D: 64 фильтра, ядро=3, ReLU\nLSTM: 128 нейронов\nDense: 64 -> 32\nDropout: 0.3\nAdam (lr=0.001), MSE"
        ctk.CTkLabel(af, text=txt, font=self.normal_font, justify="left").pack(pady=(5, 10), padx=15)
        bf = ctk.CTkFrame(self.tab_model, fg_color="transparent")
        bf.pack(pady=20)
        self.train_btn = ctk.CTkButton(bf, text="Начать обучение", command=self.train_model,
                                       font=self.normal_font, height=50, width=220,
                                       fg_color="#2E8B57", state="disabled")
        self.train_btn.pack(side="left", padx=5)
        self.stop_btn = ctk.CTkButton(bf, text="Остановить", command=self.stop_training,
                                      font=self.normal_font, height=50, width=150,
                                      fg_color="#B22222", state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.progress_bar = ctk.CTkProgressBar(self.tab_model, height=20)
        self.progress_bar.pack(fill="x", padx=50, pady=10)
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(self.tab_model, text="Готов к обучению", font=self.small_font)
        self.progress_label.pack(pady=5)
        rf = ctk.CTkFrame(self.tab_model)
        rf.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(rf, text="Результаты:", font=self.subtitle_font).pack(anchor="w", pady=(5, 5))
        self.training_results = ctk.CTkTextbox(rf, height=150)
        self.training_results.pack(fill="both", expand=True)
        self.training_results.insert("1.0", "Результаты обучения появятся здесь")
        self.training_results.configure(state="disabled")

    def setup_experiments_tab(self):
        ctk.CTkLabel(self.tab_experiments, text="Сравнение архитектур", font=self.title_font).pack(pady=20)
        bf = ctk.CTkFrame(self.tab_experiments, fg_color="transparent")
        bf.pack(pady=10)
        self.compare_btn = ctk.CTkButton(bf, text="Запустить сравнение", command=self.run_comparison,
                                         font=self.normal_font, height=50, width=250,
                                         fg_color="#8A2BE2", state="disabled")
        self.compare_btn.pack(pady=5)
        self.save_results_btn = ctk.CTkButton(bf, text="Сохранить эксперимент",
                                              command=self.save_experiment,
                                              font=self.normal_font, height=40, width=300,
                                              fg_color="#1E90FF", state="disabled")
        self.save_results_btn.pack(pady=5)
        rf = ctk.CTkFrame(self.tab_experiments)
        rf.pack(fill="both", expand=True, padx=20, pady=10)
        self.exp_tabview = ctk.CTkTabview(rf)
        self.exp_tabview.pack(fill="both", expand=True)
        self.exp_table_tab = self.exp_tabview.add("Таблица")
        self.exp_metrics_tab = self.exp_tabview.add("Метрики")
        self.exp_analysis_tab = self.exp_tabview.add("Анализ")
        self.comparison_text = ctk.CTkTextbox(self.exp_table_tab, height=300)
        self.comparison_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.comparison_text.insert("1.0", "Таблица появится после запуска")
        self.comparison_text.configure(state="disabled")
        self.metrics_text = ctk.CTkTextbox(self.exp_metrics_tab, height=300)
        self.metrics_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.metrics_text.insert("1.0", "Метрики появятся после запуска")
        self.metrics_text.configure(state="disabled")
        self.analysis_text = ctk.CTkTextbox(self.exp_analysis_tab, height=300)
        self.analysis_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.analysis_text.insert("1.0", "Анализ появится после запуска")
        self.analysis_text.configure(state="disabled")
        self.exp_tabview.pack_forget()

    def setup_forecast_tab(self):
        ctk.CTkLabel(self.tab_forecast, text="Прогнозирование ОСО", font=self.title_font).pack(pady=20)
        sf = ctk.CTkFrame(self.tab_forecast, corner_radius=8)
        sf.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(sf, text="Настройки прогноза:", font=self.subtitle_font).pack(pady=(10, 5))
        mf = ctk.CTkFrame(sf, fg_color="transparent")
        mf.pack(fill="x", pady=5, padx=15)
        ctk.CTkLabel(mf, text="Модель:", font=self.normal_font).pack(side="left", padx=5)
        self.model_selector = ctk.CTkComboBox(mf, values=["CNN-LSTM (гибридная)"],
                                              font=self.normal_font, width=200, state="disabled")
        self.model_selector.pack(side="left", padx=5)
        pf = ctk.CTkFrame(sf, fg_color="transparent")
        pf.pack(fill="x", pady=5, padx=15)
        ctk.CTkLabel(pf, text="Период (месяцев):", font=self.normal_font).pack(side="left", padx=5)
        self.forecast_period = ctk.CTkEntry(pf, placeholder_text="12", font=self.normal_font, width=100)
        self.forecast_period.pack(side="left", padx=5)
        self.forecast_period.insert(0, "12")
        self.forecast_btn = ctk.CTkButton(self.tab_forecast, text="Выполнить прогноз",
                                          command=self.run_forecast, font=self.normal_font,
                                          height=50, state="disabled")
        self.forecast_btn.pack(pady=20)
        self.forecast_results = ctk.CTkTextbox(self.tab_forecast, height=250)
        self.forecast_results.pack(fill="both", expand=True, padx=20, pady=10)
        self.forecast_results.insert("1.0", "Результаты прогноза появятся здесь")
        self.forecast_results.configure(state="disabled")

    def setup_visualization_tab(self):
        ctk.CTkLabel(self.tab_visualization, text="Визуализация", font=self.title_font).pack(pady=10)
        cf = ctk.CTkFrame(self.tab_visualization, fg_color="transparent")
        cf.pack(fill="x", padx=20, pady=10)
        buttons = [
            ("Исторические", self.show_historical),
            ("Сезонность", self.show_seasonality),
            ("Тренды", self.show_trends),
            ("Прогноз", self.show_forecast_plot),
            ("Невязка", self.show_residuals_plot),
            ("Кривая обучения", self.show_loss_plot),
            ("Сравнение", self.show_comparison_plot)
        ]
        row = None
        for i, (t, cmd) in enumerate(buttons):
            if i % 4 == 0:
                row = ctk.CTkFrame(cf, fg_color="transparent")
                row.pack(pady=5)
            state = "normal" if t != "Сравнение" else "disabled"
            ctk.CTkButton(row, text=t, command=cmd, font=self.small_font,
                          width=160, state=state).pack(side="left", padx=3)
        self.viz_frame = ctk.CTkFrame(self.tab_visualization)
        self.viz_frame.pack(fill="both", expand=True, padx=20, pady=10)
        bg, _ = self.get_colors()
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor=bg)
        self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.show_welcome_plot()

    def create_status_bar(self):
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.pack_propagate(False)
        self.status_label = ctk.CTkLabel(self.status_bar, text="Готов к работе", font=self.small_font)
        self.status_label.pack(side="left", padx=10, pady=5)

    def set_current_step(self, idx):
        self.current_step = idx
        for i, btn in enumerate(self.step_buttons):
            btn.configure(fg_color="#2E8B57" if i == idx else ("gray75", "gray25"))
        tabs = ["Данные", "Модель", "Эксперименты", "Прогноз", "Визуализация"]
        self.tabview.set(tabs[idx])

    def go_to_next_step(self, delay_ms=1000):
        if self.current_step >= len(self.step_buttons) - 1:
            return
        next_idx = self.current_step + 1
        self.after(delay_ms, lambda: self.set_current_step(next_idx))

    def update_status(self, msg):
        self.status_label.configure(text=msg)
        self.update()

    def show_about_window(self):
        win = ctk.CTkToplevel(self)
        win.title("О программе OSO Forecasting")
        win.geometry("800x600")
        win.minsize(600, 450)
        win.grab_set()
        ctk.CTkLabel(win, text="О программе", font=self.title_font).pack(pady=10)
        tabs = ctk.CTkTabview(win)
        tabs.pack(fill="both", expand=True, padx=20, pady=10)
        tab_about = tabs.add("О программе")
        tab_help = tabs.add("Помощь")
        tab_license = tabs.add("Лицензия")
        text_widgets = {}
        for name, key, tab in [
            ("О программе", "about", tab_about),
            ("Помощь", "help", tab_help),
            ("Лицензия", "license", tab_license)
        ]:
            tb = ctk.CTkTextbox(tab, wrap="word", font=self.normal_font)
            tb.pack(fill="both", expand=True, padx=10, pady=10)
            tb.insert("1.0", self.about_content.get(key, ""))
            tb.configure(state="disabled")
            text_widgets[key] = tb
            ctk.CTkButton(
                tab, text="Редактировать",
                command=lambda k=key, w=text_widgets: self.edit_about_content(k, w),
                font=self.normal_font, height=35, width=180
            ).pack(pady=5)
        ctk.CTkButton(
            win, text="Закрыть", command=win.destroy,
            font=self.normal_font, height=40, width=150
        ).pack(pady=10)
        win.after(100, win.lift)

    def edit_about_content(self, key, text_widgets):
        win = ctk.CTkToplevel(self)
        win.title("Редактор")
        win.geometry("800x600")
        win.grab_set()
        ctk.CTkLabel(win, text="Редактор содержимого", font=self.subtitle_font).pack(pady=10)
        editor = ctk.CTkTextbox(win, wrap="word", font=self.normal_font)
        editor.pack(fill="both", expand=True, padx=20, pady=10)
        editor.insert("1.0", self.about_content.get(key, ""))

        def save_and_close():
            new_text = editor.get("1.0", "end-1c")
            self.about_content[key] = new_text
            save_about_content(self.about_content)
            tb = text_widgets[key]
            tb.configure(state="normal")
            tb.delete("1.0", "end")
            tb.insert("1.0", new_text)
            tb.configure(state="disabled")
            messagebox.showinfo("Успех", "Текст сохранён в about_content.json")
            win.destroy()

        bf = ctk.CTkFrame(win, fg_color="transparent")
        bf.pack(pady=10)
        ctk.CTkButton(bf, text="Сохранить", command=save_and_close,
                      font=self.normal_font, height=40, width=150,
                      fg_color="#2E8B57").pack(side="left", padx=5)
        ctk.CTkButton(bf, text="Отмена", command=win.destroy,
                      font=self.normal_font, height=40, width=150).pack(side="left", padx=5)

    def get_dates(self, data):
        if 'year' in data.columns and 'month' in data.columns:
            try:
                return pd.to_datetime(data[['year', 'month']].assign(day=1))
            except Exception:
                pass
        return pd.date_range('2020-01-01', periods=len(data), freq='ME')

    def _is_real_format(self, path):
        """Проверяет СОДЕРЖИМОЕ файла (не имя): 'номер  дата  значение'.
        Имя файла не учитывается — только содержимое."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                first_lines = [f.readline() for _ in range(20)]
            for line in first_lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    date_str = parts[1]
                    if date_str.count('.') == 2:
                        try:
                            datetime.strptime(date_str, '%d.%m.%Y')
                            return True
                        except ValueError:
                            continue
        except Exception as e:
            logger.error(f"Ошибка проверки формата: {e}")
        return False

    def load_from_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл с данными ОСО",
            filetypes=[("Все", "*.txt *.csv *.dat"), ("TXT", "*.txt"), ("CSV", "*.csv"), ("DAT", "*.dat"), ("Все", "*.*")]
        )
        if not path:
            return
        try:
            self.update_status(f"Загрузка: {os.path.basename(path)}")
            logger.info(f"Выбран файл: {path}")

            is_real = self._is_real_format(path)
            logger.info(f"Распознан как реальный формат: {is_real}")

            if is_real:
                self.oso_data = self.data_loader.load_real_oso(path)
                self.current_data_file = path
                logger.info(f"ОСО загружено: {len(self.oso_data)} месяцев")

                density_path = filedialog.askopenfilename(
                    title="Выберите файл с плотностью (или Отмена)",
                    filetypes=[("Все", "*.txt *.csv *.dat"), ("Все", "*.*")]
                )
                if density_path:
                    try:
                        density_df = self.data_loader.load_real_density(density_path)
                        self.oso_data = self.data_loader.merge_oso_density(self.oso_data, density_df)
                        self.current_density_file = density_path
                        logger.info(f"Плотность добавлена. Итого: {len(self.oso_data)} записей")
                    except Exception as e:
                        logger.warning(f"Плотность не загружена: {e}")
            else:
                if path.endswith('.csv'):
                    self.oso_data = pd.read_csv(path, encoding='utf-8')
                elif path.endswith('.dat'):
                    try:
                        self.oso_data = pd.read_csv(path, sep=r'\s+', encoding='utf-8')
                    except Exception:
                        try:
                            self.oso_data = pd.read_csv(path, sep=',', encoding='utf-8')
                        except Exception:
                            self.oso_data = pd.read_fwf(path, encoding='utf-8')
                else:
                    self.oso_data = pd.read_csv(path, encoding='utf-8')
                self.current_data_file = path

            if 'oso' not in self.oso_data.columns:
                num_cols = self.oso_data.select_dtypes(include=[np.number]).columns
                if len(num_cols) == 0:
                    raise ValueError("Не найдены числовые столбцы")
                self.oso_data = self.oso_data.rename(columns={num_cols[0]: 'oso'})
            if 'year' not in self.oso_data.columns:
                self.oso_data['year'] = 2020 + np.arange(len(self.oso_data)) // 12
            if 'month' not in self.oso_data.columns:
                self.oso_data['month'] = (np.arange(len(self.oso_data)) % 12) + 1

            self._on_data_loaded_file(path)
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить:\n{e}")

    def _on_data_loaded_file(self, path):
        logger.info(f"Загружено из {path}")
        self.update_status(f"Загружено: {os.path.basename(path)}")
        years = self.oso_data['year'].unique()
        period = f"{min(years)}-{max(years)}" if len(years) else "—"
        has_density = 'density' in self.oso_data.columns
        density_line = "Плотность: загружена\n" if has_density else ""
        text = (f"ДАННЫЕ ЗАГРУЖЕНЫ\n\n"
                f"Файл ОСО: {os.path.basename(path)}\n"
                f"Записей: {len(self.oso_data):,}\n"
                f"Период: {period}\n"
                f"{density_line}\n"
                f"Первые строки:\n{self.oso_data.head(3).to_string(index=False)}")
        self.data_info_text.configure(state="normal")
        self.data_info_text.delete("1.0", "end")
        self.data_info_text.insert("1.0", text)
        self.data_info_text.configure(state="disabled")
        self.step_buttons[1].configure(state="normal")
        self.step_buttons[2].configure(state="normal")
        self.train_btn.configure(state="normal")
        self.compare_btn.configure(state="normal")
        self.show_historical()
        messagebox.showinfo("Успех", "Данные загружены!")
        self.go_to_next_step()

    @log_function_call
    def load_demo_data(self):
        self.update_status("Создание демо-данных...")
        self.current_data_file = None
        self.current_density_file = None
        threading.Thread(target=self._demo_thread, daemon=True).start()

    def _demo_thread(self):
        try:
            self.oso_data = self.data_loader.create_demo_oso_data()
            self.after(0, self._on_demo_loaded)
        except Exception as e:
            logger.error(f"Ошибка демо-данных: {e}")
            self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))

    def _on_demo_loaded(self):
        self.update_status("Демо-данные созданы")
        text = (f"ДЕМО-ДАННЫЕ СОЗДАНЫ\n\n"
                f"Период: 1960-2024\n"
                f"Записей: {len(self.oso_data):,}\n"
                f"Регион: Томская область\n\n"
                f"Первые строки:\n{self.oso_data.head(3).to_string(index=False)}")
        self.data_info_text.configure(state="normal")
        self.data_info_text.delete("1.0", "end")
        self.data_info_text.insert("1.0", text)
        self.data_info_text.configure(state="disabled")
        self.step_buttons[1].configure(state="normal")
        self.step_buttons[2].configure(state="normal")
        self.train_btn.configure(state="normal")
        self.compare_btn.configure(state="normal")
        self.show_historical()
        messagebox.showinfo("Успех", "Демо-данные созданы!")
        self.go_to_next_step()

    @log_function_call
    def train_model(self):
        if self.oso_data is None:
            messagebox.showwarning("Внимание", "Загрузите данные!")
            return
        self.update_status("Обучение модели...")
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Инициализация...")
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            def cb(epoch, total, logs):
                self.after(0, lambda p=epoch / total, e=epoch, t=total,
                           l=logs.get('loss', 0), v=logs.get('val_loss', 0):
                           self._update_progress(p, e, t, l, v))
            self.model.train(self.oso_data, epochs=50, progress_callback=cb)
            self.after(0, self._on_trained)
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            self.after(0, lambda: self._on_train_error(str(e)))

    def _update_progress(self, p, e, t, l, v):
        self.progress_bar.set(p)
        self.progress_label.configure(text=f"Эпоха {e}/{t} | loss: {l:.4f} | val_loss: {v:.4f}")

    def stop_training(self):
        self.model.stop_training()
        self.progress_label.configure(text="Остановка...")

    def _on_trained(self):
        self.update_status("Модель обучена")
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_bar.set(1)
        self.progress_label.configure(text="Обучение завершено")
        m = self.model.metrics
        has_density = 'density' in self.oso_data.columns if self.oso_data is not None else False
        text = (f"МОДЕЛЬ ОБУЧЕНА\n\n"
                f"МЕТРИКИ:\n"
                f"MAE: {m.get('mae', 0):.3f}\n"
                f"MSE: {m.get('mse', 0):.3f}\n"
                f"RMSE: {m.get('rmse', 0):.3f}\n"
                f"MAPE: {m.get('mape', 0):.2f}%\n"
                f"R2: {m.get('r2', 0):.3f}\n"
                f"Bias: {m.get('bias', 0):.3f}\n"
                f"Точность: {m.get('accuracy', 0):.1%}\n\n"
                f"Архитектура: Conv1D(64) + LSTM(128) + Dense(64/32) + Dropout 0.3\n"
                f"Оптимизатор: Adam (lr=0.001)\n"
                f"Признаков на входе: {'ОСО + плотность' if has_density else 'только ОСО'}")
        self.training_results.configure(state="normal")
        self.training_results.delete("1.0", "end")
        self.training_results.insert("1.0", text)
        self.training_results.configure(state="disabled")
        self.step_buttons[3].configure(state="normal")
        self.step_buttons[4].configure(state="normal")
        self.forecast_btn.configure(state="normal")
        self.model_selector.configure(state="normal")
        messagebox.showinfo("Успех", "Модель обучена!")
        self.go_to_next_step()

    def _on_train_error(self, msg):
        self.update_status("Ошибка обучения")
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        messagebox.showerror("Ошибка", msg)

    @log_function_call
    def run_comparison(self):
        if self.oso_data is None:
            messagebox.showwarning("Внимание", "Загрузите данные!")
            return
        self.update_status("Сравнение моделей...")
        self.compare_btn.configure(state="disabled")
        threading.Thread(target=self._comparison_thread, daemon=True).start()

    def _comparison_thread(self):
        try:
            self.comparator = ModelComparator()
            self.comparator.prepare_data(data=self.oso_data)
            self.comparator.build_models()
            self.comparison_results = self.comparator.train_and_evaluate(epochs=30)
            self.after(0, self._on_comparison_done)
        except Exception as e:
            logger.error(f"Ошибка сравнения: {e}")
            self.after(0, lambda: self._on_comparison_error(str(e)))

    def _on_comparison_done(self):
        self.update_status("Сравнение завершено")
        self.compare_btn.configure(state="normal")
        self.save_results_btn.configure(state="normal")
        self.exp_tabview.pack(fill="both", expand=True)
        df = self.comparator.create_comparison_table()
        header = f"{'Архитектура':<22}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'Время':>8}{'Парам.':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for _, row in df.iterrows():
            lines.append(
                f"{row['Архитектура']:<22}{row['MAE']:>8}{row['RMSE']:>8}{row['R2']:>8}{row['Время']:>8}{row['Параметры']:>10}"
            )
        table_text = "\n".join(lines)
        self.comparison_text.configure(state="normal")
        self.comparison_text.delete("1.0", "end")
        self.comparison_text.insert("1.0", table_text)
        self.comparison_text.configure(state="disabled")
        mt = f"{'Архитектура':<22}{'MAE':>10}{'RMSE':>10}{'R2':>10}\n"
        mt += "-" * 55 + "\n"
        for name, res in self.comparison_results.items():
            m = res['metrics']
            mt += f"{name:<22}{m['MAE']:>10.3f}{m['RMSE']:>10.3f}{m['R2']:>10.3f}\n"
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", mt)
        self.metrics_text.configure(state="disabled")
        best_mae = min(self.comparison_results.items(), key=lambda x: x[1]['metrics']['MAE'])
        best_r2 = max(self.comparison_results.items(), key=lambda x: x[1]['metrics']['R2'])
        fastest = min(self.comparison_results.items(), key=lambda x: x[1]['metrics']['training_time'])
        at = "АНАЛИЗ РЕЗУЛЬТАТОВ\n"
        at += "=" * 50 + "\n\n"
        at += f"Лучший MAE:    {best_mae[0]:<22} = {best_mae[1]['metrics']['MAE']:.3f}\n"
        at += f"Лучший R2:     {best_r2[0]:<22} = {best_r2[1]['metrics']['R2']:.3f}\n"
        at += f"Самый быстрый: {fastest[0]:<22} = {fastest[1]['metrics']['training_time']:.1f} сек\n"
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", at)
        self.analysis_text.configure(state="disabled")
        self.step_buttons[3].configure(state="normal")
        self.step_buttons[4].configure(state="normal")
        self.forecast_btn.configure(state="normal")
        self.model_selector.configure(state="normal")
        names = list(self.comparison_results.keys())
        self.model_selector.configure(values=names)
        if names:
            self.model_selector.set(names[0])
        messagebox.showinfo("Успех", f"Сравнено {len(self.comparison_results)} моделей")
        self.go_to_next_step()

    def _on_comparison_error(self, msg):
        self.update_status("Ошибка сравнения")
        self.compare_btn.configure(state="normal")
        messagebox.showerror("Ошибка", msg)

    def save_experiment(self):
        """Сохраняет результаты эксперимента в results/<дата>_<время>/"""
        try:
            ensure_results_dir()
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            exp_dir = os.path.join(RESULTS_DIR, ts)
            os.makedirs(exp_dir, exist_ok=True)
            logger.info(f"Сохранение в {exp_dir}")

            if self.oso_data is not None:
                self.oso_data.to_csv(os.path.join(exp_dir, "data_oso.csv"),
                                     index=False, encoding='utf-8')

            if self.comparator and self.comparison_results:
                self.comparator.save_results(os.path.join(exp_dir, "comparison"))
                try:
                    self.comparator.plot_comparison(os.path.join(exp_dir, "comparison"))
                except Exception as e:
                    logger.warning(f"Графики не сохранены: {e}")

            if self.forecast is not None:
                forecast_df = pd.DataFrame({'month': range(1, len(self.forecast) + 1),
                                            'forecast': self.forecast})
                forecast_df.to_csv(os.path.join(exp_dir, "forecast.csv"),
                                   index=False, encoding='utf-8')

            info = {
                'date': datetime.now().isoformat(),
                'data_file': self.current_data_file,
                'density_file': self.current_density_file,
                'records': len(self.oso_data) if self.oso_data is not None else 0,
                'metrics': self.model.metrics if self.model.is_trained else {},
            }
            with open(os.path.join(exp_dir, "info.json"), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

            self.update_status(f"Сохранено в {exp_dir}")
            messagebox.showinfo("Успех", f"Эксперимент сохранён в:\n{exp_dir}")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    @log_function_call
    def run_forecast(self):
        if not self.model.is_trained:
            messagebox.showwarning("Внимание", "Обучите модель!")
            return
        try:
            p = int(self.forecast_period.get())
            if p < 1 or p > 120:
                messagebox.showerror("Ошибка", "Период от 1 до 120")
                return
            self.update_status(f"Прогноз на {p} мес...")
            self.forecast = self.model.forecast(p)
            self._on_forecast_done(p)
        except ValueError:
            messagebox.showerror("Ошибка", "Введите число!")
        except Exception as e:
            logger.error(f"Ошибка прогноза: {e}")
            messagebox.showerror("Ошибка", str(e))

    def _on_forecast_done(self, periods):
        self.update_status(f"Прогноз на {periods} мес. выполнен")
        t = f"ПРОГНОЗ НА {periods} МЕСЯЦЕВ\n\n"
        for i, v in enumerate(self.forecast[:min(12, periods)], 1):
            mark = "высокий" if v > 305 else ("норма" if v > 295 else "низкий")
            t += f"Месяц {i:2d}: {v:6.1f} е.Д. ({mark})\n"
        if periods > 12:
            t += f"... и ещё {periods - 12} месяцев\n"
        t += (f"\nСТАТИСТИКА:\n"
              f"Среднее: {np.mean(self.forecast):.1f}\n"
              f"Минимум: {np.min(self.forecast):.1f}\n"
              f"Максимум: {np.max(self.forecast):.1f}")
        self.forecast_results.configure(state="normal")
        self.forecast_results.delete("1.0", "end")
        self.forecast_results.insert("1.0", t)
        self.forecast_results.configure(state="disabled")
        self.step_buttons[4].configure(state="normal")
        self.show_forecast_plot()
        messagebox.showinfo("Успех", f"Прогноз на {periods} мес. выполнен!")
        self.go_to_next_step()

    def show_welcome_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        x = np.linspace(0, 10, 100)
        y = 300 + 20 * np.sin(x)
        ax.plot(x, y, 'cyan', linewidth=2)
        ax.set_title('Система прогнозирования ОСО')
        ax.set_xlabel('Время')
        ax.set_ylabel('ОСО')
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_historical(self):
        if self.oso_data is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        dates = self.get_dates(self.oso_data)
        values = self.oso_data['oso'].values
        ax.plot(dates, values, 'tab:blue', alpha=0.7, linewidth=1, label='ОСО')
        if len(values) > 12:
            rm = pd.Series(values).rolling(12).mean()
            ax.plot(dates[11:], rm[11:], 'tab:orange', linewidth=2, label='Скользящее среднее')
        ax.set_title(f'Исторические данные ({dates.min().year}-{dates.max().year})')
        ax.set_xlabel('Год')
        ax.set_ylabel('ОСО')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_seasonality(self):
        if self.oso_data is None or 'year' not in self.oso_data.columns:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        seasonal = []
        for y in sorted(self.oso_data['year'].unique()):
            d = self.oso_data[self.oso_data['year'] == y]
            if len(d) == 12:
                seasonal.append(d['oso'].values)
        if not seasonal:
            messagebox.showinfo("Инфо", "Нет полных лет в данных")
            return
        avg = np.mean(seasonal, axis=0)
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        ax.plot(months, avg, 'tab:green', linewidth=3, marker='o', label='Сезонность')
        ax.fill_between(months, avg - 5, avg + 5, alpha=0.2, color='tab:green')
        ax.set_title('Сезонность ОСО')
        ax.set_xlabel('Месяц')
        ax.set_ylabel('ОСО')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_trends(self):
        if self.oso_data is None or 'year' not in self.oso_data.columns:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        yearly = self.oso_data.groupby('year')['oso'].mean()
        ax.plot(yearly.index, yearly.values, 'tab:orange', linewidth=2, marker='o', markersize=3, label='Среднегодовые')
        z = np.polyfit(yearly.index, yearly.values, 1)
        ax.plot(yearly.index, np.poly1d(z)(yearly.index), 'tab:red', linewidth=2,
                label=f'Тренд: {z[0]:.3f}/год')
        ax.set_title('Многолетние тренды')
        ax.set_xlabel('Год')
        ax.set_ylabel('ОСО')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_forecast_plot(self):
        if self.oso_data is None or self.forecast is None:
            messagebox.showinfo("Инфо", "Сначала выполните прогноз")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        dates = self.get_dates(self.oso_data)
        tail = min(24, len(self.oso_data))
        d_hist = dates.tail(tail)
        v_hist = self.oso_data['oso'].values[-tail:]
        last = d_hist.iloc[-1]
        d_fc = pd.date_range(last + pd.DateOffset(months=1), periods=len(self.forecast), freq='ME')
        ax.plot(d_hist, v_hist, 'tab:blue', linewidth=2, label='История')
        ax.plot(d_fc, self.forecast, 'tab:red', linewidth=2, label='Прогноз')
        ax.fill_between(d_fc, self.forecast - 3, self.forecast + 3, alpha=0.2, color='tab:red')
        ax.axhline(305, color='green', linestyle='--', alpha=0.5, label='Верхняя граница')
        ax.axhline(290, color='orange', linestyle='--', alpha=0.5, label='Нижняя граница')
        ax.set_title('Прогноз ОСО')
        ax.set_xlabel('Дата')
        ax.set_ylabel('ОСО')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg, fontsize=9)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_residuals_plot(self):
        if not self.model.is_trained:
            messagebox.showinfo("Инфо", "Обучите модель")
            return
        h = self.model.get_training_history()
        if h is None:
            messagebox.showinfo("Инфо", "История недоступна")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        epochs = range(1, len(h['loss']) + 1)
        res = np.array(h['loss']) - np.array(h.get('val_loss', h['loss']))
        ax.plot(epochs, res, 'tab:purple', linewidth=2, label='Разница loss')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title('Невязка (loss - val_loss)')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Разница')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_loss_plot(self):
        if not self.model.is_trained:
            messagebox.showinfo("Инфо", "Обучите модель")
            return
        h = self.model.get_training_history()
        if h is None:
            messagebox.showinfo("Инфо", "История недоступна")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        epochs = range(1, len(h['loss']) + 1)
        ax.plot(epochs, h['loss'], 'tab:blue', linewidth=2, label='Loss')
        if 'val_loss' in h:
            ax.plot(epochs, h['val_loss'], 'tab:orange', linewidth=2, label='Val Loss')
        ax.set_title('Кривая обучения')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('MSE')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_comparison_plot(self):
        if not self.comparator or not self.comparison_results:
            messagebox.showinfo("Инфо", "Запустите сравнение")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        models = list(self.comparison_results.keys())
        mae = [self.comparison_results[m]['metrics']['MAE'] for m in models]
        ax.bar(np.arange(len(models)), mae, color='tab:blue', alpha=0.8)
        ax.set_xlabel('Архитектура')
        ax.set_ylabel('MAE')
        ax.set_title('Сравнение архитектур')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        self.figure.tight_layout()
        self.canvas.draw()


def main():
    try:
        logger.info("Запуск приложения")
        app = ModernOzoneApp()
        app.mainloop()
        logger.info("Приложение завершено")
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}\n{traceback.format_exc()}")
        messagebox.showerror("Ошибка", f"{e}\n\nПодробности в логе")


if __name__ == "__main__":
    main()