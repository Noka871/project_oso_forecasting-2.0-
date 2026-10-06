import customtkinter as ctk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ozone_model import OzoneHybridModel
import threading
import os
import json
import traceback
from datetime import datetime
from utils.data_loader import OzoneDataLoader
from utils.logger import logger, log_function_call
from experiments.model_comparison import ModelComparator

# ---------- НАСТРОЙКИ ТЕМЫ ----------
ctk.set_appearance_mode("Light")       # по умолчанию светлая
ctk.set_default_color_theme("blue")

# FS25-стиль (светлая тема)
BG_LIGHT    = "#FFFFFF"
PANEL_LIGHT = "#F5F5F5"
BORDER      = "#E0E0E0"
ACCENT      = "#1F4E79"
ACCENT_HOV  = "#2A6BA8"
SUCCESS     = "#2E8B57"
SUCCESS_HOV = "#3CB371"
DANGER      = "#C0392B"
DANGER_HOV  = "#E74C3C"
WARN        = "#E67E22"
TEXT_DARK   = "#202020"
TEXT_GRAY   = "#606060"

# Тёмная тема
BG_DARK     = "#2b2b2b"
PANEL_DARK  = "#333333"
TEXT_LIGHT  = "#FFFFFF"

ABOUT_FILE = "about_content.json"
RESULTS_DIR = "results"

DEFAULT_ABOUT = {
    "about": "OSO Forecasting v2.6\n\nПрограмма для прогнозирования общего содержания озона (ОСО).",
    "help": "РУКОВОДСТВО\n\n1. Загрузите данные (TXT/CSV/DAT/XLSX/JSON/Parquet)\n2. Обучите модель\n3. Сравните модели\n4. Выполните прогноз\n5. Визуализация",
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


# ---------- КАСТОМНЫЙ ПРОГРЕСС-БАР «ГУСЕНИЦА» ----------
class SegmentedProgress(ctk.CTkFrame):
    """Сегментированный прогресс-бар в стиле 'гусеница'."""
    def __init__(self, master, segments=12, height=14, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.segments = segments
        self.bars = []
        self._pulse_index = 0
        self._pulse_job = None
        for i in range(segments):
            bar = ctk.CTkFrame(self, height=height, corner_radius=6,
                               fg_color=BORDER, border_width=0)
            bar.pack(side="left", padx=2, expand=True, fill="x")
            self.bars.append(bar)
        self._filled = 0

    def set(self, fraction):
        """fraction: 0.0–1.0"""
        fraction = max(0.0, min(1.0, fraction))
        self._filled = int(round(fraction * self.segments))
        for i, bar in enumerate(self.bars):
            if i < self._filled:
                bar.configure(fg_color=ACCENT)
            else:
                bar.configure(fg_color=BORDER)

    def start_pulse(self):
        """Пульсация активного сегмента."""
        self._stop_pulse()
        self._pulse()

    def _pulse(self):
        idx = self._filled if self._filled < self.segments else self.segments - 1
        for i, bar in enumerate(self.bars):
            if i == idx and i >= self._filled:
                bar.configure(fg_color="#4FC3F7")
            elif i < self._filled:
                bar.configure(fg_color=ACCENT)
            else:
                bar.configure(fg_color=BORDER)
        self._pulse_job = self.after(350, self._pulse)

    def _stop_pulse(self):
        if self._pulse_job:
            try:
                self.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None

    def reset(self):
        self._stop_pulse()
        self.set(0)


# ---------- ГЛАВНОЕ ПРИЛОЖЕНИЕ ----------
class ModernOzoneApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        logger.info("Запуск приложения")
        ensure_results_dir()

        self.title_font = ctk.CTkFont(family="Arial", size=20, weight="bold")
        self.subtitle_font = ctk.CTkFont(family="Arial", size=14, weight="bold")
        self.normal_font = ctk.CTkFont(family="Arial", size=12)
        self.small_font = ctk.CTkFont(family="Arial", size=10)

        self.title("OSO Forecasting — Прогнозирование озонового слоя")
        self.geometry("1400x950")
        self.minsize(1200, 800)

        self.data_loader = OzoneDataLoader()
        self.model = OzoneHybridModel()
        self.comparator = None
        self.oso_data = None
        self.forecast = None
        self.comparison_results = None
        self.current_step = 0
        self.current_theme = "Light"
        self.about_content = load_about_content()
        self.current_data_file = None
        self.current_density_file = None

        self.create_sidebar()
        self.create_main_content()
        self.create_status_bar()
        logger.info("Интерфейс инициализирован")

    # ---------- ЦВЕТА И ТЕМА ----------
    def get_colors(self):
        if self.current_theme == "Dark":
            return BG_DARK, TEXT_LIGHT
        return BG_LIGHT, TEXT_DARK

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
        if self.current_theme == "Light":
            self.current_theme = "Dark"
            ctk.set_appearance_mode("Dark")
            self.theme_btn.configure(text="☀️ Светлая тема")
        else:
            self.current_theme = "Light"
            ctk.set_appearance_mode("Light")
            self.theme_btn.configure(text="🌙 Тёмная тема")
        try:
            self.show_historical()
        except Exception:
            pass

    # ---------- ВСПЛЫВАЮЩИЕ ОКНА ----------
    def show_toast(self, text, kind="success", auto_close_ms=0):
        """Своё всплывающее окно вместо messagebox."""
        win = ctk.CTkToplevel(self)
        win.title("")
        win.geometry("360x170")
        win.resizable(False, False)
        win.grab_set()
        win.attributes("-topmost", True)

        if kind == "success":
            icon, color = "✅", SUCCESS
        elif kind == "error":
            icon, color = "❌", DANGER
        elif kind == "warning":
            icon, color = "⚠️", WARN
        else:
            icon, color = "ℹ️", ACCENT

        ctk.CTkLabel(win, text=icon, font=("Arial", 36)).pack(pady=(18, 4))
        ctk.CTkLabel(win, text=text, font=("Arial", 12), wraplength=320,
                     justify="center").pack(pady=4, padx=15)

        def close():
            try:
                win.destroy()
            except Exception:
                pass

        ctk.CTkButton(win, text="ОК", command=close,
                      fg_color=color, hover_color=ACCENT_HOV,
                      width=120, height=34).pack(pady=12)
        if auto_close_ms:
            win.after(auto_close_ms, close)
        win.after(100, win.lift)

    # ---------- САЙДБАР ----------
    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0,
                                     fg_color=(PANEL_LIGHT, PANEL_DARK),
                                     border_width=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(self.sidebar, text="🌍 OSO Forecasting",
                     font=self.title_font).pack(pady=(30, 6), padx=20)
        ctk.CTkLabel(self.sidebar, text="Прогнозирование и анализ",
                     font=self.small_font, text_color=TEXT_GRAY).pack(pady=(0, 15))

        self.theme_btn = ctk.CTkButton(
            self.sidebar, text="🌙 Тёмная тема", command=self.toggle_theme,
            font=self.normal_font, height=36,
            fg_color=(BORDER, "#3a3a3a"),
            text_color=(TEXT_DARK, TEXT_LIGHT),
            hover_color=("#d0d0d0", "#4a4a4a"),
            corner_radius=8,
        )
        self.theme_btn.pack(pady=(0, 18), padx=20, fill="x")

        steps_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        steps_frame.pack(fill="x", padx=20, pady=10)

        steps = [
            "📊 1. Загрузить данные",
            "🧠 2. Обучить модель",
            "🔬 3. Сравнить модели",
            "🔮 4. Выполнить прогноз",
            "📈 5. Визуализация",
        ]
        self.step_buttons = []
        for i, s in enumerate(steps):
            btn = ctk.CTkButton(
                steps_frame, text=s, font=self.normal_font, height=44,
                anchor="w", corner_radius=8,
                fg_color=(PANEL_LIGHT, PANEL_DARK),
                text_color=(TEXT_DARK, TEXT_LIGHT),
                hover_color=("#E8E8E8", "#444444"),
                command=lambda idx=i: self.set_current_step(idx),
                state="disabled" if i > 0 else "normal",
            )
            btn.pack(fill="x", pady=3)
            self.step_buttons.append(btn)
        self.step_buttons[0].configure(fg_color=SUCCESS, text_color="#FFFFFF")

        self.about_btn = ctk.CTkButton(
            self.sidebar, text="ℹ️ О программе",
            command=self.show_about_window,
            font=self.normal_font, height=40, corner_radius=8,
            fg_color=(BORDER, "#404040"),
            text_color=(TEXT_DARK, TEXT_LIGHT),
            hover_color=("#d0d0d0", "#505050"),
        )
        self.about_btn.pack(side="bottom", padx=20, pady=(0, 6), fill="x")
        ctk.CTkLabel(self.sidebar, text="Версия 2.6",
                     font=ctk.CTkFont(size=9), text_color=TEXT_GRAY).pack(side="bottom", pady=(0, 6))

    # ---------- ГЛАВНЫЙ КОНТЕНТ ----------
    def create_main_content(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=10,
                                        fg_color=(BG_LIGHT, BG_DARK),
                                        border_width=1, border_color=BORDER)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=18, pady=18)

        self.tabview = ctk.CTkTabview(self.main_frame, corner_radius=8)
        self.tabview.pack(fill="both", expand=True, padx=6, pady=6)

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

    # ---------- ВКЛАДКА «ДАННЫЕ» ----------
    def setup_data_tab(self):
        ctk.CTkLabel(self.tab_data, text="Загрузка и анализ данных",
                     font=self.title_font).pack(pady=18)
        bf = ctk.CTkFrame(self.tab_data, fg_color="transparent")
        bf.pack(pady=10)
        ctk.CTkButton(bf, text="📥 Загрузить демо-данные",
                      command=self.load_demo_data,
                      font=self.normal_font, height=42, width=220,
                      fg_color=ACCENT, hover_color=ACCENT_HOV,
                      corner_radius=8).pack(side="left", padx=6)
        ctk.CTkButton(bf, text="📁 Загрузить из файла",
                      command=self.load_from_file,
                      font=self.normal_font, height=42, width=220,
                      fg_color=SUCCESS, hover_color=SUCCESS_HOV,
                      corner_radius=8).pack(side="left", padx=6)

        hint = (
            "Поддерживаемые форматы:\n"
            "• TXT / DAT: строки «номер  дата  значение»\n"
            "• CSV: колонки year, month, oso\n"
            "• XLSX / XLS: Excel-таблицы\n"
            "• JSON: массив объектов или dict\n"
            "• Parquet / Feather: колоночные форматы\n"
            "Имя файла — любое"
        )
        ctk.CTkLabel(self.tab_data, text=hint, font=self.small_font,
                     justify="left", text_color=TEXT_GRAY).pack(pady=6)

        self.data_info_frame = ctk.CTkFrame(self.tab_data, corner_radius=8)
        self.data_info_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.data_info_text = ctk.CTkTextbox(self.data_info_frame, height=200,
                                              corner_radius=8)
        self.data_info_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.data_info_text.insert("1.0", "Данные не загружены.")
        self.data_info_text.configure(state="disabled")

    # ---------- ВКЛАДКА «МОДЕЛЬ» ----------
    def setup_model_tab(self):
        ctk.CTkLabel(self.tab_model, text="Обучение гибридной модели",
                     font=self.title_font).pack(pady=18)
        af = ctk.CTkFrame(self.tab_model, corner_radius=8)
        af.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(af, text="Архитектура Conv1D + LSTM:",
                     font=self.subtitle_font).pack(pady=(10, 5))
        txt = ("Conv1D: 64 фильтра, ядро=3, ReLU\n"
               "LSTM: 128 нейронов\nDense: 64 → 32\nDropout: 0.3\n"
               "Adam (lr=0.001), MSE")
        ctk.CTkLabel(af, text=txt, font=self.normal_font,
                     justify="left").pack(pady=(5, 10), padx=15)

        bf = ctk.CTkFrame(self.tab_model, fg_color="transparent")
        bf.pack(pady=18)
        self.train_btn = ctk.CTkButton(
            bf, text="🚀 Начать обучение", command=self.train_model,
            font=self.normal_font, height=50, width=220,
            fg_color=SUCCESS, hover_color=SUCCESS_HOV,
            corner_radius=10, state="disabled")
        self.train_btn.pack(side="left", padx=6)
        self.stop_btn = ctk.CTkButton(
            bf, text="⏹ Остановить", command=self.stop_training,
            font=self.normal_font, height=50, width=160,
            fg_color=DANGER, hover_color=DANGER_HOV,
            corner_radius=10, state="disabled")
        self.stop_btn.pack(side="left", padx=6)

        # --- ГУСЕНИЦА ---
        self.progress_bar = SegmentedProgress(self.tab_model, segments=12, height=16)
        self.progress_bar.pack(fill="x", padx=50, pady=14)
        self.progress_label = ctk.CTkLabel(self.tab_model,
                                            text="Готов к обучению",
                                            font=self.small_font)
        self.progress_label.pack(pady=4)

        rf = ctk.CTkFrame(self.tab_model, corner_radius=8)
        rf.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(rf, text="Результаты:", font=self.subtitle_font).pack(anchor="w", pady=(6, 4), padx=6)
        self.training_results = ctk.CTkTextbox(rf, height=150, corner_radius=8)
        self.training_results.pack(fill="both", expand=True, padx=6, pady=6)
        self.training_results.insert("1.0", "Результаты обучения появятся здесь")
        self.training_results.configure(state="disabled")

    # ---------- ВКЛАДКА «ЭКСПЕРИМЕНТЫ» ----------
    def setup_experiments_tab(self):
        ctk.CTkLabel(self.tab_experiments, text="Сравнение архитектур",
                     font=self.title_font).pack(pady=18)
        bf = ctk.CTkFrame(self.tab_experiments, fg_color="transparent")
        bf.pack(pady=10)
        self.compare_btn = ctk.CTkButton(
            bf, text="🔬 Запустить сравнение", command=self.run_comparison,
            font=self.normal_font, height=50, width=260,
            fg_color=ACCENT, hover_color=ACCENT_HOV,
            corner_radius=10, state="disabled")
        self.compare_btn.pack(pady=5)
        self.save_results_btn = ctk.CTkButton(
            bf, text="💾 Сохранить эксперимент", command=self.save_experiment,
            font=self.normal_font, height=42, width=300,
            fg_color=SUCCESS, hover_color=SUCCESS_HOV,
            corner_radius=10, state="disabled")
        self.save_results_btn.pack(pady=5)

        rf = ctk.CTkFrame(self.tab_experiments, corner_radius=8)
        rf.pack(fill="both", expand=True, padx=20, pady=10)
        self.exp_tabview = ctk.CTkTabview(rf, corner_radius=8)
        self.exp_tabview.pack(fill="both", expand=True, padx=6, pady=6)
        self.exp_table_tab = self.exp_tabview.add("Таблица")
        self.exp_metrics_tab = self.exp_tabview.add("Метрики")
        self.exp_analysis_tab = self.exp_tabview.add("Анализ")
        for name, tab in [("Таблица", self.exp_table_tab),
                          ("Метрики", self.exp_metrics_tab),
                          ("Анализ", self.exp_analysis_tab)]:
            tb = ctk.CTkTextbox(tab, height=300, corner_radius=8)
            tb.pack(fill="both", expand=True, padx=8, pady=8)
            tb.insert("1.0", f"{name} появится после запуска")
            tb.configure(state="disabled")
            if name == "Таблица":
                self.comparison_text = tb
            elif name == "Метрики":
                self.metrics_text = tb
            else:
                self.analysis_text = tb
        self.exp_tabview.pack_forget()

    # ---------- ВКЛАДКА «ПРОГНОЗ» ----------
    def setup_forecast_tab(self):
        ctk.CTkLabel(self.tab_forecast, text="Прогнозирование ОСО",
                     font=self.title_font).pack(pady=18)
        sf = ctk.CTkFrame(self.tab_forecast, corner_radius=8)
        sf.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(sf, text="Настройки прогноза:",
                     font=self.subtitle_font).pack(pady=(10, 5))
        mf = ctk.CTkFrame(sf, fg_color="transparent")
        mf.pack(fill="x", pady=5, padx=15)
        ctk.CTkLabel(mf, text="Модель:", font=self.normal_font).pack(side="left", padx=6)
        self.model_selector = ctk.CTkComboBox(
            mf, values=["CNN-LSTM (гибридная)"],
            font=self.normal_font, width=240,
            corner_radius=8, state="disabled")
        self.model_selector.pack(side="left", padx=6)

        pf = ctk.CTkFrame(sf, fg_color="transparent")
        pf.pack(fill="x", pady=5, padx=15)
        ctk.CTkLabel(pf, text="Период (месяцев):", font=self.normal_font).pack(side="left", padx=6)
        self.forecast_period = ctk.CTkEntry(pf, placeholder_text="12",
                                             font=self.normal_font, width=100,
                                             corner_radius=8)
        self.forecast_period.pack(side="left", padx=6)
        self.forecast_period.insert(0, "12")

        self.forecast_btn = ctk.CTkButton(
            self.tab_forecast, text="🔮 Выполнить прогноз",
            command=self.run_forecast, font=self.normal_font,
            height=50, width=260, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOV, state="disabled")
        self.forecast_btn.pack(pady=18)

        self.forecast_results = ctk.CTkTextbox(self.tab_forecast, height=250,
                                                corner_radius=8)
        self.forecast_results.pack(fill="both", expand=True, padx=20, pady=10)
        self.forecast_results.insert("1.0", "Результаты прогноза появятся здесь")
        self.forecast_results.configure(state="disabled")

    # ---------- ВКЛАДКА «ВИЗУАЛИЗАЦИЯ» ----------
    def setup_visualization_tab(self):
        ctk.CTkLabel(self.tab_visualization, text="Визуализация",
                     font=self.title_font).pack(pady=8)
        cf = ctk.CTkFrame(self.tab_visualization, fg_color="transparent")
        cf.pack(fill="x", padx=20, pady=8)
        buttons = [
            ("📉 Исторические", self.show_historical),
            ("🌱 Сезонность", self.show_seasonality),
            ("📊 Тренды", self.show_trends),
            ("🔮 Прогноз", self.show_forecast_plot),
            ("📐 Невязка", self.show_residuals_plot),
            ("📈 Кривая обучения", self.show_loss_plot),
            ("🔬 Сравнение", self.show_comparison_plot),
        ]
        row = None
        for i, (t, cmd) in enumerate(buttons):
            if i % 4 == 0:
                row = ctk.CTkFrame(cf, fg_color="transparent")
                row.pack(pady=5)
            state = "normal" if t != "🔬 Сравнение" else "disabled"
            ctk.CTkButton(row, text=t, command=cmd, font=self.small_font,
                          width=170, height=38, corner_radius=8,
                          fg_color=ACCENT, hover_color=ACCENT_HOV,
                          state=state).pack(side="left", padx=3)

        self.viz_frame = ctk.CTkFrame(self.tab_visualization, corner_radius=8)
        self.viz_frame.pack(fill="both", expand=True, padx=20, pady=10)
        bg, _ = self.get_colors()
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor=bg)
        self.canvas = FigureCanvasTkAgg(self.figure, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self.show_welcome_plot()

    # ---------- СТАТУС-БАР ----------
    def create_status_bar(self):
        self.status_bar = ctk.CTkFrame(self, height=32, corner_radius=0,
                                        fg_color=(PANEL_LIGHT, PANEL_DARK))
        self.status_bar.pack(side="bottom", fill="x")
        self.status_bar.pack_propagate(False)
        self.status_label = ctk.CTkLabel(self.status_bar,
                                          text="✓ Готов к работе",
                                          font=self.small_font)
        self.status_label.pack(side="left", padx=12, pady=6)

    # ---------- ЛОГИКА ----------
    def set_current_step(self, idx):
        self.current_step = idx
        for i, btn in enumerate(self.step_buttons):
            if i == idx:
                btn.configure(fg_color=SUCCESS, text_color="#FFFFFF")
            else:
                btn.configure(fg_color=(PANEL_LIGHT, PANEL_DARK),
                              text_color=(TEXT_DARK, TEXT_LIGHT))
        tabs = ["Данные", "Модель", "Эксперименты", "Прогноз", "Визуализация"]
        self.tabview.set(tabs[idx])

    def go_to_next_step(self, delay_ms=1000):
        if self.current_step >= len(self.step_buttons) - 1:
            return
        next_idx = self.current_step + 1
        self.after(delay_ms, lambda: self.set_current_step(next_idx))

    def update_status(self, msg):
        self.status_label.configure(text=f"⏳ {msg}")
        self.update()

    def show_about_window(self):
        win = ctk.CTkToplevel(self)
        win.title("О программе OSO Forecasting")
        win.geometry("800x600")
        win.minsize(600, 450)
        win.grab_set()
        ctk.CTkLabel(win, text="ℹ️ О программе", font=self.title_font).pack(pady=10)
        tabs = ctk.CTkTabview(win, corner_radius=8)
        tabs.pack(fill="both", expand=True, padx=20, pady=10)
        tab_about = tabs.add("О программе")
        tab_help = tabs.add("Помощь")
        tab_license = tabs.add("Лицензия")
        text_widgets = {}
        for name, key, tab in [
            ("О программе", "about", tab_about),
            ("Помощь", "help", tab_help),
            ("Лицензия", "license", tab_license),
        ]:
            tb = ctk.CTkTextbox(tab, wrap="word", font=self.normal_font,
                                 corner_radius=8)
            tb.pack(fill="both", expand=True, padx=10, pady=10)
            tb.insert("1.0", self.about_content.get(key, ""))
            tb.configure(state="disabled")
            text_widgets[key] = tb
            ctk.CTkButton(tab, text="✏️ Редактировать",
                          command=lambda k=key, w=text_widgets: self.edit_about_content(k, w),
                          font=self.normal_font, height=36, width=180,
                          corner_radius=8, fg_color=ACCENT, hover_color=ACCENT_HOV
                          ).pack(pady=6)
        ctk.CTkButton(win, text="Закрыть", command=win.destroy,
                      font=self.normal_font, height=40, width=150,
                      fg_color=DANGER, hover_color=DANGER_HOV,
                      corner_radius=8).pack(pady=10)
        win.after(100, win.lift)

    def edit_about_content(self, key, text_widgets):
        win = ctk.CTkToplevel(self)
        win.title("Редактор")
        win.geometry("800x600")
        win.grab_set()
        ctk.CTkLabel(win, text="Редактор содержимого",
                     font=self.subtitle_font).pack(pady=10)
        editor = ctk.CTkTextbox(win, wrap="word", font=self.normal_font,
                                 corner_radius=8)
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
            self.show_toast("Текст сохранён в about_content.json", "success")
            win.destroy()

        bf = ctk.CTkFrame(win, fg_color="transparent")
        bf.pack(pady=10)
        ctk.CTkButton(bf, text="💾 Сохранить", command=save_and_close,
                      font=self.normal_font, height=40, width=150,
                      fg_color=SUCCESS, hover_color=SUCCESS_HOV,
                      corner_radius=8).pack(side="left", padx=6)
        ctk.CTkButton(bf, text="Отмена", command=win.destroy,
                      font=self.normal_font, height=40, width=150,
                      fg_color=DANGER, hover_color=DANGER_HOV,
                      corner_radius=8).pack(side="left", padx=6)

    # ---------- ЗАГРУЗКА ДАННЫХ ----------
    def get_dates(self, data):
        if 'year' in data.columns and 'month' in data.columns:
            try:
                return pd.to_datetime(data[['year', 'month']].assign(day=1))
            except Exception:
                pass
        return pd.date_range('2020-01-01', periods=len(data), freq='MS')

    def _is_real_format(self, path):
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

    def _load_any_file(self, path):
        """Универсальная загрузка: TXT/DAT/CSV/XLSX/XLS/JSON/Parquet/Feather."""
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.txt', '.dat'):
            # пробуем разные разделители
            for sep in [r'\s+', ',', ';', '\t']:
                try:
                    df = pd.read_csv(path, sep=sep, engine='python')
                    if df.shape[1] >= 2:
                        return df
                except Exception:
                    continue
            return pd.read_fwf(path)
        if ext == '.csv':
            return pd.read_csv(path)
        if ext in ('.xlsx', '.xls'):
            return pd.read_excel(path)
        if ext == '.json':
            return pd.read_json(path)
        if ext == '.parquet':
            return pd.read_parquet(path)
        if ext == '.feather':
            return pd.read_feather(path)
        # fallback
        return pd.read_csv(path)

    def load_from_file(self):
        path = filedialog.askopenfilename(
            title="Выберите файл с данными ОСО",
            filetypes=[
                ("Все поддерживаемые", "*.txt *.dat *.csv *.xlsx *.xls *.json *.parquet *.feather"),
                ("TXT", "*.txt"), ("DAT", "*.dat"), ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("JSON", "*.json"),
                ("Parquet", "*.parquet"),
                ("Feather", "*.feather"),
                ("Все файлы", "*.*"),
            ]
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
                    filetypes=[("Все", "*.*")]
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
                self.oso_data = self._load_any_file(path)
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
            self.show_toast(f"Не удалось загрузить:\n{e}", "error")

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
        self.show_toast("Данные загружены!", "success")
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
            self.after(0, lambda: self.show_toast(str(e), "error"))

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
        self.show_toast("Демо-данные созданы!", "success")
        self.go_to_next_step()

    # ---------- ОБУЧЕНИЕ ----------
    @log_function_call
    def train_model(self):
        if self.oso_data is None:
            self.show_toast("Загрузите данные!", "warning")
            return
        self.update_status("Обучение модели...")
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.reset()
        self.progress_bar.start_pulse()
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
        self.progress_label.configure(
            text=f"Эпоха {e}/{t} | loss: {l:.4f} | val_loss: {v:.4f}")

    def stop_training(self):
        self.model.stop_training()
        self.progress_label.configure(text="Остановка...")

    def _on_trained(self):
        self.update_status("Модель обучена")
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_bar.set(1)
        self.progress_bar._stop_pulse()
        self.progress_label.configure(text="✓ Обучение завершено")
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
        self.show_toast("Модель обучена!", "success")
        self.go_to_next_step()

    def _on_train_error(self, msg):
        self.update_status("Ошибка обучения")
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_bar._stop_pulse()
        self.show_toast(msg, "error")

    # ---------- СРАВНЕНИЕ ----------
    @log_function_call
    def run_comparison(self):
        if self.oso_data is None:
            self.show_toast("Загрузите данные!", "warning")
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
                f"{row['Архитектура']:<22}{row['MAE']:>8}{row['RMSE']:>8}"
                f"{row['R2']:>8}{row['Время']:>8}{row['Параметры']:>10}"
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
        self.show_toast(f"Сравнено {len(self.comparison_results)} моделей", "success")
        self.go_to_next_step()

    def _on_comparison_error(self, msg):
        self.update_status("Ошибка сравнения")
        self.compare_btn.configure(state="normal")
        self.show_toast(msg, "error")

    # ---------- СОХРАНЕНИЕ ЭКСПЕРИМЕНТА ----------
    def save_experiment(self):
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
            self.show_toast(f"Эксперимент сохранён в:\n{exp_dir}", "success")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            self.show_toast(f"Не удалось сохранить:\n{e}", "error")

    # ---------- ПРОГНОЗ ----------
    @log_function_call
    def run_forecast(self):
        if not self.model.is_trained:
            self.show_toast("Обучите модель!", "warning")
            return
        try:
            p = int(self.forecast_period.get())
            if p < 1 or p > 120:
                self.show_toast("Период от 1 до 120", "error")
                return
            self.update_status(f"Прогноз на {p} мес...")
            self.forecast = self.model.forecast(p)
            self._on_forecast_done(p)
        except ValueError:
            self.show_toast("Введите число!", "error")
        except Exception as e:
            logger.error(f"Ошибка прогноза: {e}")
            self.show_toast(str(e), "error")

    def _on_forecast_done(self, periods):
        self.update_status(f"Прогноз на {periods} мес. выполнен")
        # границы по статистике истории
        if self.oso_data is not None:
            mean_val = float(np.mean(self.oso_data['oso'].values))
            std_val = float(np.std(self.oso_data['oso'].values))
        else:
            mean_val, std_val = 300.0, 15.0
        high_th = mean_val + std_val
        low_th = mean_val - std_val
        t = f"ПРОГНОЗ НА {periods} МЕСЯЦЕВ\n\n"
        for i, v in enumerate(self.forecast[:min(12, periods)], 1):
            if v > high_th:
                mark = "высокий"
            elif v > low_th:
                mark = "норма"
            else:
                mark = "низкий"
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
        self.show_toast(f"Прогноз на {periods} мес. выполнен!", "success")
        self.go_to_next_step()

    # ---------- ГРАФИКИ ----------
    def show_welcome_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        x = np.linspace(0, 10, 100)
        y = 300 + 20 * np.sin(x)
        ax.plot(x, y, color='#4FC3F7', linewidth=2)
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
        ax.plot(dates, values, color='#4FC3F7', alpha=0.8, linewidth=1, label='ОСО')
        if len(values) > 12:
            rm = pd.Series(values).rolling(12).mean()
            ax.plot(dates[11:], rm[11:], color='#FF7043',
                    linewidth=2, label='Скользящее среднее (12 мес)')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        self.figure.autofmt_xdate(rotation=45)
        ax.set_title(f'Исторические данные ({dates.min().year}–{dates.max().year})')
        ax.set_xlabel('Год')
        ax.set_ylabel('ОСО, е.Д.')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_seasonality(self):
        if self.oso_data is None or 'month' not in self.oso_data.columns:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        seasonal = self.oso_data.groupby('month')['oso'].mean()
        std = self.oso_data.groupby('month')['oso'].std()
        months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                  'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
        x = np.arange(12)
        vals = seasonal.reindex(range(1, 13)).values
        ax.plot(x, vals, color='#66BB6A', linewidth=3, marker='o', label='Среднее по месяцам')
        ax.fill_between(x, vals - std.values, vals + std.values,
                        alpha=0.2, color='#66BB6A', label='±1 СКО')
        ax.set_xticks(x)
        ax.set_xticklabels(months)
        ax.set_title('Сезонность ОСО (среднее по всем годам)')
        ax.set_xlabel('Месяц')
        ax.set_ylabel('ОСО, е.Д.')
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
        ax.plot(yearly.index, yearly.values, color='#FF7043',
                linewidth=2, marker='o', markersize=3, label='Среднегодовые')
        z = np.polyfit(yearly.index, yearly.values, 1)
        ax.plot(yearly.index, np.poly1d(z)(yearly.index), color='#E53935',
                linewidth=2, label=f'Тренд: {z[0]:+.3f} е.Д./год')
        # ЦЕЛЫЕ ГОДЫ, ШАГ 2
        ymin, ymax = int(yearly.index.min()), int(yearly.index.max())
        ticks = list(range(ymin, ymax + 1, 2))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(y) for y in ticks], rotation=45)
        ax.set_title('Многолетний тренд ОСО')
        ax.set_xlabel('Год')
        ax.set_ylabel('ОСО, е.Д.')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_forecast_plot(self):
        if self.oso_data is None or self.forecast is None:
            self.show_toast("Сначала выполните прогноз", "info")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        dates = self.get_dates(self.oso_data)
        tail = min(24, len(self.oso_data))
        d_hist = dates.tail(tail)
        v_hist = self.oso_data['oso'].values[-tail:]
        last = d_hist.iloc[-1]
        d_fc = pd.date_range(last + pd.DateOffset(months=1),
                              periods=len(self.forecast), freq='MS')
        ax.plot(d_hist, v_hist, color='#4FC3F7', linewidth=2, label='История')
        ax.plot(d_fc, self.forecast, color='#E53935', linewidth=2, label='Прогноз')
        sigma = self.model.metrics.get('rmse', 3.0)
        ax.fill_between(d_fc, self.forecast - sigma, self.forecast + sigma,
                        alpha=0.2, color='#E53935',
                        label=f'±RMSE ({sigma:.1f})')
        # границы mean±std из истории
        if self.oso_data is not None:
            mean_val = float(np.mean(self.oso_data['oso'].values))
            std_val = float(np.std(self.oso_data['oso'].values))
            ax.axhline(mean_val + std_val, color='#2E8B57', linestyle='--',
                       alpha=0.5, label='+1 СКО')
            ax.axhline(mean_val - std_val, color='#E67E22', linestyle='--',
                       alpha=0.5, label='−1 СКО')
        # вертикальная линия на стыке
        ax.axvline(last, color='gray', linestyle=':', alpha=0.7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.figure.autofmt_xdate(rotation=45)
        ax.set_title('Прогноз ОСО')
        ax.set_xlabel('Дата')
        ax.set_ylabel('ОСО, е.Д.')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg,
                  labelcolor=fg, fontsize=9)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_residuals_plot(self):
        if not self.model.is_trained:
            self.show_toast("Обучите модель", "info")
            return
        h = self.model.get_training_history()
        if h is None:
            self.show_toast("История недоступна", "info")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        epochs = range(1, len(h['loss']) + 1)
        res = np.array(h['loss']) - np.array(h.get('val_loss', h['loss']))
        ax.plot(epochs, res, color='#AB47BC', linewidth=2, label='Разница loss')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title('Невязка (loss − val_loss)')
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Разница')
        _, fg = self.get_colors()
        ax.legend(facecolor=self.figure.get_facecolor(), edgecolor=fg, labelcolor=fg)
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()

    def show_loss_plot(self):
        if not self.model.is_trained:
            self.show_toast("Обучите модель", "info")
            return
        h = self.model.get_training_history()
        if h is None:
            self.show_toast("История недоступна", "info")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        epochs = range(1, len(h['loss']) + 1)
        ax.plot(epochs, h['loss'], color='#4FC3F7', linewidth=2, label='Loss')
        if 'val_loss' in h:
            ax.plot(epochs, h['val_loss'], color='#FF7043', linewidth=2, label='Val Loss')
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
            self.show_toast("Запустите сравнение", "info")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.apply_theme(ax)
        models = list(self.comparison_results.keys())
        mae = [self.comparison_results[m]['metrics']['MAE'] for m in models]
        ax.bar(np.arange(len(models)), mae, color='#4FC3F7', alpha=0.85)
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
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()