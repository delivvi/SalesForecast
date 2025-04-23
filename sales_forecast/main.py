#!/usr/bin/env python
"""
Приложение для анализа и прогнозирования продаж

Это приложение предоставляет инструменты для загрузки, анализа и прогнозирования данных о продажах
с использованием методов машинного обучения и визуализацией ключевых метрик.
"""

import os
import tkinter as tk
import sys

# Добавление родительского каталога в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.interface import SalesForecastApp

def main():
    """Основная функция для запуска приложения"""
    root = tk.Tk()
    app = SalesForecastApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 