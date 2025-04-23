import tkinter as tk
from tkinter import ttk

class SalesForecastApp:
    """Основной класс приложения для прогнозирования продаж"""
    
    def __init__(self, root):
        """Инициализация оконного приложения"""
        self.root = root
        self.root.title("Sales Forecasting Application")
        self.root.geometry("800x600")
        
        # Настройка основного окна
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Добавление надписи  с приветсвием
        ttk.Label(
            self.main_frame, 
            text="Welcome to the Sales Forecasting Application",
            font=("Arial", 16)
        ).pack(pady=20)
        
        # Добавление кнопки
        ttk.Button(
            self.main_frame,
            text="Exit",
            command=self.root.destroy
        ).pack(pady=10) 