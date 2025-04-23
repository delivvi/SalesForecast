import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from datetime import datetime, timedelta
import threading
import webbrowser
import traceback

# Добавить родительский каталог в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager
from models import SalesForecaster
from visualization import SalesDashboard

class Tooltip:
    """
    Создает всплывающую подсказку для заданного виджета
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Создать окно подсказки
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Создать содержимое подсказки
        frame = ttk.Frame(self.tooltip_window, borderwidth=1, relief="solid")
        frame.pack(ipadx=5, ipady=5)
        
        # Разделить длинный текст на несколько строк
        wrapped_text = self._wrap_text(self.text, 80)
        label = ttk.Label(frame, text=wrapped_text, justify="left", 
                         background="#ffffcc", relief="solid", borderwidth=0)
        label.pack(ipadx=3, ipady=3)
    
    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def _wrap_text(self, text, width):
        """Перенос текста на заданную ширину путем добавления новых строк"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + len(current_line) <= width:
                current_line.append(word)
                current_length += len(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)


class SalesForecastApp:
    def __init__(self, root):
        """Инициализация приложения для прогнозирования продаж"""
        self.root = root
        self.root.title("Sales Analysis & Forecasting")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Путь к базе данных
        self.db_path_var = tk.StringVar()
        
        # Создать менеджер базы данных с путем по умолчанию
        self.db_manager = DatabaseManager()
        self.db_path_var.set(self.db_manager.db_file_path)
        
        # Инициализировать прогнозировщик
        self.forecaster = SalesForecaster()
        
        # Инициализировать панель мониторинга
        self.dashboard = SalesDashboard()
        
        # Поток сервера панели мониторинга
        self.dashboard_thread = None
        self.dashboard_running = False
        
        # Создать пользовательский интерфейс
        self.create_ui()
        
        # Загрузить существующие данные и модель
        self.load_existing_data()
    
    def load_existing_data(self):
        """Загрузить существующие данные и модель при запуске приложения"""
        try:
            # Попытаться инициализировать базу данных, если она не существует
            self.db_manager.init_database()
            
            # Попытаться загрузить модель
            model_loaded = False
            try:
                model_loaded = self.forecaster.load_model()
            except Exception as model_error:
                print(f"Warning: Could not load existing model: {str(model_error)}")
                # Просто используйте свежий прогнозировщик
                model_loaded = False
            
            if model_loaded:
                # Если модель загружена успешно, получите данные и обновите панель мониторинга
                self.db_manager.open_session()
                sales_data = self.db_manager.get_sales_data()
                
                if sales_data:
                    # Преобразуйте в DataFrame
                    df = pd.DataFrame([
                        {
                            'sale_date': sale.sale_date,
                            'total_amount': sale.total_amount,
                            'product_id': sale.product_id
                        }
                        for sale in sales_data
                    ])
                    
                    # Сгенерируйте прогноз с текущей загруженной моделью
                    try:
                        forecast_days = self.forecast_days_var.get()
                        forecast_df = self.forecaster.predict(df, future_days=forecast_days)
                        
                        # Обновите данные панели мониторинга
                        feature_importance = self.forecaster.get_feature_importance()
                        self.dashboard.update_data(df, forecast_df, feature_importance)
                        
                        # Обновите пользовательский интерфейс
                        metrics = self.forecaster.metrics
                        if metrics:
                            self.mae_var.set(f"MAE: {metrics.get('mae', 0):.2f}")
                            self.mse_var.set(f"MSE: {metrics.get('mse', 0):.2f}")
                            self.rmse_var.set(f"RMSE: {metrics.get('rmse', 0):.2f}")
                            self.r2_var.set(f"R²: {metrics.get('r2', 0):.2f}")
                        
                        if feature_importance is not None:
                            # Очистите существующие данные
                            for i in self.feature_tree.get_children():
                                self.feature_tree.delete(i)
                            
                            # Добавьте новые данные
                            for _, row in feature_importance.iterrows():
                                self.feature_tree.insert('', 'end', values=(row['Feature'], f"{row['Importance']:.4f}"))
                    except Exception as forecast_error:
                        print(f"Warning: Could not generate forecast: {str(forecast_error)}")
                
                self.db_manager.close_session()
        except Exception as e:
            # Не удалось загрузить данные, но нет необходимости показывать пользователю ошибку при запуске
            print(f"Warning: Could not load existing data: {str(e)}")
    
    def create_ui(self):
        """Создать пользовательский интерфейс"""
        # Создайте ноутбук для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создайте вкладки
        self.data_tab = ttk.Frame(self.notebook)
        self.forecast_tab = ttk.Frame(self.notebook)
        self.dashboard_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Management")
        self.notebook.add(self.forecast_tab, text="Forecasting")
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        
        # Настройте каждую вкладку
        self.setup_data_tab()
        self.setup_forecast_tab()
        self.setup_dashboard_tab()
    
    def setup_data_tab(self):
        """Настройте вкладку управления данными"""
        # Фрейм для инициализации базы данных
        db_frame = ttk.LabelFrame(self.data_tab, text="Database")
        db_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Входной путь базы данных
        ttk.Label(db_frame, text="Database Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(db_frame, textvariable=self.db_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(db_frame, text="Browse", command=self.browse_db_path).grid(row=0, column=2, padx=5, pady=5)
        
        # Кнопка инициализации базы данных
        ttk.Button(db_frame, text="Initialize Database", command=self.initialize_database).grid(row=1, column=1, padx=5, pady=5)
        
        # Фрейм для импорта данных
        import_frame = ttk.LabelFrame(self.data_tab, text="Import Coffee Sales Data")
        import_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Конфигурация импорта данных
        ttk.Label(import_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(import_frame, textvariable=self.csv_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(import_frame, text="Browse", command=lambda: self.browse_file(self.csv_path_var)).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(import_frame, text="Import Coffee Sales", command=self.import_data).grid(row=1, column=1, padx=5, pady=10)
        
        # Фрейм для просмотра данных
        view_frame = ttk.LabelFrame(self.data_tab, text="Data View")
        view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создайте дерево для отображения данных
        self.tree = ttk.Treeview(view_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Добавьте полосу прокрутки
        scrollbar = ttk.Scrollbar(view_frame, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Кнопки для просмотра данных
        button_frame = ttk.Frame(self.data_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="View Coffee Products", command=self.view_coffee_products).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="View Coffee Sales", command=self.view_coffee_sales).pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_forecast_tab(self):
        """Настройте вкладку прогнозирования"""
        # Фрейм для параметров модели
        param_frame = ttk.LabelFrame(self.forecast_tab, text="Model Parameters")
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(param_frame, text="Forecast Days:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.forecast_days_var = tk.IntVar(value=30)
        ttk.Spinbox(param_frame, from_=7, to=90, increment=1, textvariable=self.forecast_days_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        forecast_days_help = ttk.Label(param_frame, text="?", width=2, anchor=tk.CENTER)
        forecast_days_help.grid(row=0, column=2, padx=5, pady=5)
        Tooltip(forecast_days_help, "Number of days to forecast into the future (7-90)")
        
        ttk.Label(param_frame, text="Test Size (%):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=20)
        ttk.Spinbox(param_frame, from_=10, to=40, increment=5, textvariable=self.test_size_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        test_size_help = ttk.Label(param_frame, text="?", width=2, anchor=tk.CENTER)
        test_size_help.grid(row=1, column=2, padx=5, pady=5)
        Tooltip(test_size_help, "Percentage of data to use for testing the model (10-40%)")
        
        ttk.Label(param_frame, text="Note:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Label(param_frame, text="Using Random Forest model for optimal results").grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Фрейм для обучения модели и прогнозирования
        model_frame = ttk.LabelFrame(self.forecast_tab, text="Model Training and Prediction")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(model_frame, text="Train Model", command=self.train_model).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(model_frame, text="Generate Forecast", command=self.generate_forecast).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(model_frame, text="Save Model", command=self.save_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5, pady=5)
        
        # Фрейм для параметров сохранения модели
        save_frame = ttk.LabelFrame(self.forecast_tab, text="Model Save Options")
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(save_frame, text="Custom Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_name_var = tk.StringVar()
        ttk.Entry(save_frame, textvariable=self.model_name_var, width=30).grid(row=0, column=1, padx=5, pady=5)
        
        # Фрейм для важности функции
        feature_frame = ttk.LabelFrame(self.forecast_tab, text="Feature Importance")
        feature_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Создайте дерево для отображения важности функции
        self.feature_tree = ttk.Treeview(feature_frame, columns=('Feature', 'Importance'), show='headings')
        self.feature_tree.heading('Feature', text='Feature')
        self.feature_tree.heading('Importance', text='Importance')
        self.feature_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Привяжите событие наведения для объяснения функций
        self.feature_tree.bind("<Motion>", self.show_feature_explanation)
        
        # Добавьте полосу прокрутки
        feature_scrollbar = ttk.Scrollbar(feature_frame, orient="vertical", command=self.feature_tree.yview)
        feature_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.feature_tree.configure(yscrollcommand=feature_scrollbar.set)
        
        # Фрейм для метрик
        metrics_frame = ttk.LabelFrame(self.forecast_tab, text="Model Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.mae_var = tk.StringVar(value="MAE: N/A")
        self.mse_var = tk.StringVar(value="MSE: N/A")
        self.rmse_var = tk.StringVar(value="RMSE: N/A")
        self.r2_var = tk.StringVar(value="R²: N/A")
        
        mae_label = ttk.Label(metrics_frame, textvariable=self.mae_var)
        mae_label.grid(row=0, column=0, padx=5, pady=5)
        Tooltip(mae_label, self.forecaster.get_metric_explanation('mae'))
        
        mse_label = ttk.Label(metrics_frame, textvariable=self.mse_var)
        mse_label.grid(row=0, column=1, padx=5, pady=5)
        Tooltip(mse_label, self.forecaster.get_metric_explanation('mse'))
        
        rmse_label = ttk.Label(metrics_frame, textvariable=self.rmse_var)
        rmse_label.grid(row=0, column=2, padx=5, pady=5)
        Tooltip(rmse_label, self.forecaster.get_metric_explanation('rmse'))
        
        r2_label = ttk.Label(metrics_frame, textvariable=self.r2_var)
        r2_label.grid(row=0, column=3, padx=5, pady=5)
        Tooltip(r2_label, self.forecaster.get_metric_explanation('r2'))
    
    def setup_dashboard_tab(self):
        """Настройте вкладку панели мониторинга"""
        # Фрейм для управления панелью мониторинга
        control_frame = ttk.Frame(self.dashboard_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="Dashboard Port:").pack(side=tk.LEFT, padx=5, pady=5)
        self.port_var = tk.IntVar(value=8050)
        ttk.Spinbox(control_frame, from_=8000, to=9000, increment=1, textvariable=self.port_var, width=10).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.dashboard_button = ttk.Button(control_frame, text="Start Dashboard", command=self.toggle_dashboard)
        self.dashboard_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Open in Browser", command=self.open_dashboard).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Фрейм для статуса
        status_frame = ttk.LabelFrame(self.dashboard_tab, text="Dashboard Status")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Dashboard is not running")
        ttk.Label(status_frame, textvariable=self.status_var, font=('Arial', 12)).pack(padx=20, pady=20)
        
        # Инструкции
        instruction_frame = ttk.LabelFrame(self.dashboard_tab, text="Instructions")
        instruction_frame.pack(fill=tk.X, padx=10, pady=10)
        
        instructions = """
        1. Start the dashboard server using the 'Start Dashboard' button
        2. Click 'Open in Browser' to view the dashboard in your web browser
        3. Use the dashboard to visualize sales data and forecasts
        4. Stop the dashboard server when you're done
        """
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=10)
    
    def show_feature_explanation(self, event):
        """Показать объяснение подсказки для функции под курсором"""
        item_id = self.feature_tree.identify_row(event.y)
        if not item_id:
            return

        # Получите имя функции из выбранного элемента
        feature = self.feature_tree.item(item_id)['values'][0]
        
        # Получите объяснение из прогнозировщика
        explanation = self.forecaster.get_feature_explanation(feature)
        
        # Показать подсказку
        x, y = event.x_root, event.y_root
        self.show_tooltip(x, y, explanation)
    
    def show_tooltip(self, x, y, text):
        """Показать временную подсказку окна"""
        # Закройте любую существующую подсказку
        try:
            if hasattr(self, 'tooltip_window') and self.tooltip_window:
                self.tooltip_window.destroy()
        except:
            pass
        
        # Создайте окно подсказки
        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x+10}+{y+10}")
        
        # Создайте рамку с содержимым
        frame = ttk.Frame(self.tooltip_window, borderwidth=1, relief="solid")
        frame.pack(ipadx=5, ipady=5)
        
        # Перенос текста до максимальной ширины
        wrapped_text = text
        if len(text) > 80:
            lines = []
            current_line = []
            for word in text.split():
                if len(' '.join(current_line + [word])) <= 80:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            wrapped_text = '\n'.join(lines)
        
        label = ttk.Label(frame, text=wrapped_text, justify="left", 
                         background="#ffffcc", relief="solid", borderwidth=0)
        label.pack(ipadx=3, ipady=3)
        
        # Автоматически уничтожить после задержки
        self.root.after(3000, self.hide_tooltip)
    
    def hide_tooltip(self):
        """Скрыть временную подсказку окна"""
        try:
            if hasattr(self, 'tooltip_window') and self.tooltip_window:
                self.tooltip_window.destroy()
                self.tooltip_window = None
        except:
            pass
    
    def browse_db_path(self):
        """Выберите файл базы данных или расположение"""
        filename = filedialog.asksaveasfilename(
            title="Select Database Path",
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        if filename:
            # Обновите путь базы данных
            self.db_path_var.set(filename)
            
            # Создайте новый менеджер базы данных с этим путем
            self.db_manager = DatabaseManager(db_path=filename)
    
    def browse_file(self, string_var):
        """Выберите файл и обновите StringVar"""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            string_var.set(filename)
    
    def initialize_database(self):
        """Инициализировать базу данных"""
        try:
            # Получите текущий путь из пользовательского интерфейса
            db_path = self.db_path_var.get()
            
            # Создайте новый менеджер базы данных, если путь изменился
            if db_path != self.db_manager.db_file_path:
                self.db_manager = DatabaseManager(db_path=db_path)
            
            db_file = self.db_manager.init_database()
            messagebox.showinfo("Success", f"Database initialized successfully at:\n{db_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize database: {str(e)}")
    
    def import_data(self):
        """Импортировать данные о продажах кофе из CSV файла"""
        if not self.db_manager:
            messagebox.showerror("Error", "Database not initialized")
            return
        
        csv_path = self.csv_path_var.get()
        if not csv_path:
            messagebox.showerror("Error", "No CSV file selected")
            return
        
        try:
            # Покажите пользователю сообщение о том, что импорт в процессе
            self.root.config(cursor="watch")
            messagebox.showinfo("Processing", "Importing coffee sales data. This may take a moment...")
            
            # Импортируйте данные о продажах кофе
            n_products, n_sales = self.db_manager.import_coffee_sales(csv_path)
            
            # Сбросьте курсор
            self.root.config(cursor="")
            messagebox.showinfo("Success", f"Imported {n_products} coffee products and {n_sales} coffee sales")
            
        except ValueError as e:
            # Сбросьте курсор
            self.root.config(cursor="")
            
            # Покажите более полезное сообщение об ошибке для общих проблем
            error_msg = str(e)
            if "Missing required columns" in error_msg:
                messagebox.showerror("CSV Format Error", 
                    f"The CSV file has an incorrect format.\n\n{error_msg}\n\n"
                    "Your CSV file should have columns for:\n"
                    "- date (sale date)\n"
                    "- cash_type (payment method)\n" 
                    "- money (price)\n"
                    "- coffee_name (product name)\n\n"
                    "Note: The app can recognize columns with spaces or underscores.\n"
                    "      For example, 'coffee name' or 'coffee_name' would both work.\n\n"
                    "Check your CSV file or use the template at:\n"
                    "sales_forecast/data/coffee_sales_template.csv")
            else:
                messagebox.showerror("Error", f"Failed to import data: {error_msg}")
            
        except Exception as e:
            # Сбросьте курсор
            self.root.config(cursor="")
            messagebox.showerror("Error", f"Failed to import data: {str(e)}")
            traceback.print_exc()
    
    def view_data(self, table_name):
        """Просмотрите данные из указанной таблицы"""
        try:
            self.db_manager.open_session()
            
            # Очистите существующее дерево
            for i in self.tree.get_children():
                self.tree.delete(i)
            
            # Получите данные из соответствующей таблицы
            if table_name == "coffee_products":
                data = self.db_manager.session.query(self.db_manager.CoffeeProduct).all()
                self.tree['columns'] = ('id', 'name')
                self.tree.column('id', width=50)
                self.tree.column('name', width=200)
                self.tree.heading('id', text='ID')
                self.tree.heading('name', text='Coffee Name')
                
                for item in data:
                    self.tree.insert('', 'end', values=(item.id, item.name))
                
                if len(data) == 0:
                    messagebox.showinfo("No Data", "No coffee products found in the database.")
            
            elif table_name == "coffee_sales":
                # Объедините CoffeeSale с CoffeeProduct для получения имени продукта
                data = self.db_manager.session.query(
                    self.db_manager.CoffeeSale,
                    self.db_manager.CoffeeProduct.name
                ).join(
                    self.db_manager.CoffeeProduct,
                    self.db_manager.CoffeeSale.product_id == self.db_manager.CoffeeProduct.id
                ).all()
                
                self.tree['columns'] = ('id', 'product', 'date', 'payment_type', 'price')
                self.tree.column('id', width=50)
                self.tree.column('product', width=150)
                self.tree.column('date', width=150)
                self.tree.column('payment_type', width=100)
                self.tree.column('price', width=80)
                self.tree.heading('id', text='ID')
                self.tree.heading('product', text='Coffee Product')
                self.tree.heading('date', text='Sale Date')
                self.tree.heading('payment_type', text='Payment Type')
                self.tree.heading('price', text='Price')
                
                for sale, product_name in data:
                    # Форматируйте дату для лучшего отображения
                    date_str = sale.sale_date.strftime("%Y-%m-%d %H:%M") if sale.sale_date else "N/A"
                    
                    self.tree.insert('', 'end', values=(
                        sale.id, 
                        product_name, 
                        date_str,
                        sale.payment_type, 
                        f"${sale.price:.2f}"
                    ))
                
                if len(data) == 0:
                    messagebox.showinfo("No Data", "No coffee sales found in the database.")
            
            self.db_manager.close_session()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view data: {str(e)}")
            traceback.print_exc()
            self.db_manager.close_session()
    
    def train_model(self):
        """Обучите модель прогнозирования для кофе"""
        try:
            # Получите параметры
            test_size = self.test_size_var.get() / 100  # Преобразуйте в пропорцию
            
            # Обновите прогнозировщик
            self.forecaster = SalesForecaster()
            
            # Получите данные о продажах кофе
            self.db_manager.open_session()
            
            # Получите данные о продажах кофе, объединенные с информацией о продукте
            coffee_sales_data = self.db_manager.get_coffee_sales_by_product()
            
            # Преобразуйте в DataFrame
            df = pd.DataFrame([
                {
                    'product_name': sale[0],  # Coffee product name
                    'sale_date': sale[1],     # Date of sale
                    'price': sale[2],         # Price
                    'payment_type': sale[3]   # Payment type
                }
                for sale in coffee_sales_data
            ])
            
            self.db_manager.close_session()
            
            if len(df) < 50:
                messagebox.showwarning("Warning", "Not enough coffee sales data for reliable training (minimum 50 records recommended)")
            
            # Train model specific to coffee sales
            metrics = self.forecaster.train_coffee_sales_model(df, 
                                           date_col='sale_date', 
                                           price_col='price', 
                                           product_col='product_name', 
                                           test_size=test_size)
            
            # Обновите метрики
            self.mae_var.set(f"MAE: {metrics['mae']:.2f}")
            self.mse_var.set(f"MSE: {metrics['mse']:.2f}")
            self.rmse_var.set(f"RMSE: {metrics['rmse']:.2f}")
            self.r2_var.set(f"R²: {metrics['r2']:.2f}")
            
            # Обновите важность функции
            feature_importance = self.forecaster.get_feature_importance()
            if feature_importance is not None:
                # Очистите существующие данные
                for i in self.feature_tree.get_children():
                    self.feature_tree.delete(i)
                
                # Добавьте новые данные
                for _, row in feature_importance.iterrows():
                    self.feature_tree.insert('', 'end', values=(row['Feature'], f"{row['Importance']:.4f}"))
            
            messagebox.showinfo("Success", "Coffee sales forecasting model trained successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            traceback.print_exc()
    
    def generate_forecast(self):
        """Сгенерируйте прогноз продаж кофе"""
        if self.forecaster.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        try:
            # Получите параметры
            forecast_days = self.forecast_days_var.get()
            
            # Получите данные о продажах кофе
            self.db_manager.open_session()
            
            # Получите данные о продажах кофе, объединенные с информацией о продукте
            coffee_sales_data = self.db_manager.get_coffee_sales_by_product()
            
            # Преобразуйте в DataFrame
            df = pd.DataFrame([
                {
                    'product_name': sale[0],  # Coffee product name
                    'sale_date': sale[1],     # Date of sale
                    'price': sale[2],         # Price
                    'payment_type': sale[3]   # Payment type
                }
                for sale in coffee_sales_data
            ])
            
            self.db_manager.close_session()
            
            # Проверка наличия необходимых колонок
            if len(df) == 0:
                messagebox.showwarning("No Data", "No sales data found. Please import data first.")
                return
                
            # Проверка, что колонки существуют и имеют правильные имена
            required_columns = ['sale_date', 'price', 'product_name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                error_message = f"Missing required columns: {', '.join(missing_columns)}"
                messagebox.showerror("Error", error_message)
                return
                
            # Убедитесь, что колонка даты имеет правильный тип данных
            if not pd.api.types.is_datetime64_any_dtype(df['sale_date']):
                try:
                    df['sale_date'] = pd.to_datetime(df['sale_date'])
                except Exception as e:
                    messagebox.showerror("Error", f"Could not convert sale_date to datetime: {str(e)}")
                    return
            
            # Сгенерируйте прогноз с использованием метода кофе-специфичного
            forecast_df = self.forecaster.predict_coffee_sales(df, 
                                               future_days=forecast_days,
                                               date_col='sale_date', 
                                               price_col='price', 
                                               product_col='product_name')
            
            # Убедитесь, что данные прогноза имеют стандартные имена столбцов для панели мониторинга
            # Проверьте, содержит ли прогноз прогнозируемые значения цены
            if 'price_pred' in forecast_df.columns and 'predicted_amount' not in forecast_df.columns:
                # Сделайте копию прогнозируемых значений цены под стандартным именем
                forecast_df['predicted_amount'] = forecast_df['price_pred']
            
            # Обновите данные панели мониторинга
            feature_importance = self.forecaster.get_feature_importance()
            self.dashboard.update_data(df, forecast_df, feature_importance)
            
            messagebox.showinfo("Success", f"Coffee sales forecast generated for the next {forecast_days} days")
            
            # Рекомендуйте открыть панель мониторинга, если она не запущена
            if not hasattr(self, 'dashboard_running') or not self.dashboard_running:
                if messagebox.askyesno("Open Dashboard", "Would you like to open the dashboard to view the forecast?"):
                    self.toggle_dashboard()
                    self.open_dashboard()
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate forecast: {str(e)}\n\n{traceback.format_exc()}")
            traceback.print_exc()
    
    def save_model(self):
        """Сохраните обученную модель"""
        if self.forecaster.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        try:
            # Получите директорию сохранения
            save_dir = filedialog.askdirectory(title="Select Directory to Save Model")
            if not save_dir:
                return
            
            # Получите пользовательское имя, если оно указано
            custom_name = self.model_name_var.get() or None
            
            # Сохраните модель
            saved_path = self.forecaster.save_model(model_path=save_dir, custom_name=custom_name)
            messagebox.showinfo("Success", f"Model saved to {saved_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Загрузите обученную модель"""
        try:
            # Получите файл модели
            model_file = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Model files", "*.pkl"), ("All files", "*.*")]
            )
            if not model_file:
                return
            
            # Загрузите модель
            success = self.forecaster.load_model(model_path=model_file)
            
            if not success:
                messagebox.showwarning("Warning", "Could not load the selected model file")
                return
                
            # Обновите важность функции
            feature_importance = self.forecaster.get_feature_importance()
            if feature_importance is not None:
                # Очистите существующие данные
                for i in self.feature_tree.get_children():
                    self.feature_tree.delete(i)
                
                # Добавьте новые данные
                for _, row in feature_importance.iterrows():
                    self.feature_tree.insert('', 'end', values=(row['Feature'], f"{row['Importance']:.4f}"))
            
            # Обновите метрики, если они доступны
            if hasattr(self.forecaster, 'metrics') and self.forecaster.metrics:
                metrics = self.forecaster.metrics
                self.mae_var.set(f"MAE: {metrics.get('mae', 0):.2f}")
                self.mse_var.set(f"MSE: {metrics.get('mse', 0):.2f}")
                self.rmse_var.set(f"RMSE: {metrics.get('rmse', 0):.2f}")
                self.r2_var.set(f"R²: {metrics.get('r2', 0):.2f}")
            
            messagebox.showinfo("Success", "Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def toggle_dashboard(self):
        """Запустите или остановите сервер панели мониторинга"""
        if self.dashboard_running:
            # Остановите панель
            self.dashboard_running = False
            self.status_var.set("Dashboard shutting down...")
            self.dashboard_button.configure(text="Start Dashboard")
            # Завершите сервер панели
            if hasattr(self, 'dashboard'):
                try:
                    self.dashboard.shutdown()
                    print("Dashboard server stopped")
                except Exception as e:
                    print(f"Error stopping dashboard: {str(e)}")
            # Дождитесь завершения потока
            if hasattr(self, 'dashboard_thread') and self.dashboard_thread.is_alive():
                self.dashboard_thread.join(timeout=5)
                print("Dashboard thread joined")
        else:
            # Запустите панель
            self.dashboard_running = True
            port = self.port_var.get()
            self.status_var.set(f"Dashboard running at http://localhost:{port}")
            self.dashboard_button.configure(text="Stop Dashboard")
            
            # Запустите панель в отдельном потоке
            self.dashboard_thread = threading.Thread(target=self.run_dashboard, args=(port,))
            self.dashboard_thread.daemon = True
            self.dashboard_thread.start()
    
    def run_dashboard(self, port):
        """Запустите сервер панели мониторинга"""
        try:
            success = self.dashboard.run_server(debug=False, port=port)
            if not success:
                # Обновите статус на главном потоке
                self.root.after(0, lambda: self.status_var.set("Dashboard error: Failed to start server"))
                self.root.after(0, lambda: self.dashboard_button.configure(text="Start Dashboard"))
                self.dashboard_running = False
        except Exception as e:
            print(f"Dashboard error: {str(e)}")
            self.dashboard_running = False
            # Обновите статус на главном потоке
            self.root.after(0, lambda: self.status_var.set(f"Dashboard error: {str(e)}"))
            self.root.after(0, lambda: self.dashboard_button.configure(text="Start Dashboard"))
    
    def open_dashboard(self):
        """Откройте панель мониторинга в веб-браузере"""
        if not self.dashboard_running:
            messagebox.showwarning("Warning", "Dashboard is not running. Please start it first.")
            return
        
        port = self.port_var.get()
        webbrowser.open(f"http://localhost:{port}")
    
    def on_closing(self):
        """Вызывается при закрытии приложения"""
        if self.dashboard_running:
            self.dashboard_running = False
        self.root.destroy()
    
    def view_coffee_products(self):
        """Просмотрите таблицу продуктов кофе"""
        self.view_data("coffee_products")
    
    def view_coffee_sales(self):
        """Просмотрите таблицу продаж кофе"""
        self.view_data("coffee_sales")

def main():
    """Основная функция для запуска приложения"""
    root = tk.Tk()
    app = SalesForecastApp(root)
    
    # Настройте обработчик закрытия
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main() 