#!/usr/bin/env python
import os
import sys
from pathlib import Path

# Добавить корень проекта в путь Python
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.database.db_manager import DatabaseManager
import appdirs

def reset_database():
    """Сбросить базу данных приложения до чистого состояния"""
    # Получить путь к базе данных по умолчанию
    data_dir = appdirs.user_data_dir("SalesForecast", "SalesApp")
    db_file = os.path.join(data_dir, 'sales_data.db')
    
    print(f"Расположение базы данных: {db_file}")
    
    # Проверить, существует ли база данных
    if os.path.exists(db_file):
        print(f"Удаление существующего файла базы данных: {db_file}")
        try:
            os.remove(db_file)
            print("Файл базы данных удалён.")
        except Exception as e:
            print(f"Ошибка при удалении файла базы данных: {e}")
            return False
    
    # Инициализировать новую базу данных
    print("Создание новой базы данных...")
    db_manager = DatabaseManager()
    db_manager.init_db()
    print("Сброс базы данных успешно завершён.")
    
    return True

if __name__ == "__main__":
    success = reset_database()
    sys.exit(0 if success else 1) 