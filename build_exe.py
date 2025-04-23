#!/usr/bin/env python
"""
Скрипт сборки для приложения анализа продаж кофе

Этот скрипт компилирует Python-приложение в автономный исполняемый файл Windows
с использованием PyInstaller.
"""

import os
import subprocess
import sys
import shutil

def install_requirements():
    """Установка всех необходимых пакетов из requirements.txt"""
    print("Установка необходимых пакетов...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                          os.path.join("sales_forecast", "requirements.txt")])
    
    # Установка дополнительных зависимостей, которых может не быть в requirements.txt
    additional_deps = ["appdirs", "pyinstaller"]
    for dep in additional_deps:
        print(f"Установка {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

def build_executable():
    """Сборка исполняемого файла с помощью PyInstaller"""
    print("Сборка исполняемого файла...")
    
    # Определение команды PyInstaller с скрытыми импортами для отсутствующих модулей
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--hidden-import=appdirs",
        "--hidden-import=sqlalchemy.sql.default_comparator",  # Распространенная зависимость SQLAlchemy
        "--hidden-import=pandas._libs.tslibs.timedeltas",     # Распространенная зависимость pandas
        "--hidden-import=pkg_resources.py2_warn",            # Распространенная проблема зависимостей
        "--icon=sales_forecast/resources/icon.ico" if os.path.exists("sales_forecast/resources/icon.ico") else "",
        "--add-data=sales_forecast/resources;resources",
        "--add-data=coffee_sales.csv;.",
        "--add-data=coffee_sales_coffee model.pkl;.",
        "--name=CoffeeSalesAnalysis",
        "sales_forecast/main.py"
    ]
    
    # Удаление пустых аргументов
    cmd = [arg for arg in cmd if arg]
    
    # Запуск PyInstaller
    subprocess.check_call(cmd)
    
    print("Сборка завершена! Исполняемый файл создан в папке 'dist'.")

def main():
    """Основная функция для запуска процесса сборки"""
    # Создание чистой сборки
    if os.path.exists("build"):
        print("Удаление предыдущей директории build...")
        shutil.rmtree("build")
    
    if os.path.exists("dist"):
        print("Удаление предыдущей директории dist...")
        shutil.rmtree("dist")
    
    # Установка зависимостей
    install_requirements()
    
    # Сборка исполняемого файла
    build_executable()
    
    print("\nПроцесс сборки успешно завершен!")
    print("Вы можете найти исполняемый файл в папке 'dist'.")

if __name__ == "__main__":
    main() 