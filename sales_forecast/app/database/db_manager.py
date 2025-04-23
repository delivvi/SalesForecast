import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import appdirs
from datetime import datetime
import re

from .models import Base, Customer, Product, Sale, SaleItem, CoffeeProduct, CoffeeSale

class DatabaseManager:
    def __init__(self, db_path=None):
        """Инициализация подключения к базе данных и сессии"""
        if db_path is None:
            # Создать директорию данных, если она не существует
            data_dir = appdirs.user_data_dir("SalesForecast", "SalesApp")
            os.makedirs(data_dir, exist_ok=True)
            db_path = f"sqlite:///{os.path.join(data_dir, 'sales_data.db')}"
        
        self.db_path = db_path
        
        # Удалить префикс sqlite:/// для отображения в интерфейсе
        if db_path.startswith('sqlite:///'):
            self.db_file_path = db_path[10:]
        else:
            self.db_file_path = db_path
            db_path = f"sqlite:///{db_path}"
            self.db_path = db_path
        
        self.engine = create_engine(db_path)
        self.Session = sessionmaker(bind=self.engine)
        self.session = None
        
        # Сделать модели доступными как атрибуты
        self.Customer = Customer
        self.Product = Product
        self.Sale = Sale
        self.SaleItem = SaleItem
        self.CoffeeProduct = CoffeeProduct
        self.CoffeeSale = CoffeeSale
    
    def init_database(self):
        """Создание таблицы в базе данных"""
        Base.metadata.create_all(self.engine)
        print(f"База данных успешно инициализирована по пути {self.db_file_path}")
        return self.db_file_path
    
    def open_session(self):
        """Открыть новую сессию"""
        self.session = self.Session()
        return self.session
    
    def close_session(self):
        """Закрыть текущую сессию"""
        if self.session:
            self.session.close()
            self.session = None
    
    def add_customer(self, name, email=None, address=None, phone=None):
        """Добавление нового клиента в базу данных"""
        try:
            customer = Customer(
                name=name,
                email=email,
                address=address,
                phone=phone
            )
            self.session.add(customer)
            self.session.commit()
            return customer
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def add_product(self, name, price, description=None, category=None, brand=None, sku=None):
        """Добавить новый продукт в базу данных"""
        try:
            product = Product(
                name=name,
                price=price,
                description=description,
                category=category,
                brand=brand,
                sku=sku
            )
            self.session.add(product)
            self.session.commit()
            return product
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def add_sale(self, product_id, sale_date, quantity, price):
        """
        Добавить новую продажу в базу данных
        
        Аргументы:
            product_id: ID продукта
            sale_date: Дата и время продажи
            quantity: Проданное количество
            price: Цена за единицу на момент продажи
        """
        try:
            total_amount = quantity * price
            
            sale = Sale(
                product_id=product_id,
                sale_date=sale_date,
                quantity=quantity,
                price=price,
                total_amount=total_amount
            )
            self.session.add(sale)
            self.session.commit()
            return sale
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def get_sales_data(self, start_date=None, end_date=None):
        """Получить данные о продажах для анализа, опционально фильтруя по диапазону дат"""
        query = self.session.query(Sale)
        
        if start_date:
            query = query.filter(Sale.sale_date >= start_date)
        if end_date:
            query = query.filter(Sale.sale_date <= end_date)
        
        return query.all()
    
    def get_product_sales(self, product_id=None, start_date=None, end_date=None):
        """Получить данные о продажах для конкретного продукта или всех продуктов"""
        query = self.session.query(
            Product.name,
            Product.category,
            Sale.quantity,
            Sale.price,
            Sale.sale_date,
            Sale.total_amount
        ).join(
            Sale, Sale.product_id == Product.id
        )
        
        if product_id:
            query = query.filter(Product.id == product_id)
        if start_date:
            query = query.filter(Sale.sale_date >= start_date)
        if end_date:
            query = query.filter(Sale.sale_date <= end_date)
        
        return query.all()
    
    def import_data_from_csv(self, csv_file):
        """
        Импортировать данные из одного CSV файла с объединенными данными о продуктах и продажах
        
        Аргументы:
            csv_file: Путь к CSV файлу
        """
        if not csv_file or not os.path.exists(csv_file):
            raise ValueError(f"CSV файл не существует: {csv_file}")
        
        try:
            # Попробовать разные кодировки
            encodings = ['utf-8', 'latin1', 'cp1251', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Не удалось прочитать CSV файл с любой из кодировок: {encodings}")
            
            # Сохранить исходные имена столбцов
            self.orig_columns = list(df.columns)
            
            # Обработать DataFrame для извлечения продуктов и продаж
            self._process_csv_data(df)
            
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Ошибка импорта данных: {str(e)}")
            raise e
    
    def _process_csv_data(self, df):
        """
        Обработать dataframe для извлечения информации о продуктах и продажах
        
        Аргументы:
            df: DataFrame, содержащий объединенные данные о продуктах и продажах
        """
        # Получить имена столбцов, чтобы определить структуру
        columns = [col.lower() for col in df.columns]
        print(f"Найдены столбцы: {columns}")
        
        # Попробовать идентифицировать ключевые столбцы
        product_name_col = self._find_column(columns, ['product', 'name', 'item', 'title'])
        price_col = self._find_column(columns, ['price', 'amount', 'cost', 'actual price'])
        date_col = self._find_column(columns, ['date', 'time', 'day'])
        quantity_col = self._find_column(columns, ['quantity', 'qty', 'count', 'units'])
        category_col = self._find_column(columns, ['category', 'type', 'department'])
        brand_col = self._find_column(columns, ['brand', 'manufacturer', 'maker'])
        sku_col = self._find_column(columns, ['sku', 'id', 'code', 'product id', 'item id'])
        
        # Проверить обязательные столбцы
        if not product_name_col:
            raise ValueError("Не удалось идентифицировать столбец с названием продукта")
        if not price_col:
            raise ValueError("Не удалось идентифицировать столбец с ценой")
        if not date_col:
            raise ValueError("Не удалось идентифицировать столбец с датой")
        
        # Использовать столбец количества, если он найден, иначе использовать 1
        quantity_default = 1
        
        # Отслеживать продукты по SKU или имени, чтобы избежать дубликатов
        product_map = {}
        
        # Обработать каждую строку
        for _, row in df.iterrows():
            try:
                # Извлечь информацию о продукте
                product_name = str(row[product_name_col])
                
                # Обработать цену - попробовать извлечь числовое значение
                price_str = str(row[price_col]).replace('₹', '').replace(',', '').strip()
                price_match = re.search(r'(\d+(\.\d+)?)', price_str)
                if price_match:
                    price = float(price_match.group(1))
                else:
                    price = 0.0
                
                # Обработать дату
                date_str = row[date_col]
                sale_date = self._parse_date(date_str)
                
                # Дополнительные поля
                if quantity_col and pd.notna(row[quantity_col]):
                    quantity_str = str(row[quantity_col]).replace('₹', '').replace(',', '').strip()
                    quantity_match = re.search(r'(\d+)', quantity_str)
                    if quantity_match:
                        quantity = int(quantity_match.group(1))
                    else:
                        quantity = quantity_default
                else:
                    quantity = quantity_default
                
                category = row[category_col] if category_col and pd.notna(row[category_col]) else None
                brand = row[brand_col] if brand_col and pd.notna(row[brand_col]) else None
                sku = str(row[sku_col]) if sku_col and pd.notna(row[sku_col]) else None
                
                # Пропустить строки без названия продукта или нулевой цены
                if not product_name or pd.isna(product_name) or product_name.strip() == '' or price <= 0:
                    continue
                
                # Использовать SKU в качестве ключа, если он доступен, иначе использовать имя продукта
                product_key = sku if sku else product_name
                
                # Добавить продукт, если его еще нет
                if product_key not in product_map:
                    product = Product(
                        name=product_name,
                        price=price,
                        category=category,
                        brand=brand,
                        sku=sku
                    )
                    self.session.add(product)
                    self.session.flush()  # Получить ID без коммита
                    product_map[product_key] = product.id
                
                # Add sale
                sale = Sale(
                    product_id=product_map[product_key],
                    sale_date=sale_date,
                    quantity=quantity,
                    price=price,
                    total_amount=price * quantity
                )
                self.session.add(sale)
            
            except Exception as e:
                print(f"Ошибка обработки строки: {e}")
                # Продолжить обработку других строк
                continue
        
        # Зафиксировать все данные
        self.session.commit()
    
    def _find_column(self, columns, possible_names):
        """
        Найти столбец в DataFrame на основе возможных имен
        
        Аргументы:
            columns: Список имен столбцов (в нижнем регистре)
            possible_names: Список возможных частей имен столбцов
            
        Возвращает:
            Оригинальное имя столбца или None, если не найдено
        """
        # Сначала проверяем точные совпадения
        for name in possible_names:
            if name in columns:
                idx = columns.index(name)
                return self.orig_columns[idx]
        
        # Затем проверяем частичные совпадения
        for name in possible_names:
            for idx, col in enumerate(columns):
                if name in col:
                    return self.orig_columns[idx]
        
        return None
    
    def _parse_date(self, date_str):
        """
        Обработать строку с датой в различных форматах
        
        Аргументы:
            date_str: Строка с датой
            
        Возвращает:
            Объект datetime
        """
        # Попробовать различные форматы даты
        formats = [
            '%Y-%m-%d',          # 2023-01-01
            '%d/%m/%Y',          # 01/01/2023
            '%m/%d/%Y',          # 01/01/2023
            '%d-%m-%Y',          # 01-01-2023
            '%m-%d-%Y',          # 01-01-2023
            '%d.%m.%Y',          # 01.01.2023
            '%Y/%m/%d',          # 2023/01/01
            '%b %d, %Y',         # Jan 01, 2023
            '%d %b %Y',          # 01 Jan 2023
            '%B %d, %Y',         # January 01, 2023
            '%d %B %Y',          # 01 January 2023
            '%Y-%m-%d %H:%M:%S'  # 2023-01-01 12:00:00
        ]
        
        # Очистить строку с датой
        date_str = str(date_str).strip()
        
        # Попробовать каждый формат
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Попробовать извлечь дату с помощью регулярных выражений, если все форматы не подошли
        date_patterns = [
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD или YYYY/MM/DD
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD-MM-YYYY или MM-DD-YYYY
            r'(\d{1,2})/(\d{1,2})/(\d{2,4})'       # MM/DD/YY или DD/MM/YY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                groups = match.groups()
                if len(groups[0]) == 4:  # YYYY-MM-DD
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                elif len(groups[2]) == 4:  # DD-MM-YYYY или MM-DD-YYYY
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    # Предполагаем американский формат MM-DD-YYYY, если месяц <= 12
                    if day > 12:
                        day, month = month, day
                else:  # MM/DD/YY или DD/MM/YY
                    if int(groups[0]) > 12:  # Должен быть DD/MM/YY
                        day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    else:  # Предполагаем MM/DD/YY
                        month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                    
                    # Исправить двузначные годы
                    if year < 100:
                        year = 2000 + year if year < 50 else 1900 + year
                
                # Обработать недопустимые даты
                if month > 12:
                    month = 12
                if day > 31:
                    day = 31
                
                return datetime(year, month, day)
        
        # Если все методы анализа не удались, используем текущую дату
        print(f"Предупреждение: Не удалось проанализировать дату '{date_str}', используется текущая дата")
        return datetime.now()
    
    def init_db(self):
        """Создать таблицы в базе данных"""
        Base.metadata.create_all(self.engine)
        print(f"База данных успешно инициализирована по пути {self.db_file_path}")
        return self.db_file_path
        
    def recreate_db(self):
        """Удалить все таблицы и заново создать схему базы данных"""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        print(f"База данных успешно пересоздана по пути {self.db_file_path}")
        return self.db_file_path
        
    def session_scope(self):
        """Обеспечить транзакционную область для серии операций."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def import_coffee_sales(self, csv_file):
        """
        Импортировать данные о продажах кофе из CSV-файла
        
        Аргументы:
            csv_file: Путь к CSV-файлу с данными о продажах кофе
        
        Возвращает:
            Кортеж из (количество импортированных продуктов, количество импортированных продаж)
        """
        if not csv_file or not os.path.exists(csv_file):
            raise ValueError(f"CSV-файл не существует: {csv_file}")
        
        # Открыть сессию перед любой операцией с базой данных
        self.open_session()
        
        try:
            # Прочитать CSV-файл - попробовать разные кодировки
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_file, encoding='latin1')
                except:
                    df = pd.read_csv(csv_file, encoding='cp1252')
            
            # Вывести столбцы для отладки
            print(f"Найдены столбцы CSV: {df.columns.tolist()}")
            
            # Сопоставить возможные имена столбцов с нашими ожидаемыми именами
            column_mappings = {
                'date': ['date', 'timestamp', 'sale_date', 'day', 'time'],
                'cash type': ['cash type', 'cash_type', 'payment_type', 'payment', 'payment method', 'type', 'card'],
                'money': ['money', 'price', 'amount', 'cost', 'value', 'sales'],
                'coffee name': ['coffee name', 'coffee_name', 'product', 'item', 'coffee', 'product name', 'name']
            }
            
            # Попытаться сопоставить фактические столбцы с необходимыми
            mapped_columns = {}
            for required_col, possible_cols in column_mappings.items():
                for col in df.columns:
                    # Преобразовать в нижний регистр для сравнения
                    col_lower = col.lower()
                    # Сравнить с пробелами/подчеркиваниями и без них
                    normalized_col = col_lower.replace('_', ' ')
                    
                    # Проверить, соответствует ли столбец любой из возможных вариаций
                    for possible_col in possible_cols:
                        possible_col_lower = possible_col.lower()
                        normalized_possible = possible_col_lower.replace('_', ' ')
                        
                        if col_lower == possible_col_lower or normalized_col == normalized_possible:
                            mapped_columns[required_col] = col
                            break
                    
                    if required_col in mapped_columns:
                        break
            
            print(f"Сопоставленные столбцы: {mapped_columns}")
            
            # Создать представление dataframe со стандартизированными именами столбцов
            df_view = pd.DataFrame()
            for required_col, actual_col in mapped_columns.items():
                df_view[required_col] = df[actual_col]
            
            # Проверить, есть ли у нас все необходимые столбцы после сопоставления
            required_columns = ['date', 'cash type', 'money', 'coffee name']
            missing_columns = [col for col in required_columns if col not in df_view.columns]
            
            if missing_columns:
                print(f"Отсутствуют необходимые столбцы после сопоставления: {missing_columns}")
                print(f"Доступные столбцы: {df.columns.tolist()}")
                print(f"Сопоставленные столбцы: {mapped_columns}")
                raise ValueError(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}\n\n"
                               f"Ожидалось: {required_columns}\n"
                               f"Найдено: {df.columns.tolist()}\n\n"
                               f"Пожалуйста, переименуйте столбцы, чтобы они соответствовали ожидаемому формату, или предоставьте CSV-файл с правильными именами столбцов.")
            
            # Преобразовать дату и создать временную метку, если доступно
            df_view['date'] = pd.to_datetime(df_view['date'])
            
            # Получить уникальные названия кофе
            unique_coffees = df_view['coffee name'].unique()
            print(f"Найдено {len(unique_coffees)} уникальных типов кофе")
            
            # Добавить продукты кофе в базу данных
            product_map = {}
            for coffee_name in unique_coffees:
                # Проверить, существует ли продукт уже
                product = self.session.query(self.CoffeeProduct).filter_by(name=coffee_name).first()
                if not product:
                    product = self.CoffeeProduct(name=coffee_name)
                    self.session.add(product)
                    self.session.flush()
                
                product_map[coffee_name] = product.id
            
            # Добавить данные о продажах
            sales_count = 0
            for _, row in df_view.iterrows():
                try:
                    product_id = product_map[row['coffee name']]
                    sale_date = row['date']
                    payment_type = row['cash type']
                    price = float(row['money'])
                    
                    sale = self.CoffeeSale(
                        product_id=product_id,
                        sale_date=sale_date,
                        timestamp=sale_date,  # Использовать sale_date как timestamp, если отдельный timestamp недоступен
                        payment_type=payment_type,
                        price=price
                    )
                    
                    self.session.add(sale)
                    sales_count += 1
                except Exception as e:
                    print(f"Ошибка обработки строки: {e}")
                    continue
            
            # Зафиксировать изменения
            self.session.commit()
            
            return len(unique_coffees), sales_count
        
        except Exception as e:
            # Выполнить откат изменений только если сессия существует и открыта
            if self.session:
                self.session.rollback()
            print(f"Ошибка импорта данных о продажах кофе: {str(e)}")
            raise e
        finally:
            # Всегда закрывать сессию
            self.close_session()
    
    def get_coffee_sales_data(self, start_date=None, end_date=None, product_id=None):
        """
        Получить данные о продажах кофе для анализа, опционально фильтруя по диапазону дат и продукту
        
        Аргументы:
            start_date: Опциональный фильтр начальной даты
            end_date: Опциональный фильтр конечной даты
            product_id: Опциональный фильтр по ID продукта
            
        Возвращает:
            Список объектов CoffeeSale
        """
        query = self.session.query(self.CoffeeSale)
        
        if start_date:
            query = query.filter(self.CoffeeSale.sale_date >= start_date)
        if end_date:
            query = query.filter(self.CoffeeSale.sale_date <= end_date)
        if product_id:
            query = query.filter(self.CoffeeSale.product_id == product_id)
        
        # Сортировать по дате
        query = query.order_by(self.CoffeeSale.sale_date)
        
        return query.all()
    
    def get_coffee_products(self):
        """
        Получить все продукты кофе
        
        Возвращает:
            Список объектов CoffeeProduct
        """
        return self.session.query(self.CoffeeProduct).all()
    
    def get_coffee_sales_by_product(self, start_date=None, end_date=None):
        """
        Получить данные о продажах кофе, объединенные с информацией о продукте
        
        Аргументы:
            start_date: Опциональный фильтр начальной даты
            end_date: Опциональный фильтр конечной даты
            
        Возвращает:
            Список кортежей (название_продукта, дата_продажи, цена)
        """
        query = self.session.query(
            self.CoffeeProduct.name,
            self.CoffeeSale.sale_date,
            self.CoffeeSale.price,
            self.CoffeeSale.payment_type
        ).join(
            self.CoffeeSale, self.CoffeeSale.product_id == self.CoffeeProduct.id
        )
        
        if start_date:
            query = query.filter(self.CoffeeSale.sale_date >= start_date)
        if end_date:
            query = query.filter(self.CoffeeSale.sale_date <= end_date)
        
        # Сортировать по дате
        query = query.order_by(self.CoffeeSale.sale_date)
        
        return query.all() 