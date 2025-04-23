import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import appdirs
import json
import time

class SalesForecaster:
    def __init__(self, model_type='random_forest'):
        """
        Инициализация прогнозировщика продаж
        
        Аргументы:
            model_type: Тип модели для использования ('random_forest' или 'linear')
        """
        self.model_type = 'random_forest'  # Только random forest поддерживается сейчас
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.target = None
        self.metrics = {}
        self.feature_importance = None
        self.metadata = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'random_forest'
        }
        
        # Настройка директории модели по умолчанию
        self.default_model_dir = os.path.join(
            appdirs.user_data_dir("SalesForecast", "SalesApp"),
            "models"
        )
        os.makedirs(self.default_model_dir, exist_ok=True)
        
        # Словарь объяснений признаков
        self.feature_explanations = {
            'day_of_week': 'День недели (0=Понедельник, 6=Воскресенье). Отражает недельные паттерны продаж.',
            'day_of_month': 'День месяца (1-31). Определяет месячные паттерны, такие как дни зарплаты или циклы счетов.',
            'month': 'Месяц года (1-12). Отражает сезонные паттерны и влияние праздников.',
            'year': 'Календарный год. Отражает долгосрочные тренды и экономические циклы.',
            'quarter': 'Квартал бизнес-года (1-4). Отражает квартальные бизнес-циклы и отчетные периоды.',
            'is_weekend': 'Является ли день выходным (1) или рабочим (0). Часто паттерны продаж отличаются в выходные дни.',
            'sales_lag_1': 'Продажи за 1 день назад. Сильный предиктор очень недавних тенденций продаж.',
            'sales_lag_2': 'Продажи за 2 дня назад. Отражает очень недавние паттерны продаж.',
            'sales_lag_3': 'Продажи за 3 дня назад. Отражает очень недавние паттерны продаж.',
            'sales_lag_7': 'Продажи за 7 дней назад (тот же день на прошлой неделе). Отражает недельные паттерны продаж.',
            'sales_lag_14': 'Продажи за 14 дней назад (тот же день две недели назад). Отражает двухнедельные паттерны.',
            'sales_lag_30': 'Продажи за 30 дней назад (примерно тот же день в прошлом месяце). Отражает месячные паттерны.',
            'sales_rolling_mean_7': 'Средние продажи за последние 7 дней. Отражает недельные тренды, сглаживая ежедневные колебания.',
            'sales_rolling_mean_14': 'Средние продажи за последние 14 дней. Отражает двухнедельные тренды.',
            'sales_rolling_mean_30': 'Средние продажи за последние 30 дней. Отражает месячные тренды, сглаживая недельные колебания.',
            'total_sales_lag_1': 'Общие продажи за 1 день назад.',
            'total_sales_lag_2': 'Общие продажи за 2 дня назад.',
            'total_sales_lag_3': 'Общие продажи за 3 дня назад.',
            'total_sales_lag_7': 'Общие продажи за 7 дней назад (тот же день на прошлой неделе).',
            'total_sales_rolling_3': 'Средние общие продажи за последние 3 дня.',
            'total_sales_rolling_7': 'Средние общие продажи за последние 7 дней.'
        }
        
        # Словарь объяснений метрик
        self.metric_explanations = {
            'mae': 'Средняя абсолютная ошибка: Средняя абсолютная разница между предсказанными и фактическими значениями. Чем ниже, тем лучше. Измеряет точность прогнозирования в тех же единицах, что и продажи.',
            'mse': 'Средняя квадратичная ошибка: Среднее значение квадратов разностей между предсказанными и фактическими значениями. Сильнее штрафует большие ошибки.',
            'rmse': 'Корень средней квадратичной ошибки: Квадратный корень из MSE. Более интерпретируемый, так как выражен в тех же единицах, что и продажи. Более низкие значения указывают на лучшее соответствие.',
            'r2': 'R-квадрат (Коэффициент детерминации): Доля дисперсии, объясненная моделью, варьируется от 0 до 1. Более высокие значения указывают на лучшее соответствие.'
        }
    
    def get_feature_explanation(self, feature):
        """Получить объяснение для конкретного признака"""
        # Проверить, является ли это признаком для конкретного продукта
        if '_lag_' in feature and not feature.startswith('total_sales_') and not feature.startswith('sales_'):
            product = feature.split('_lag_')[0]
            lag = feature.split('_lag_')[1]
            return f"Продажи {product} за {lag} день(дней) назад. Важно для тенденций конкретного продукта."
        
        # Стандартные объяснения признаков
        explanation = self.feature_explanations.get(feature, None)
        
        # Обработка признаков, начинающихся с 'total_sales_'
        if explanation is None and feature.startswith('total_sales_lag_'):
            lag = feature.split('_lag_')[1]
            explanation = f"Общие продажи кофе за {lag} день(дней) назад. Сильный предиктор недавних тенденций продаж."
        elif explanation is None and feature.startswith('total_sales_rolling_'):
            window = feature.split('_rolling_')[1]
            explanation = f"Средние общие продажи кофе за последние {window} дней. Отражает {window}-дневные тренды, сглаживая ежедневные колебания."
        
        # Если все еще нет объяснения
        if explanation is None:
            explanation = "Объяснение недоступно"
        
        return explanation
    
    def get_metric_explanation(self, metric):
        """Получить объяснение для конкретной метрики"""
        return self.metric_explanations.get(metric, "Объяснение недоступно")
    
    def _prepare_features(self, sales_data, date_col='sale_date', amount_col='total_amount'):
        """
        Подготовить признаки для модели из необработанных данных о продажах
        
        Аргументы:
            sales_data: DataFrame, содержащий данные о продажах
            date_col: Имя столбца для даты
            amount_col: Имя столбца для суммы продаж
        
        Возвращает:
            DataFrame с инженерными признаками
        """
        # Убедиться, что данные отсортированы по дате
        df = sales_data.copy()
        if not isinstance(df[date_col], pd.DatetimeIndex):
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        
        # Извлечь признаки даты
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Создать лагорифмические признаки (продажи за предыдущие дни/недели)
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'sales_lag_{lag}'] = df[amount_col].shift(lag)
        
        # Признаки скользящего среднего
        for window in [7, 14, 30]:
            df[f'sales_rolling_mean_{window}'] = df[amount_col].rolling(window=window).mean()
        
        # Удалить строки с NaN значениями (из-за лагорифмических признаков/скользящего среднего)
        df = df.dropna()
        
        # Установить признаки и целевую переменную
        self.features = [
            'day_of_week', 'day_of_month', 'month', 'year', 'quarter', 'is_weekend',
            'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
            'sales_rolling_mean_7', 'sales_rolling_mean_14', 'sales_rolling_mean_30'
        ]
        self.target = amount_col
        
        return df
    
    def train(self, sales_data, date_col='sale_date', amount_col='total_amount', test_size=0.2):
        """
        Обучить модель прогнозирования
        
        Аргументы:
            sales_data: DataFrame, содержащий данные о продажах
            date_col: Имя столбца для даты
            amount_col: Имя столбца для суммы продаж
            test_size: Доля данных для использования в тестировании
            
        Возвращает:
            Словарь метрик оценки
        """
        # Подготовить признаки
        df = self._prepare_features(sales_data, date_col, amount_col)
        
        # Проверить, достаточно ли у нас данных
        if len(df) < 30:
            print(f"Предупреждение: Доступно только {len(df)} точек данных. Рекомендуется минимум 30 для надежных моделей.")
        
        # Разделить данные
        X = df[self.features]
        y = df[self.target]
        
        # Масштабировать целевую переменную, чтобы предотвратить большие метрики ошибок
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=test_size, shuffle=False)
        
        # Масштабировать признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Инициализировать и обучить модель
        start_time = time.time()
        
        # Использовать оптимизированный Random Forest с большим количеством деревьев
        self.model = RandomForestRegressor(
            n_estimators=200,  # Увеличено со 100
            max_depth=None,    # Позволить деревьям расти полностью
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1  # Использовать все доступные ядра для более быстрого обучения
        )
        
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Оценить модель
        y_pred = self.model.predict(X_test_scaled)
        
        # Преобразовать предсказания и фактические значения обратно в исходный масштаб для метрик
        y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        self.metrics = {
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'mse': mean_squared_error(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'r2': r2_score(y_test_orig, y_pred_orig)
        }
        
        # Сохранить y_scaler для будущих предсказаний
        self.y_scaler = y_scaler
        
        # Обновить метаданные
        self.metadata.update({
            'training_time': training_time,
            'test_size': test_size,
            'data_points': len(df),
            'features': self.features,
            'metrics': self.metrics,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Сохранить модель автоматически в директорию по умолчанию
        self.save_model()
        
        return self.metrics
    
    def predict(self, sales_data, future_days=30, date_col='sale_date', amount_col='total_amount'):
        """
        Генерировать прогноз продаж на будущие дни
        
        Аргументы:
            sales_data: DataFrame, содержащий исторические данные о продажах
            future_days: Количество дней для прогнозирования
            date_col: Имя столбца для даты
            amount_col: Имя столбца для суммы продаж
            
        Возвращает:
            DataFrame с датой и прогнозируемой суммой продаж
        """
        if self.model is None:
            self.load_model()  # Попытаться загрузить модель, если она не загружена
            if self.model is None:
                raise ValueError("Модель должна быть обучена перед прогнозированием")
        
        # Подготовить признаки для исторических данных
        df = self._prepare_features(sales_data, date_col, amount_col)
        
        # Получить последнюю дату в данных
        last_date = df[date_col].max()
        
        # Инициализировать DataFrame прогноза
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
        forecast_df = pd.DataFrame({date_col: forecast_dates})
        
        # Инициализировать последними известными значениями
        last_values = df.iloc[-1].copy()
        
        # Генерировать прогнозы для каждого будущего дня
        for i, date in enumerate(forecast_dates):
            # Извлечь признаки даты
            day_of_week = date.dayofweek
            day_of_month = date.day
            month = date.month
            year = date.year
            quarter = (month - 1) // 3 + 1
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Подготовить вектор признаков
            features = {
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'month': month,
                'year': year,
                'quarter': quarter,
                'is_weekend': is_weekend
            }
            
            # Добавить лаговые признаки
            for lag, lag_days in enumerate([1, 2, 3, 7, 14, 30]):
                if i >= lag_days:
                    # Использовать уже предсказанное значение
                    lag_idx = i - lag_days
                    features[f'sales_lag_{lag_days}'] = forecast_df.iloc[lag_idx]['predicted_amount']
                else:
                    # Использовать исторические данные
                    features[f'sales_lag_{lag_days}'] = last_values[f'sales_lag_{lag_days}']
            
            # Добавить признаки скользящего среднего (упрощенный подход)
            for window in [7, 14, 30]:
                features[f'sales_rolling_mean_{window}'] = last_values[f'sales_rolling_mean_{window}']
            
            # Подготовить вектор признаков
            X = pd.DataFrame([features])[self.features]
            X_scaled = self.scaler.transform(X)
            
            # Сделать прогноз (возвращает масштабированное значение)
            scaled_prediction = self.model.predict(X_scaled)[0]
            
            # Преобразовать обратно в исходный масштаб, если y_scaler существует
            if hasattr(self, 'y_scaler') and self.y_scaler is not None:
                prediction = self.y_scaler.inverse_transform([[scaled_prediction]])[0][0]
            else:
                prediction = scaled_prediction
                
            forecast_df.loc[i, 'predicted_amount'] = prediction
            
            # Обновить скользящие средние, если нужно для следующих прогнозов
            for window in [7, 14, 30]:
                # Простой подход: просто сдвинуть скользящее среднее немного в сторону нового прогноза
                alpha = 1 / window  # Фактор сглаживания
                features[f'sales_rolling_mean_{window}'] = (
                    (1 - alpha) * features[f'sales_rolling_mean_{window}'] + alpha * prediction
                )
        
        return forecast_df[[date_col, 'predicted_amount']]
    
    def save_model(self, model_path=None, custom_name=None):
        """
        Сохранить обученную модель на диск
        
        Аргументы:
            model_path: Директория или путь к файлу для сохранения модели
            custom_name: Пользовательское имя для файла модели
            
        Возвращает:
            Путь к сохраненной модели
        """
        if self.model is None:
            print("Нет обученной модели для сохранения")
            return None
        
        try:
            import joblib
            import os
            
            # Генерировать имя файла
            if custom_name:
                filename = f"coffee_sales_{custom_name}.pkl"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"coffee_sales_model_{self.model_type}_{timestamp}.pkl"
            
            # Если model_path - это директория, добавить имя файла
            if model_path is None:
                # По умолчанию текущая директория, если не указано
                model_path = os.path.join(os.getcwd(), filename)
            elif os.path.isdir(model_path):
                model_path = os.path.join(model_path, filename)
            elif not model_path.endswith('.pkl'):
                model_path = f"{model_path}.pkl"
            
            # Получить важность признаков, если доступно
            feature_importance = None
            if hasattr(self, 'get_feature_importance'):
                try:
                    feature_importance = self.get_feature_importance()
                except:
                    feature_importance = None
            
            # Создать словарь данных модели
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'scaler': self.scaler,
                'features': self.features,
                'metrics': self.metrics,
                'feature_importance': feature_importance,
                'timestamp': datetime.now().isoformat(),
                'data_type': 'coffee_sales'
            }
            
            # Сохранить модель
            joblib.dump(model_data, model_path)
            print(f"Модель сохранена в {model_path}")
            
            return model_path
        
        except Exception as e:
            print(f"Ошибка сохранения модели: {str(e)}")
            return None

    def load_model(self, model_path=None, custom_name=None):
        """
        Загрузить обученную модель с диска
        
        Аргументы:
            model_path: Путь к файлу модели
            custom_name: Пользовательское имя файла модели
            
        Возвращает:
            Логическое значение, указывающее на успех
        """
        try:
            import joblib
            import os
            
            # Попытка загрузить модель
            if model_path is None:
                # Проверка, существует ли директория с моделью
                if hasattr(self, 'default_model_dir') and os.path.exists(self.default_model_dir):
                    # Поиск файлов модели в директории по умолчанию
                    model_files = [f for f in os.listdir(self.default_model_dir) if f.endswith('.pkl')]
                    if model_files:
                        # Использовать самый последний файл модели (предполагая, что в имени файла есть временная метка)
                        # Фильтрация файлов, которые не начинаются с "coffee_sales_model"
                        valid_models = [f for f in model_files if f.startswith("coffee_sales_model")]
                        if valid_models:
                            valid_models.sort(reverse=True)
                            model_path = os.path.join(self.default_model_dir, valid_models[0])
                            print(f"Использование последней модели файла: {model_path}")
            
            # Генерация имени файла, если custom_name предоставлен, но model_path все еще не указан
            if model_path is None and custom_name:
                # Поиск в текущей директории
                filename = f"coffee_sales_{custom_name}.pkl"
                model_path = os.path.join(os.getcwd(), filename)
            
            # Если model_path все еще не указан, вернуть False (нет модели)
            if model_path is None or not os.path.exists(model_path):
                if model_path is not None:
                    print(f"Файл модели не найден: {model_path}")
                else:
                    print("Не указан файл модели и не найдена модель по умолчанию")
                return False
            
            # Попытка загрузить файл модели
            try:
                model_data = joblib.load(model_path)
                
                # Проверка, является ли model_data словарь (ожидаемый формат для model_data)
                if not isinstance(model_data, dict):
                    print(f"Неверный формат файла модели: {model_path}")
                    return False
                
                # Проверка, является ли это моделью продаж кофе
                if 'data_type' in model_data and model_data['data_type'] == 'coffee_sales':
                    # Это модель продаж кофе, продолжить нормально
                    pass
                else:
                    # Это устаревшая модель, сделайте пользователю подтвердить
                    print("Предупреждение: Это кажется устаревшей (не кофе-продажи) моделью.")
                    print("Она может не быть совместимой с текущим приложением для прогнозирования продаж кофе.")
                    # В реальном приложении мы добавили бы диалоговое окно для подтверждения
                
                # Восстановить атрибуты модели
                self.model = model_data.get('model')
                self.model_type = model_data.get('model_type', 'random_forest')  # По умолчанию random_forest
                self.scaler = model_data.get('scaler')
                self.features = model_data.get('features', [])
                self.metrics = model_data.get('metrics', {})
                self.feature_importance = model_data.get('feature_importance')
                
                print(f"Модель загружена из {model_path}")
                return True
            except Exception as inner_e:
                print(f"Ошибка чтения файла модели {model_path}: {str(inner_e)}")
                return False
        
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            return False
    
    def get_feature_importance(self):
        """
        Получить важность признаков, если модель поддерживает ее
        
        Возвращает:
            DataFrame с важностью признаков
        """
        if self.model is None:
            try:
                self.load_model()
            except:
                raise ValueError("Сначала обучите модель")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances,
                'Explanation': [self.get_feature_explanation(f) for f in self.features]
            }).sort_values(by='Importance', ascending=False)
            return feature_importance
        elif hasattr(self.model, 'coef_'):
            # Для линейных моделей
            importances = np.abs(self.model.coef_)
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances,
                'Explanation': [self.get_feature_explanation(f) for f in self.features]
            }).sort_values(by='Importance', ascending=False)
            return feature_importance
        else:
            return None
    
    def prepare_coffee_sales_features(self, sales_data, date_col='sale_date', price_col='price', product_col='product_name'):
        """
        Подготовить признаки для данных о продажах кофе, опционально сгруппированных по продукту
        
        Аргументы:
            sales_data: DataFrame, содержащий данные о продажах кофе
            date_col: Имя столбца для даты
            price_col: Имя столбца для цены
            product_col: Имя столбца для названия продукта
            
        Returns:
            DataFrame с подготовленными признаками
        """
        # Убедиться, что данные отсортированы по дате
        df = sales_data.copy()
        
        # Проверить наличие необходимых колонок
        required_cols = [date_col, price_col]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
        
        # Преобразовать дату в datetime, если она еще не в этом формате
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        
        # Сгруппировать по дате и продукту для получения ежедневных продаж
        if product_col in df.columns:
            # Получим только дату (без времени) для группировки
            df['date_only'] = df[date_col].dt.date
            
            # Сгруппировать по дате и продукту для получения метрик по продуктам
            daily_sales = df.groupby(['date_only', product_col])[price_col].sum().reset_index()
            daily_sales.columns = ['date_only', product_col, 'total_sales']
            
            # Поворот для получения столбцов по продуктам
            pivoted = daily_sales.pivot(index='date_only', columns=product_col, values='total_sales').fillna(0)
            
            # Преобразуем индекс обратно в datetime
            pivoted.index = pd.to_datetime(pivoted.index)
            
            # Сохраним дату как отдельную колонку (для удобства)
            pivoted[date_col] = pivoted.index
            
            # Создать признаки даты
            pivoted['day_of_week'] = pivoted.index.dayofweek
            pivoted['day_of_month'] = pivoted.index.day
            pivoted['month'] = pivoted.index.month
            pivoted['year'] = pivoted.index.year
            pivoted['quarter'] = pivoted.index.quarter
            pivoted['is_weekend'] = pivoted['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Рассчитать общие продажи по всем продуктам
            # Исключим колонки признаков и дату при суммировании
            cols_to_exclude = [date_col, 'day_of_week', 'day_of_month', 'month', 'year', 'quarter', 'is_weekend']
            cols_to_sum = [col for col in pivoted.columns if col not in cols_to_exclude]
            pivoted['total_sales'] = pivoted[cols_to_sum].sum(axis=1)
            
            # Создать признаки для продуктов
            product_columns = df[product_col].unique()
            for product in product_columns:
                if product in pivoted.columns:
                    # Добавить лаговые признаки для каждого продукта
                    for lag in [1, 3]:
                        if len(pivoted) > lag:
                            pivoted[f'{product}_lag_{lag}'] = pivoted[product].shift(lag)
            
            # Создать лаговые признаки для общих продаж
            for lag in [1, 2, 3, 7]:
                if len(pivoted) > lag:
                    pivoted[f'total_sales_lag_{lag}'] = pivoted['total_sales'].shift(lag)
            
            # Создать скользящие средние для общих продаж
            for window in [3, 7]:
                if len(pivoted) > window:
                    pivoted[f'total_sales_rolling_{window}'] = pivoted['total_sales'].rolling(window=window).mean()
            
            # Заполнить пропущенные значения средними
            pivoted = pivoted.fillna(pivoted.mean())
            
            # Сохраним имена признаков
            self.date_features = ['day_of_week', 'day_of_month', 'month', 'year', 'quarter', 'is_weekend']
            self.total_features = [col for col in pivoted.columns if col.startswith('total_sales_lag_') or col.startswith('total_sales_rolling_')]
            self.product_features = {}
            for product in product_columns:
                if product in pivoted.columns:
                    self.product_features[product] = [col for col in pivoted.columns if col.startswith(f'{product}_lag_')]
            
            return pivoted
        else:
            # Если нет данных о продуктах, просто группируем по дате
            df['date_only'] = df[date_col].dt.date
            daily_sales = df.groupby('date_only')[price_col].sum().reset_index()
            daily_sales.columns = ['date_only', 'total_sales']
            
            # Преобразуем в datetime
            daily_sales['date_only'] = pd.to_datetime(daily_sales['date_only'])
            daily_sales = daily_sales.set_index('date_only')
            
            # Сохраним дату как отдельную колонку (для удобства)
            daily_sales[date_col] = daily_sales.index
            
            # Создать признаки даты
            daily_sales['day_of_week'] = daily_sales.index.dayofweek
            daily_sales['day_of_month'] = daily_sales.index.day
            daily_sales['month'] = daily_sales.index.month
            daily_sales['year'] = daily_sales.index.year
            daily_sales['quarter'] = daily_sales.index.quarter
            daily_sales['is_weekend'] = daily_sales['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Создать лаговые признаки
            for lag in [1, 2, 3, 7]:
                if len(daily_sales) > lag:
                    daily_sales[f'total_sales_lag_{lag}'] = daily_sales['total_sales'].shift(lag)
            
            # Создать скользящие средние
            for window in [3, 7]:
                if len(daily_sales) > window:
                    daily_sales[f'total_sales_rolling_{window}'] = daily_sales['total_sales'].rolling(window=window).mean()
            
            # Заполнить пропущенные значения средними
            daily_sales = daily_sales.fillna(daily_sales.mean())
            
            # Сохраним имена признаков
            self.date_features = ['day_of_week', 'day_of_month', 'month', 'year', 'quarter', 'is_weekend']
            self.total_features = [col for col in daily_sales.columns if col.startswith('total_sales_lag_') or col.startswith('total_sales_rolling_')]
            
            return daily_sales
    
    def train_coffee_sales_model(self, sales_data, date_col='sale_date', price_col='price', product_col='product_name', test_size=0.2):
        """
        Обучить модели для прогнозирования продаж кофе
        
        Аргументы:
            sales_data: DataFrame, содержащий данные о продажах кофе
            date_col: Имя столбца для даты
            price_col: Имя столбца для цены
            product_col: Имя столбца для названия продукта
            test_size: Доля данных для использования для тестирования
            
        Возвращает:
            Словарь с оценками метрик
        """
        # Подготовить признаки
        features_df = self.prepare_coffee_sales_features(sales_data, date_col, price_col, product_col)
        
        # Проверить, есть ли достаточно данных
        if len(features_df) < 14:
            print(f"Предупреждение: Доступно только {len(features_df)} точек данных. По крайней мере 14 рекомендуется для надежных моделей.")
        
        # Сохранить информацию о наборе данных
        self.has_product_data = product_col in sales_data.columns
        
        # Обучить модель для общих продаж
        all_features = self.date_features + self.total_features
        if not all(feature in features_df.columns for feature in all_features):
            all_features = [feature for feature in all_features if feature in features_df.columns]
        
        # Обучить основную модель для общих продаж
        X = features_df[all_features]
        y = features_df['total_sales']
        
        # Сохраним информацию о средних и максимальных значениях для валидации прогнозов
        self.sales_mean = y.mean()
        self.sales_max = y.max()
        self.sales_min = y.min()
        self.sales_std = y.std()
        
        # Масштабировать целевую переменную
        self.y_scaler = StandardScaler()
        y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Масштабировать признаки
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделить на обучающий и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=42)
        
        # Инициализировать и обучить модель
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        
        # Получить предсказания и оценить метрики
        y_pred = self.model.predict(X_test)
        
        # Преобразовать обратно в исходный масштаб
        y_test_orig = self.y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Рассчитать метрики
        metrics = {
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'mse': mean_squared_error(y_test_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'r2': r2_score(y_test_orig, y_pred_orig)
        }
        
        # Сохранить признаки и прочие данные
        self.features = all_features
        self.metrics = metrics
        
        # Сохранить метаданные
        self.metadata = {
            'model_type': self.model_type,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': len(features_df),
            'has_product_data': self.has_product_data,
            'metrics': metrics,
            'features': all_features,
            'sales_statistics': {
                'mean': float(self.sales_mean),
                'max': float(self.sales_max),
                'min': float(self.sales_min),
                'std': float(self.sales_std)
            }
        }
        
        return metrics
    
    def predict_coffee_sales(self, sales_data, future_days=30, date_col='sale_date', price_col='price', product_col='product_name'):
        """
        Генерировать прогноз продаж кофе
        
        Аргументы:
            sales_data: DataFrame с данными продаж
            future_days: Количество дней для прогнозирования
            date_col: Имя столбца с датой
            price_col: Имя столбца с ценой
            product_col: Имя столбца с названием продукта
            
        Возвращает:
            DataFrame с прогнозом продаж кофе
        """
        if not hasattr(self, 'model') or self.model is None:
            self.load_model()  # Попытаться загрузить модель, если она еще не загружена
            if self.model is None:
                raise ValueError("Модель должна быть обучена перед прогнозированием")
        
        # Получить последнюю дату в данных перед преобразованием
        # Убедимся, что колонка даты в правильном формате
        df_copy = sales_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        last_date = df_copy[date_col].max()
        
        # Получить список уникальных продуктов ПЕРЕД подготовкой признаков
        all_products = []
        if product_col in df_copy.columns:
            all_products = df_copy[product_col].unique().tolist()
        else:
            print(f"Предупреждение: колонка {product_col} не найдена в исходных данных")
            
        # Подготовить признаки для исторических данных
        df = self.prepare_coffee_sales_features(sales_data, date_col, price_col, product_col)
        
        # Создать датафрейм для прогноза с будущими датами
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
        
        # Создать датафрейм для общей суммы по дням
        date_df = pd.DataFrame({date_col: forecast_dates})
        
        # Получить последние значения для инициализации прогноза
        # Используем последнюю доступную дату
        last_row = df.iloc[-1].copy()
        
        # Создать DataFrame для прогноза
        future_df = pd.DataFrame(index=range(future_days))
        future_df[date_col] = forecast_dates
        
        # Инициализировать массив признаков для прогноза
        X_future = np.zeros((future_days, len(self.features)))
        
        # Инициализировать DataFrame для сохранения текущих значений
        current_values = pd.DataFrame(index=range(1))
        
        # Копируем последние значения из исторических данных
        for feature in self.features:
            if feature in last_row.index:
                current_values[feature] = last_row[feature]
            else:
                current_values[feature] = 0.0
        
        # Для каждого дня прогноза
        for i in range(future_days):
            date = forecast_dates[i]
            
            # Установить признаки даты
            current_values['day_of_week'] = date.dayofweek
            current_values['day_of_month'] = date.day
            current_values['month'] = date.month
            current_values['year'] = date.year
            current_values['quarter'] = (date.month - 1) // 3 + 1
            current_values['is_weekend'] = 1 if date.dayofweek >= 5 else 0
            
            # Создать массив признаков для текущего дня
            for j, feature in enumerate(self.features):
                if feature in current_values.columns:
                    X_future[i, j] = current_values[feature].values[0]
            
            # Масштабировать признаки
            X_future_scaled = self.scaler.transform(X_future[i:i+1])
            
            # Сделать прогноз (масштабированный)
            y_pred_scaled = self.model.predict(X_future_scaled)[0]
            
            # Преобразовать обратно в исходный масштаб
            y_pred = self.y_scaler.inverse_transform([[y_pred_scaled]])[0][0]
            
            # Применить ограничения, чтобы прогноз был правдоподобным
            # Не должен быть отрицательным
            y_pred = max(0, y_pred)
            
            # Не должен быть слишком аномальным (более 3-х стандартных отклонений от среднего)
            if hasattr(self, 'sales_max') and hasattr(self, 'sales_std'):
                max_reasonable_value = self.sales_max + 2 * self.sales_std
                if y_pred > max_reasonable_value:
                    y_pred = max_reasonable_value
            
            # Сохранить прогноз
            future_df.loc[i, 'predicted_total'] = y_pred
            
            # Обновить лаговые значения для следующего прогноза
            for lag_feat in self.total_features:
                if lag_feat.startswith('total_sales_lag_'):
                    lag = int(lag_feat.split('_')[-1])
                    if lag > 1:
                        prev_lag_feat = f'total_sales_lag_{lag-1}'
                        if prev_lag_feat in current_values.columns:
                            current_values[lag_feat] = current_values[prev_lag_feat]
                    else:  # lag == 1
                        current_values[lag_feat] = y_pred
                elif lag_feat.startswith('total_sales_rolling_'):
                    window = int(lag_feat.split('_')[-1])
                    alpha = 1.0 / window
                    current_values[lag_feat] = (1 - alpha) * current_values[lag_feat] + alpha * y_pred
        
        # Добавить имена столбцов, совместимые с панелью мониторинга
        future_df['predicted_amount'] = future_df['predicted_total']
        future_df['price_pred'] = future_df['predicted_total']
        
        # Добавить доверительные интервалы (упрощенный подход)
        error_margin = 0.15
        future_df['lower_bound'] = future_df['predicted_total'] * (1 - error_margin)
        future_df['upper_bound'] = future_df['predicted_total'] * (1 + error_margin)
        
        # Вычислить разные горизонты прогноза для визуализации
        future_df['forecast_7d'] = future_df['predicted_total'].rolling(7, min_periods=1).mean()
        future_df['forecast_30d'] = future_df['predicted_total'].rolling(30, min_periods=1).mean()
        
        print(f"Сгенерирован прогноз на {len(future_df)} дней с {future_df[date_col].min()} по {future_df[date_col].max()}")
        
        return future_df 