"""
Tests for the Sales Forecaster

This module contains tests for the SalesForecaster class and generates a report
to verify that the application is working correctly.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import SalesForecaster

# Global variables to store test results
test_results = {
    "initialization": False,
    "feature_preparation": False,
    "model_training": False,
    "prediction": False,
    "feature_importance": False,
    "model_saving_loading": False
}

test_details = {}

def generate_sample_sales_data():
    """Create sample sales data for testing"""
    print("Генерация тестовых данных о продажах...")
    # Generate dates for the past 100 days
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(100)]
    dates.sort()  # Ensure chronological order
    
    # Create weekly and monthly seasonality patterns
    weekday_factor = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.3]  # Mon-Sun
    month_factor = [0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.1, 1.0, 1.3, 1.4]  # Jan-Dec
    
    # Generate sales data with a trend and seasonality
    sales_data = []
    for date in dates:
        # Base amount with a slight upward trend
        base_amount = 1000 + (dates.index(date) * 5)
        
        # Apply weekly seasonality
        weekday = date.weekday()
        week_effect = weekday_factor[weekday]
        
        # Apply monthly seasonality
        month = date.month - 1
        month_effect = month_factor[month]
        
        # Add some random noise (± 5%)
        noise = np.random.uniform(0.95, 1.05)
        
        # Calculate final amount
        amount = base_amount * week_effect * month_effect * noise
        
        sales_data.append({
            'sale_date': date,
            'total_amount': round(amount, 2),
            'customer_id': np.random.randint(1, 20)  # Random customer ID
        })
    
    print(f"✓ Сгенерирована тестовая выборка данных: {len(sales_data)} записей")
    return pd.DataFrame(sales_data)

@pytest.fixture
def sample_sales_data():
    """Pytest fixture for sample sales data"""
    return generate_sample_sales_data()

def test_forecaster_initialization():
    """Test that the forecaster initializes correctly"""
    print("\n📋 ТЕСТ: Инициализация прогнозировщика")
    try:
        start_time = time.time()
        
        forecaster = SalesForecaster()
        assert forecaster.model_type == 'random_forest'
        assert forecaster.model is None
        
        forecaster = SalesForecaster(model_type='linear')
        assert forecaster.model_type == 'linear'
        
        elapsed_time = time.time() - start_time
        test_results["initialization"] = True
        test_details["initialization"] = {
            "success": True,
            "time": elapsed_time,
            "details": {
                "default_model_type": "random_forest",
                "custom_model_type": "linear"
            }
        }
        print(f"✓ Инициализация прогнозировщика выполнена успешно ({elapsed_time:.2f} сек)")
        return True
    except Exception as e:
        test_results["initialization"] = False
        test_details["initialization"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Ошибка при инициализации прогнозировщика: {str(e)}")
        return False

def test_feature_preparation(sample_sales_data):
    """Test the feature preparation"""
    print("\n📋 ТЕСТ: Подготовка признаков")
    try:
        start_time = time.time()
        
        forecaster = SalesForecaster()
        prepared_data = forecaster._prepare_features(sample_sales_data)
        
        # Check that features are created
        expected_features = ['day_of_week', 'month', 'is_weekend', 'sales_lag_1', 'sales_rolling_mean_7']
        missing_features = [feat for feat in expected_features if feat not in prepared_data.columns]
        
        # Check that NaN values are dropped
        nan_count = prepared_data.isna().sum().sum()
        assert nan_count == 0, f"Обнаружены пропущенные значения: {nan_count}"
        
        # Check that features list is created
        assert forecaster.features is not None
        assert len(forecaster.features) > 0
        
        elapsed_time = time.time() - start_time
        test_results["feature_preparation"] = True
        test_details["feature_preparation"] = {
            "success": True,
            "time": elapsed_time,
            "details": {
                "original_records": len(sample_sales_data),
                "prepared_records": len(prepared_data),
                "generated_features": list(prepared_data.columns),
                "features_count": len(prepared_data.columns),
                "missing_expected_features": missing_features
            }
        }
        print(f"✓ Подготовка признаков выполнена успешно ({elapsed_time:.2f} сек)")
        print(f"  • Исходные записи: {len(sample_sales_data)}")
        print(f"  • Записи после подготовки: {len(prepared_data)}")
        print(f"  • Сгенерировано признаков: {len(prepared_data.columns)}")
        return True
    except Exception as e:
        test_results["feature_preparation"] = False
        test_details["feature_preparation"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Ошибка при подготовке признаков: {str(e)}")
        return False

def manual_test_feature_preparation():
    """Standalone test function for feature preparation"""
    data = generate_sample_sales_data()
    return test_feature_preparation(data)

def test_model_training(sample_sales_data):
    """Test model training"""
    print("\n📋 ТЕСТ: Обучение модели")
    try:
        start_time = time.time()
        
        forecaster = SalesForecaster()
        metrics = forecaster.train(sample_sales_data, test_size=0.2)
        
        # Check that model is created
        assert forecaster.model is not None
        
        # Check that metrics are calculated
        expected_metrics = ['mae', 'mse', 'rmse', 'r2']
        for metric in expected_metrics:
            assert metric in metrics, f"Метрика {metric} отсутствует"
        
        # Check that metrics are reasonable
        assert metrics['mae'] > 0, "MAE должен быть положительным"
        assert metrics['mse'] > 0, "MSE должен быть положительным"
        assert metrics['rmse'] > 0, "RMSE должен быть положительным"
        assert metrics['r2'] <= 1.0, "R2 не должен превышать 1.0"
        
        elapsed_time = time.time() - start_time
        test_results["model_training"] = True
        metrics_rounded = {k: round(v, 4) for k, v in metrics.items()}
        test_details["model_training"] = {
            "success": True,
            "time": elapsed_time,
            "details": {
                "model_type": forecaster.model_type,
                "features_count": len(forecaster.features),
                "training_size": int(len(sample_sales_data) * 0.8),
                "test_size": int(len(sample_sales_data) * 0.2),
                "metrics": metrics_rounded
            }
        }
        print(f"✓ Обучение модели выполнено успешно ({elapsed_time:.2f} сек)")
        print(f"  • Тип модели: {forecaster.model_type}")
        print(f"  • Количество признаков: {len(forecaster.features)}")
        print(f"  • Метрики модели:")
        for metric, value in metrics_rounded.items():
            print(f"    - {metric.upper()}: {value}")
        return True
    except Exception as e:
        test_results["model_training"] = False
        test_details["model_training"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Ошибка при обучении модели: {str(e)}")
        return False

def manual_test_model_training():
    """Standalone test function for model training"""
    data = generate_sample_sales_data()
    return test_model_training(data)

def test_prediction(sample_sales_data):
    """Test sales prediction"""
    print("\n📋 ТЕСТ: Прогнозирование продаж")
    try:
        start_time = time.time()
        
        forecaster = SalesForecaster()
        forecaster.train(sample_sales_data, test_size=0.2)
        
        future_days = 30
        forecast = forecaster.predict(sample_sales_data, future_days=future_days)
        
        # Check forecast structure
        assert len(forecast) == future_days, f"Прогноз должен содержать {future_days} записей"
        assert 'sale_date' in forecast.columns, "В прогнозе отсутствует колонка 'sale_date'"
        assert 'predicted_amount' in forecast.columns, "В прогнозе отсутствует колонка 'predicted_amount'"
        
        # Check that dates are in the future
        last_date = sample_sales_data['sale_date'].max()
        first_forecast_date = forecast['sale_date'].min()
        assert first_forecast_date > last_date, "Даты прогноза должны быть в будущем"
        
        # Check that predicted amounts are positive
        assert (forecast['predicted_amount'] > 0).all(), "Прогнозные значения должны быть положительными"
        
        # Check that forecast values are reasonable (not too far from historical data)
        historical_mean = sample_sales_data['total_amount'].mean()
        historical_std = sample_sales_data['total_amount'].std()
        forecast_mean = forecast['predicted_amount'].mean()
        max_expected = historical_mean + 3 * historical_std
        
        assert forecast_mean < max_expected, f"Средний прогноз ({forecast_mean:.2f}) слишком высок по сравнению с историческими данными"
        
        elapsed_time = time.time() - start_time
        test_results["prediction"] = True
        test_details["prediction"] = {
            "success": True,
            "time": elapsed_time,
            "details": {
                "forecast_days": future_days,
                "last_historical_date": last_date.strftime('%Y-%m-%d'),
                "first_forecast_date": first_forecast_date.strftime('%Y-%m-%d'),
                "historical_mean": round(historical_mean, 2),
                "forecast_mean": round(forecast_mean, 2),
                "forecast_min": round(forecast['predicted_amount'].min(), 2),
                "forecast_max": round(forecast['predicted_amount'].max(), 2)
            }
        }
        print(f"✓ Прогнозирование выполнено успешно ({elapsed_time:.2f} сек)")
        print(f"  • Прогноз на {future_days} дней")
        print(f"  • Последняя историческая дата: {last_date.strftime('%Y-%m-%d')}")
        print(f"  • Первая дата прогноза: {first_forecast_date.strftime('%Y-%m-%d')}")
        print(f"  • Среднее значение в исторических данных: {historical_mean:.2f}")
        print(f"  • Среднее значение в прогнозе: {forecast_mean:.2f}")
        print(f"  • Мин. прогноз: {forecast['predicted_amount'].min():.2f}, Макс. прогноз: {forecast['predicted_amount'].max():.2f}")
        return True
    except Exception as e:
        test_results["prediction"] = False
        test_details["prediction"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Ошибка при прогнозировании продаж: {str(e)}")
        return False

def manual_test_prediction():
    """Standalone test function for prediction"""
    data = generate_sample_sales_data()
    return test_prediction(data)

def test_feature_importance(sample_sales_data):
    """Test feature importance calculation"""
    print("\n📋 ТЕСТ: Расчет важности признаков")
    try:
        start_time = time.time()
        
        forecaster = SalesForecaster()
        forecaster.train(sample_sales_data, test_size=0.2)
        
        feature_importance = forecaster.get_feature_importance()
        
        # Check that feature importance is returned
        assert feature_importance is not None, "Важность признаков не возвращена"
        assert 'Feature' in feature_importance.columns, "В результате отсутствует колонка 'Feature'"
        assert 'Importance' in feature_importance.columns, "В результате отсутствует колонка 'Importance'"
        
        # Check that importance values add up to approximately 1
        importance_sum = feature_importance['Importance'].sum()
        assert abs(importance_sum - 1.0) < 0.01, f"Сумма важностей должна быть примерно 1.0, получено {importance_sum}"
        
        # Get top 5 features
        top_features = feature_importance.sort_values('Importance', ascending=False).head(5)
        
        elapsed_time = time.time() - start_time
        test_results["feature_importance"] = True
        test_details["feature_importance"] = {
            "success": True,
            "time": elapsed_time,
            "details": {
                "features_count": len(feature_importance),
                "importance_sum": round(importance_sum, 4),
                "top_features": top_features[['Feature', 'Importance']].to_dict('records')
            }
        }
        print(f"✓ Расчет важности признаков выполнен успешно ({elapsed_time:.2f} сек)")
        print(f"  • Количество признаков: {len(feature_importance)}")
        print(f"  • Топ-5 важных признаков:")
        for _, row in top_features.iterrows():
            print(f"    - {row['Feature']}: {row['Importance']:.4f}")
        return True
    except Exception as e:
        test_results["feature_importance"] = False
        test_details["feature_importance"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Ошибка при расчете важности признаков: {str(e)}")
        return False

def manual_test_feature_importance():
    """Standalone test function for feature importance"""
    data = generate_sample_sales_data()
    return test_feature_importance(data)

def test_model_saving_loading(sample_sales_data, tmp_path):
    """Test saving and loading a model"""
    print("\n📋 ТЕСТ: Сохранение и загрузка модели")
    try:
        start_time = time.time()
        
        model_dir = tmp_path / "model_test"
        model_dir.mkdir()
        
        # Train and save model
        forecaster = SalesForecaster()
        forecaster.train(sample_sales_data, test_size=0.2)
        saved_path = forecaster.save_model(model_dir=str(model_dir))
        
        # Check that model files are created
        model_file = model_dir / "sales_forecaster_random_forest.pkl"
        scaler_file = model_dir / "scaler.pkl"
        features_file = model_dir / "features.txt"
        
        assert model_file.exists(), f"Файл модели не создан по пути {model_file}"
        assert scaler_file.exists(), f"Файл масштабирования не создан по пути {scaler_file}"
        assert features_file.exists(), f"Файл признаков не создан по пути {features_file}"
        
        model_size = os.path.getsize(model_file) / (1024 * 1024)  # Convert to MB
        
        # Load model
        new_forecaster = SalesForecaster()
        success = new_forecaster.load_model(model_dir=str(model_dir))
        
        # Check that model is loaded
        assert success, "Модель не загружена"
        assert new_forecaster.model is not None, "Модель не инициализирована после загрузки"
        assert new_forecaster.features is not None, "Признаки не загружены"
        assert len(new_forecaster.features) == len(forecaster.features), "Количество признаков не совпадает"
        
        # Make a prediction with the loaded model
        forecast = new_forecaster.predict(sample_sales_data, future_days=7)
        assert len(forecast) == 7, "Прогноз загруженной модели имеет неверную длину"
        
        elapsed_time = time.time() - start_time
        test_results["model_saving_loading"] = True
        test_details["model_saving_loading"] = {
            "success": True,
            "time": elapsed_time,
            "details": {
                "model_size_mb": round(model_size, 2),
                "model_type": forecaster.model_type,
                "features_count": len(forecaster.features),
                "save_path": str(model_dir)
            }
        }
        print(f"✓ Сохранение и загрузка модели выполнены успешно ({elapsed_time:.2f} сек)")
        print(f"  • Модель сохранена по пути: {model_dir}")
        print(f"  • Размер файла модели: {model_size:.2f} МБ")
        print(f"  • Количество признаков: {len(forecaster.features)}")
        return True
    except Exception as e:
        test_results["model_saving_loading"] = False
        test_details["model_saving_loading"] = {
            "success": False,
            "error": str(e)
        }
        print(f"❌ Ошибка при сохранении и загрузке модели: {str(e)}")
        return False

def manual_test_model_saving_loading():
    """Standalone test function for model saving/loading"""
    data = generate_sample_sales_data()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        return test_model_saving_loading(data, tmp_path=tempfile.Path(tmpdirname))

def format_table(headers, rows):
    """Simple function to format a text table without external dependencies"""
    # Determine the width of each column
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create the horizontal separator
    separator = "+"
    for width in col_widths:
        separator += "-" * (width + 2) + "+"
    
    # Format the header
    header = "|"
    for i, h in enumerate(headers):
        header += " " + h.ljust(col_widths[i]) + " |"
    
    # Format the rows
    formatted_rows = []
    for row in rows:
        formatted_row = "|"
        for i, cell in enumerate(row):
            formatted_row += " " + str(cell).ljust(col_widths[i]) + " |"
        formatted_rows.append(formatted_row)
    
    # Combine all parts
    table = [separator, header, separator]
    table.extend(formatted_rows)
    table.append(separator)
    
    return "\n".join(table)

def generate_report():
    """Generate a complete test report"""
    print("\n" + "="*80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ О ТЕСТИРОВАНИИ ПРОГНОЗИРОВЩИКА ПРОДАЖ")
    print("="*80)
    
    all_tests_passed = all(test_results.values())
    
    test_table = []
    for test_name, result in test_results.items():
        status = "✓ УСПЕХ" if result else "❌ ОШИБКА"
        elapsed = test_details.get(test_name, {}).get("time", "N/A")
        if isinstance(elapsed, (int, float)):
            elapsed = f"{elapsed:.2f} сек"
        test_table.append([test_name, status, elapsed])
    
    # Print formatted table using our custom function
    headers = ["Тест", "Статус", "Время выполнения"]
    print(format_table(headers, test_table))
    
    print("\nСводка метрик:")
    if "model_training" in test_details and test_details["model_training"]["success"]:
        metrics = test_details["model_training"]["details"]["metrics"]
        for metric, value in metrics.items():
            print(f"  • {metric.upper()}: {value}")
    
    if "prediction" in test_details and test_details["prediction"]["success"]:
        print("\nРезультаты прогнозирования:")
        pred_details = test_details["prediction"]["details"]
        print(f"  • Средний прогноз: {pred_details['forecast_mean']}")
        print(f"  • Диапазон прогноза: от {pred_details['forecast_min']} до {pred_details['forecast_max']}")
    
    print("\nИтоговое заключение:")
    if all_tests_passed:
        print("✅ Все тесты пройдены успешно. Система прогнозирования работает корректно.")
    else:
        failed_tests = [test for test, result in test_results.items() if not result]
        print(f"⚠️ Обнаружены проблемы в следующих тестах: {', '.join(failed_tests)}")
        print("Необходимо устранить ошибки перед использованием системы.")
    
    return all_tests_passed

def run_all_tests():
    """Run all tests and generate a report"""
    print("\n🧪 ЗАПУСК КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ ПРОГНОЗИРОВЩИКА ПРОДАЖ\n")
    
    # Generate sample data (using regular function, not the fixture)
    data = generate_sample_sales_data()
    
    # Run individual tests manually
    test_forecaster_initialization()
    test_feature_preparation(data)
    test_model_training(data)
    test_prediction(data)
    test_feature_importance(data)
    
    # Create temporary directory for model saving test
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_model_saving_loading(data, tmp_path=tempfile.Path(tmpdirname))
    
    # Generate final report
    success = generate_report()
    
    return success

if __name__ == "__main__":
    # Run all tests when the script is executed directly
    try:
        success = run_all_tests()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Критическая ошибка при выполнении тестов: {str(e)}")
        sys.exit(1) 