# Этот файл принимает на вход новые данные и делает предсказание стоимости

import joblib  # для загрузки модели
import pandas as pd

# Модель была предварительно обучена и сохранена в model.joblib

MODEL_PATH = "model.joblib"


def predict_fare(data: pd.DataFrame):
    """Функция для предсказания стоимости по новым данным."""
    # Загружаем модель
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Ошибка: файл модели не найден по пути {MODEL_PATH}")
        return None

    # Убедимся, что колонки на месте
    required_cols = ['pickup_latitude', 'pickup_longitude', 'passenger_count']
    if not all([col in data.columns for col in required_cols]):
        print("Ошибка: в данных отсутствуют необходимые колонки.")
        return None

    predictions = model.predict(data[required_cols])
    return predictions


if __name__ == "__main__":
    # Пример данных для предсказания
    test_data = pd.DataFrame(
        {
            'pickup_latitude': [40.7128],
            'pickup_longitude': [-74.0060],
            'passenger_count': [1],
        }
    )

    predictions = predict_fare(test_data)
    if predictions is not None:
        print("Предсказанная стоимость поездки:")
        print(predictions)
