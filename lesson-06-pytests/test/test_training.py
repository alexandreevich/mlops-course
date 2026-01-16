# tests/test_training.py
from unittest.mock import patch
from src.training import train_and_save_model


@patch("src.training.joblib.dump")
def test_train_and_save_model(mock_dump, sample_training_data):
    """
    Проверяем, что модель обучается и передаётся в joblib.dump.

    Тест полностью изолирован от файловой системы: вместо реального сохранения
    используется mock-объект, который проверяет факт вызова с правильными аргументами.
    """
    # Подготовка
    X, y = sample_training_data
    model_path = "test_model.pkl"

    # Действие
    train_and_save_model(X, y, model_path)

    # Проверка
    assert mock_dump.called_once_with(model_path)
    assert hasattr(mock_dump.call_args[0][0], "predict")
