from typing import Any
from sklearn.metrics import r2_score


def evaluate_model(model: Any, X_test: Any, y_test: Any) -> float:
    """
    Оценивает качество модели с помощью R^2 метрики.

    Args:
        model: Обученная модель.
        X_test: Признаки тестовой выборки.
        y_test: Целевая переменная тестовой выборки.

    Returns:
        R^2 score.
    """
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    return score
