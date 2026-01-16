from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Tuple, Any


def train_model(X, y, random_state=42, test_size=0.2) -> Tuple[Any, Any, Any]:
    """
    Обучает модель линейной регрессии.

    Args:
        X: Матрица признаков.
        y: Вектор целевой переменной.
        random_state: Сид для воспроизводимости.
        test_size: Доля тестовой выборки.

    Returns:
        Обученная модель, X_test, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test
