import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

# 1. Загрузка и предобработка данных
df = pd.read_csv("uber.csv")
df = df.dropna()
df = df[
    (df["fare_amount"] > 0) & (df["passenger_count"] > 0) & (df["passenger_count"] <= 6)
]

# 2. Выбор признаков и цели
X = df[
    [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "passenger_count",
    ]
]
y = df["fare_amount"]

# 3. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Обучение модели
model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Оценка модели
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Модель обучена. RMSE на тестовой выборке: {rmse:.3f}")
