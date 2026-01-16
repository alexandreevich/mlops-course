import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df[(df['fare_amount'] > 0)]
    return df


def train_model(df):
    X = df[['pickup_latitude', 'pickup_longitude', 'passenger_count']]
    y = df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model


if __name__ == "__main__":
    data_path = "uber.csv"
    data = load_and_prepare_data(data_path)
    train_model(data)
