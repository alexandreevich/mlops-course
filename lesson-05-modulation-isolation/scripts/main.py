import os
import pickle
import logging
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.ingestion import load_data
from src.data.preprocessing import preprocess_data
from src.features.engineering import add_wind_humidity_ratio
from src.models.training import train_model
from src.models.evaluation import evaluate_model

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Начало обучения модели")
    config = load_config()

    # Настройка логгера
    setup_logger(config)

    # Загрузка данных
    df = load_data(config["data"]["raw_path"])
    logger.info("Данные загружены")

    # Предобработка
    df = preprocess_data(df)
    logger.info("Данные предобработаны")

    # Feature engineering
    df = add_wind_humidity_ratio(df)
    logger.info("Feature engineering выполнен")

    # Подготовка признаков и таргета
    features = config["features"]["selected"] + config["features"]["engineered"]
    target = config["target"]

    X = df[features]
    y = df[target]

    # Обучение
    model, X_test, y_test = train_model(
        X,
        y,
        random_state=config["model"]["random_state"],
        test_size=config["model"]["test_size"],
    )
    logger.info("Модель обучена")

    # Оценка
    score = evaluate_model(model, X_test, y_test)
    logger.info(f"Model R^2 score: {score:.4f}")

    # Сохранение модели
    os.makedirs(os.path.dirname(config["model"]["output_path"]), exist_ok=True)
    with open(config["model"]["output_path"], "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Модель сохранена в {config['model']['output_path']}")


if __name__ == "__main__":
    main()
