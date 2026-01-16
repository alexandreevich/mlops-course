import logging
import pickle
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.ingestion import load_data
from src.features.engineering import add_wind_humidity_ratio

logger: logging.Logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Начало предсказания")
    config = load_config()

    # Настройка логгера
    setup_logger(config)

    # Загрузка данных
    df = load_data(config["data"]["raw_path"])
    logger.info("Данные загружены")

    # Feature engineering
    df = add_wind_humidity_ratio(df)
    logger.info("Feature engineering выполнен")

    # Подготовка признаков
    features = config["features"]["selected"] + config["features"]["engineered"]
    X = df[features]

    # Загрузка модели
    model_path = config["model"]["output_path"]
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Модель загружена из {model_path}")

    # Предсказание
    predictions = model.predict(X)
    logger.info("Предсказания выполнены")
    print(f"Первые 5 предсказаний: {predictions[:5]}")


if __name__ == "__main__":
    main()
