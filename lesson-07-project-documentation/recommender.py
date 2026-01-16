import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class RecommenderModel:
    """
    Класс модели рекомендаций треков на основе косинусного сходства признаков.

    Включает методы для обучения модели на датасетах с характеристиками треков
    и получения рекомендаций по названию трека.
    """

    def __init__(self):
        """
        Инициализация модели.

        Атрибуты:
            data_encoded (pd.DataFrame или None): Закодированные данные треков после обучения.
            similarity_matrix (np.ndarray или None): Матрица косинусных сходств между треками.
        """
        self.data_encoded = None
        self.similarity_matrix = None

    def fit(self, df: pd.DataFrame, df_year: pd.DataFrame):
        """
        Обучает модель на входных данных.

        Объединяет данные с информацией о годе выпуска, кодирует жанры,
        нормализует числовые признаки и рассчитывает матрицу сходств.

        Args:
            df (pd.DataFrame): Основной датасет с треками и признаками.
            df_year (pd.DataFrame): Датасет с информацией о годе выпуска треков.
        """
        df_year = df_year[["id", "year"]].copy()
        df_year["track_id"] = df_year["id"]
        df_year.drop(columns="id", inplace=True)

        df = pd.merge(df, df_year, on="track_id")

        xtab_song = pd.crosstab(df["track_id"], df["track_genre"]) * 2
        xtab_song.reset_index(inplace=True)

        df_distinct = (
            df.drop_duplicates("track_id")
            .sort_values("track_id")
            .reset_index(drop=True)
        )
        data_encoded = pd.concat(
            [df_distinct, xtab_song.drop(columns=["track_id"])], axis=1
        )

        numerical_features = [
            "explicit",
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "year",
        ]
        scaler = MinMaxScaler()
        data_encoded[numerical_features] = scaler.fit_transform(
            data_encoded[numerical_features]
        )

        self.data_encoded = data_encoded
        self.similarity_matrix = cosine_similarity(
            data_encoded[numerical_features + list(xtab_song.columns[1:])]
        )

    def recommend(self, track_title: str, N: int = 5) -> list[dict]:
        """
        Генерирует рекомендации треков, похожих на заданный трек.

        Args:
            track_title (str): Название трека для поиска похожих.
            N (int, optional): Количество рекомендуемых треков. По умолчанию 5.

        Returns:
            list of dict: Список рекомендованных треков с ключами:
                'track_name', 'artists', 'album_name'.
                Пустой список, если трек не найден.
        """
        indices = pd.Series(
            self.data_encoded.index, index=self.data_encoded["track_name"]
        ).drop_duplicates()

        if track_title not in indices:
            return []

        idx = indices[track_title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : N + 1]

        song_indices = [i[0] for i in sim_scores]
        recommended = self.data_encoded[["track_name", "artists", "album_name"]].iloc[
            song_indices
        ]

        return recommended.to_dict(orient="records")


def train_and_save_model():
    """
    Обучает модель на подготовленных данных и сохраняет её на диск.

    Загружает датасеты из CSV, обучает объект RecommenderModel и
    сохраняет сериализованный объект в файл 'model.pkl'.

    Returns:
        None
    """
    df = pd.read_csv("dataset.csv").drop(columns="Unnamed: 0")
    df_year = pd.read_csv("data.csv")
    model = RecommenderModel()
    model.fit(df, df_year)
    joblib.dump(model, "model.pkl")


def load_model() -> RecommenderModel:
    """
    Загружает сохранённую модель рекомендаций из файла.

    Returns:
        RecommenderModel: Загруженный объект модели.
    """
    return joblib.load("model.pkl")


def recommend_songs(
    model: RecommenderModel, track_title: str, N: int = 5
) -> list[dict]:
    """
    Возвращает список рекомендованных треков для заданного трека.

    Args:
        model (RecommenderModel): Обученная модель рекомендаций.
        track_title (str): Название трека, для которого нужны рекомендации.
        N (int, optional): Количество рекомендаций. По умолчанию 5.

    Returns:
        list of dict: Список рекомендованных треков с полями
        'track_name', 'artists', 'album_name'.
    """
    return model.recommend(track_title, N)
