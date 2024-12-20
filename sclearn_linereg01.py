import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data(filename: str) -> pd.DataFrame:
    """
    Загружает данные из указанного файла.

    :param filename: Имя файла для загрузки данных.
    :return: DataFrame с загруженными данными.
    """
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Предобрабатывает данные, переименовывает столбцы и добавляет случайные значения.

    :param data: DataFrame с исходными данными.
    :return: DataFrame с предобработанными данными.
    """
    data.columns = ['Мальчики', 'Девочки', 'Год']
    np.random.seed(0)
    data['Количество'] = np.random.randint(1, 1000, size=len(data))
    data['Год'] = pd.to_numeric(data['Год'], errors='coerce')
    return data


def filter_data(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Фильтрует данные по имени.

    :param data: DataFrame с предобработанными данными.
    :param name: Имя для фильтрации данных.
    :return: Отфильтрованный DataFrame.
    """
    return data[data['Мальчики'] == name]


def create_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """
    Создает и обучает модель линейной регрессии.

    :param X: Независимая переменная.
    :param y: Зависимая переменная.
    :return: Обученная модель линейной регрессии.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def plot_results(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray, name: str) -> None:
    """
    Строит график фактических и предсказанных значений.

    :param X_test: Тестовые данные (независимая переменная).
    :param y_test: Фактические значения (зависимая переменная).
    :param y_pred: Предсказанные значения.
    :param name: Имя для заголовка графика.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Фактические данные')
    plt.scatter(X_test, y_pred, color='red', label='Предсказанные данные')
    plt.plot(X_test, y_pred, color='green', linewidth=2, label='Линейная регрессия')
    plt.title(f'Линейная регрессия для имени {name}')
    plt.xlabel('Год')
    plt.ylabel('Количество упоминаний')
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """Основная функция для выполнения всего процесса."""
    filename = 'train.csv'

    # Загружаем и обрабатываем данные
    data = load_data(filename)
    data = preprocess_data(data)

    name_to_analyze = 'Александр'

    # Фильтруем данные по имени
    name_data = filter_data(data, name_to_analyze)

    if name_data.empty:
        print(f"Нет данных для имени {name_to_analyze}.")
        return

    # Определяем переменные для модели
    X = name_data[['Год']]
    y = name_data['Количество']

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Создаем и обучаем модель
    model = create_model(X_train, y_train)

    # Прогнозируем значения на тестовых данных
    y_pred = model.predict(X_test)

    # Визуализируем результаты
    plot_results(X_test, y_test, y_pred, name_to_analyze)

    # Оцениваем модель
    print(f'Коэффициенты: {model.coef_}')
    print(f'Перехват: {model.intercept_}')


if __name__ == "__main__":
    main()
