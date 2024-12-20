import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple


def load_data(filename: str) -> pd.DataFrame:
    """Загружает данные из CSV файла и возвращает DataFrame.

    Args:
        filename (str): Путь к файлу CSV.

    Returns:
        pd.DataFrame: Загруженные данные.
    """
    try:
        return pd.read_csv(filename)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        raise


def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Преобразует данные для модели.

    Преобразует текстовые метки в числовые значения и создает целевую переменную.

    Args:
        data (pd.DataFrame): Исходные данные.

    Returns:
        X (pd.DataFrame): Признаки для модели.
        y (pd.Series): Целевая переменная.
    """
    le = LabelEncoder()
    data['Мальчики'] = le.fit_transform(data['Мальчики'])
    data['Девочки'] = le.fit_transform(data['Девочки'])

    X = data[['Мальчики', 'Девочки']]
    y = (data['Девочки'] > 0).astype(int)  # Если имя девочки, то 1, иначе 0
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Обучает модель логистической регрессии.

    Args:
        X_train (pd.DataFrame): Признаки для обучения.
        y_train (pd.Series): Целевая переменная для обучения.

    Returns:
        model (LogisticRegression): Обученная модель.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Оценивает модель и выводит метрики точности и отчет о классификации.

    Args:
        model (LogisticRegression): Обученная модель.
        X_test (pd.DataFrame): Признаки тестовой выборки.
        y_test (pd.Series): Целевая переменная тестовой выборки.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)


def plot_results(y_pred: pd.Series, y_test: pd.Series) -> None:
    """Визуализирует результаты предсказаний и Confusion Matrix.

    Args:
        y_pred (pd.Series): Предсказанные классы.
        y_test (pd.Series): Истинные классы.
    """
    plt.figure(figsize=(10, 5))

    # График распределения предсказанных классов
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_pred)
    plt.title('Распределение предсказанных классов')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    plt.xticks(ticks=[0, 1], labels=['Мальчик', 'Девочка'])

    # Построение Confusion Matrix
    plt.subplot(1, 2, 2)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Мальчик', 'Девочка'],
                yticklabels=['Мальчик', 'Девочка'])
    plt.title('Confusion Matrix')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')

    plt.tight_layout()
    plt.show()


def main():
    """Основная функция для выполнения всего процесса."""
    filename = 'train.csv'

    data = load_data(filename)
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    # Визуализация результатов
    plot_results(model.predict(X_test), y_test)


if __name__ == "__main__":
    main()


