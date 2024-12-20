import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filename: str) -> pd.DataFrame:
    """
    Загрузка данных из указанного файла.

    :param filename: Имя файла для загрузки данных.
    :return: DataFrame с загруженными данными.
    """
    data = pd.read_csv(filename)

    if data.empty:
        raise ValueError("Данные не загружены или файл пуст.")

    return data


def preprocess_data(data: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Предобработка данных: объединение имен и создание меток.

    :param data: DataFrame с исходными данными.
    :return: Tuple (X, y) - массивы входных данных и меток.
    """
    data['Label'] = data['Мальчики'].notnull().astype(int)
    names = data['Мальчики'].fillna('') + ' ' + data['Девочки'].fillna('')

    if names.isnull().any():
        raise ValueError("Объединенные имена содержат пустые значения.")

    encoder = LabelEncoder()
    encoded_names = encoder.fit_transform(names)

    X = encoded_names.reshape(-1, 1)  # Преобразуем в двумерный массив
    y = data['Label'].values

    return X, y


def create_model(input_dim: int) -> tf.keras.Sequential:
    """
    Создание модели нейронной сети.

    :param input_dim: Размер входных данных.
    :return: Скомпилированная модель Keras.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=8),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def plot_history(history):
    """
    Построение графиков потерь и точности.

    :param history: История обучения модели.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Потери')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.show()


def main():
    # Загрузка и предобработка данных
    data = load_data('train.csv')
    X, y = preprocess_data(data)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение модели
    model = create_model(input_dim=len(np.unique(X)))

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Оценка модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Построение графиков потерь и точности
    plot_history(history)


if __name__ == "__main__":
    main()
