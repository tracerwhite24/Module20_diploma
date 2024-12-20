import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filename: str) -> pd.DataFrame:
    data = pd.read_csv(filename)
    return data

def preprocess_data(data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    data['Label_Boys'] = data['Мальчики'].notnull().astype(int)
    data['Label_Girls'] = data['Девочки'].notnull().astype(int)

    names = data['Мальчики'].fillna('') + ' ' + data['Девочки'].fillna('')
    encoder = LabelEncoder()
    encoded_names = encoder.fit_transform(names)

    X = encoded_names.reshape(-1, 1)  # Преобразуем в двумерный массив
    y_boys = data['Label_Boys'].values
    y_girls = data['Label_Girls'].values

    return X, y_boys, y_girls


def create_model(input_dim: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Embedding(input_dim=input_dim, output_dim=8),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(model: tf.keras.Sequential, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                y_val: np.ndarray):
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    return history


def evaluate_model(model: tf.keras.Sequential, X_test: np.ndarray, y_test: np.ndarray):
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, accuracy


def plot_history(history, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title(f'Потери для {title}')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title(f'Точность для {title}')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.show()


def main():
    # Загрузка и предобработка данных
    data = load_data('train.csv')
    X, y_boys, y_girls = preprocess_data(data)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_boys_train, y_boys_test, y_girls_train, y_girls_test = train_test_split(
        X, y_boys, y_girls, test_size=0.2, random_state=42)

    # Создание и обучение модели для мальчиков
    model_boys = create_model(len(np.unique(X)))
    history_boys = train_model(model_boys, X_train, y_boys_train, X_test, y_boys_test)

    # Оценка модели для мальчиков
    loss_boys, accuracy_boys = evaluate_model(model_boys, X_test, y_boys_test)
    print(f'Потери мальчиков: {loss_boys}, Точность для мальчиков: {accuracy_boys}')

    # Создание и обучение модели для девочек
    model_girls = create_model(len(np.unique(X)))
    history_girls = train_model(model_girls, X_train, y_girls_train, X_test, y_girls_test)

    # Оценка модели для девочек
    loss_girls, accuracy_girls = evaluate_model(model_girls, X_test, y_girls_test)
    print(f'Потери для девочек: {loss_girls}, Точность для девочек: {accuracy_girls}')

    # Визуализация результатов
    plot_history(history_boys, 'мальчиков')
    plot_history(history_girls, 'девочек')


if __name__ == "__main__":
    main()



