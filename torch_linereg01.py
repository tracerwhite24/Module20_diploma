import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

def load_and_preprocess_data(filename):
    """
    Загружает и обрабатывает данные из CSV файла.

    :param filename: str, имя файла с данными.
    :return: DataFrame, обработанный DataFrame.
    """
    data = pd.read_csv(filename)
    data['Год'] = pd.to_numeric(data['Год'], errors='coerce')
    data.dropna(subset=['Год'], inplace=True)
    return data

def prepare_data(data, target_column):
    """
    Подготавливает данные для обучения модели.

    :param data: DataFrame, исходные данные.
    :param target_column: str, название целевой переменной.
    :return: tuple, (X_train, X_test, y_train, y_test) - обучающие и тестовые наборы данных.
    """
    X = data['Год'].values.reshape(-1, 1)
    y = data[target_column].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

class SimpleModel(nn.Module):
    """
    Простая нейронная сеть с одним скрытым слоем.

    :param output_size: int, количество выходных классов.
    """
    def __init__(self, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 10)  # Скрытый слой
        self.output = nn.Linear(10, output_size)  # Выходной слой

    def forward(self, x):
        """
        Прямое распространение данных через модель.

        :param x: Tensor, входные данные.
        :return: Tensor, выходные данные модели.
        """
        x = torch.relu(self.fc(x))
        return self.output(x)

def train_model(model, X_train_tensor, y_train_tensor, criterion, optimizer, epochs=100):
    """
    Обучает модель на заданных данных.

    :param model: nn.Module, модель для обучения.
    :param X_train_tensor: Tensor, обучающие входные данные.
    :param y_train_tensor: Tensor, обучающие целевые данные.
    :param criterion: функция потерь.
    :param optimizer: оптимизатор для обновления весов модели.
    :param epochs: int, количество эпох для обучения.
    :return: list, потери на каждой эпохе.
    """
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return losses

def evaluate_model(model, X_test_tensor, y_test):
    """
    Оценивает производительность модели на тестовых данных.

    :param model: nn.Module, модель для оценки.
    :param X_test_tensor: Tensor, тестовые входные данные.
    :param y_test: array-like, истинные метки классов для тестовых данных.
    :return: float, точность модели на тестовых данных.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.numpy() == y_test).mean()

    return accuracy

def plot_losses(losses_boys, losses_girls):
    """
    Строит график потерь во время обучения.

    :param losses_boys: list, потери модели для мальчиков.
    :param losses_girls: list, потери модели для девочек.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(losses_boys, label='Loss (Мальчики)')
    plt.plot(losses_girls, label='Loss (Девочки)')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Основной код
if __name__ == "__main__":
    # Загрузка и предобработка данных
    data = load_data('train.csv')

    # Обработка данных
    data = load_and_preprocess_data('train.csv')

    # Подготовка данных для мальчиков
    X_train_boys, X_test_boys, y_train_boys, y_test_boys, label_encoder_boys = prepare_data(data, 'Мальчики')

    # Подготовка данных для девочек
    X_train_girls, X_test_girls, y_train_girls, y_test_girls, label_encoder_girls = prepare_data(data, 'Девочки')

    # Преобразование данных в тензоры PyTorch
    X_train_tensor_boys = torch.tensor(X_train_boys, dtype=torch.float32)
    y_train_boys_tensor = torch.tensor(y_train_boys, dtype=torch.long)

    X_train_tensor_girls = torch.tensor(X_train_girls, dtype=torch.float32)
    y_train_girls_tensor = torch.tensor(y_train_girls, dtype=torch.long)

    # Инициализация моделей и оптимизаторов
    model_boys = SimpleModel(len(label_encoder_boys.classes_))
    criterion_boys = nn.CrossEntropyLoss()
    optimizer_boys = optim.Adam(model_boys.parameters(), lr=0.001)

    model_girls = SimpleModel(len(label_encoder_girls.classes_))
    criterion_girls = nn.CrossEntropyLoss()
    optimizer_girls = optim.Adam(model_girls.parameters(), lr=0.001)

    # Обучение моделей
    losses_boys = train_model(model_boys, X_train_tensor_boys, y_train_boys_tensor,
                              criterion_boys, optimizer_boys)

    losses_girls = train_model(model_girls, X_train_tensor_girls, y_train_girls_tensor,
                                criterion_girls, optimizer_girls)

    # Оценка моделей
    accuracy_boys = evaluate_model(model_boys, torch.tensor(X_test_boys, dtype=torch.float32), y_test_boys)
    print(f'Accuracy (Мальчики): {accuracy_boys:.4f}')

    accuracy_girls = evaluate_model(model_girls, torch.tensor(X_test_girls, dtype=torch.float32), y_test_girls)
    print(f'Accuracy (Девочки): {accuracy_girls:.4f}')

    # Визуализация потерь
    plot_losses(losses_boys, losses_girls)

