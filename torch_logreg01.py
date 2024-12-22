import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Загружает и обрабатывает данные из CSV файла.

    :param file_path: Путь к CSV файлу.
    :return: Обработанный DataFrame.
    """
    data = pd.read_csv(file_path)

    # Преобразуем годы и имена в числовые метки
    data['Год'] = data['Год'].astype(int)
    data['Мальчики'] = data['Мальчики'].astype('category')
    data['Девочки'] = data['Девочки'].astype('category')

    # Преобразуем имена в числовые метки
    data['Мальчики'] = data['Мальчики'].cat.codes
    data['Девочки'] = data['Девочки'].cat.codes

    return data

class CustomDataset(Dataset):
    """
    Пользовательский класс Dataset для загрузки данных.

    :param features: Входные данные (признаки).
    :param labels: Целевые данные (метки).
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.X = features
        self.y = labels

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.int64)

class LogisticRegressionModel(nn.Module):
    """
    Модель логистической регрессии.

    :param num_classes: Количество классов для классификации.
    """

    def __init__(self, num_classes: int) -> None:
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, num_classes)  # 1 вход (Год), num_classes выходов (Имена)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def train_model(dataloader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int) -> list:
    """
    Обучает модель на заданном даталоадере.

    :param dataloader: Даталоадер с данными.
    :param model: Модель для обучения.
    :param criterion: Функция потерь.
    :param optimizer: Оптимизатор.
    :param num_epochs: Количество эпох для обучения.
    :return: Список потерь за каждую эпоху.
    """
    losses = []

    for epoch in range(num_epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())  # Сохраняем потерю

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return losses

def plot_losses(losses_male: list, losses_female: list, num_epochs: int) -> None:
    """
    Визуализирует потери в ходе обучения.

    :param losses_male: Список потерь для мальчиков.
    :param losses_female: Список потерь для девочек.
    :param num_epochs: Количество эпох.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), losses_male, label='Потери для мальчиков', color='blue')
    plt.plot(range(1, num_epochs + 1), losses_female, label='Потери для девочек', color='red')
    plt.title('Потери в ходе обучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid()
    plt.show()

# Основной блок выполнения
if __name__ == "__main__":
    # Загружаем датасет
    data = pd.read_csv('train.csv')
    # Преобразуем столбцы 'Мальчики' и 'Девочки' в категориальный тип
    data['Мальчики'] = data['Мальчики'].astype('category')
    data['Девочки'] = data['Девочки'].astype('category')

    # Определяем X и y для мальчиков и девочек
    X = data['Год'].values.astype(np.float32).reshape(-1, 1)
    y_male = data['Мальчики'].cat.codes.values.astype(np.int64)  # Используем .cat.codes для получения кодов категорий
    y_female = data['Девочки'].cat.codes.values.astype(np.int64)


    # Получаем уникальные имена для мальчиков и девочек
    male_categories = data['Мальчики'].cat.categories.tolist()
    female_categories = data['Девочки'].cat.categories.tolist()

    # Создаем экземпляр Dataset и DataLoader для мальчиков и девочек
    dataset_male = CustomDataset(X, y_male)
    dataset_female = CustomDataset(X, y_female)

    dataloader_male = DataLoader(dataset_male, batch_size=4, shuffle=True)
    dataloader_female = DataLoader(dataset_female, batch_size=4, shuffle=True)

    # Количество уникальных имен для мальчиков и девочек
    num_classes_male = len(male_categories)
    num_classes_female = len(female_categories)

    # Инициализируем модели, функции потерь и оптимизаторы
    model_male = LogisticRegressionModel(num_classes_male)
    model_female = LogisticRegressionModel(num_classes_female)

    criterion = nn.CrossEntropyLoss()  # Используем кросс-энтропию для многоклассовой классификации
    optimizer_male = optim.SGD(model_male.parameters(), lr=0.01)
    optimizer_female = optim.SGD(model_female.parameters(), lr=0.01)

    # Обучаем модель для мальчиков и девочек
    num_epochs = 100
    losses_male = train_model(dataloader_male, model_male, criterion, optimizer_male, num_epochs)
    losses_female = train_model(dataloader_female, model_female, criterion, optimizer_female, num_epochs)

    # Визуализация потерь
    plot_losses(losses_male, losses_female, num_epochs)

    # Тестирование модели на новых данных
    with torch.no_grad():
        test_years = torch.tensor([[2025], [2026]], dtype=torch.float32)  # Модель постарается предсказать имена в следующие годы
        predictions_male = model_male(test_years)
        predictions_female = model_female(test_years)
        predicted_classes_male = torch.argmax(predictions_male, dim=1)
        predicted_classes_female = torch.argmax(predictions_female, dim=1)

        predicted_names_male = [male_categories[i] for i in predicted_classes_male.numpy()]
        predicted_names_female = [female_categories[i] for i in predicted_classes_female.numpy()]

        print(f'Предсказанные имена для мальчиков в 2025 и 2026: {predicted_names_male}')
        print(f'Предсказанные имена для девочек в 2025 и 2026: {predicted_names_female}')
