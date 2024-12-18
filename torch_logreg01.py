import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv('train.csv')

# Преобразуем годы и имена в числовые метки
data['Год'] = data['Год'].astype(int)
data['Мальчики'] = data['Мальчики'].astype('category')
data['Девочки'] = data['Девочки'].astype('category')

# Получаем уникальные категории для мальчиков и девочек
male_categories = data['Мальчики'].cat.categories
female_categories = data['Девочки'].cat.categories

# Преобразуем имена в числовые метки
data['Мальчики'] = data['Мальчики'].cat.codes
data['Девочки'] = data['Девочки'].cat.codes

# Определяем X и y для мальчиков и девочек
X = data['Год'].values.astype(np.float32).reshape(-1, 1)
y_male = data['Мальчики'].values.astype(np.int64)
y_female = data['Девочки'].values.astype(np.int64)

# Определяем пользовательский класс Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.X = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# Создаем экземпляр Dataset и DataLoader для мальчиков и девочек
dataset_male = CustomDataset(X, y_male)
dataset_female = CustomDataset(X, y_female)

dataloader_male = DataLoader(dataset_male, batch_size=4, shuffle=True)
dataloader_female = DataLoader(dataset_female, batch_size=4, shuffle=True)

# Определяем модель логистической регрессии
class LogisticRegressionModel(nn.Module):
    def __init__(self, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(1, num_classes)  # 1 вход (Год), num_classes выходов (Имена)

    def forward(self, x):
        return self.linear(x)

# Количество уникальных имен для мальчиков и девочек
num_classes_male = len(male_categories)
num_classes_female = len(female_categories)

# Инициализируем модели, функции потерь и оптимизаторы
model_male = LogisticRegressionModel(num_classes_male)
model_female = LogisticRegressionModel(num_classes_female)

criterion = nn.CrossEntropyLoss()  # Используем кросс-энтропию для многоклассовой классификации
optimizer_male = optim.SGD(model_male.parameters(), lr=0.01)
optimizer_female = optim.SGD(model_female.parameters(), lr=0.01)

# Списки для хранения потерь
losses_male = []
losses_female = []

# Обучаем модель для мальчиков
num_epochs = 100
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader_male:
        optimizer_male.zero_grad()
        outputs = model_male(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_male.step()

    losses_male.append(loss.item())  # Сохраняем потерю

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Male Loss: {loss.item():.4f}')

# Обучаем модель для девочек
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader_female:
        optimizer_female.zero_grad()
        outputs = model_female(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_female.step()

    losses_female.append(loss.item())  # Сохраняем потерю

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Female Loss: {loss.item():.4f}')

# Визуализация потерь
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs + 1), losses_male, label='Потери для мальчиков', color='blue')
plt.plot(range(1, num_epochs + 1), losses_female, label='Потери для девочек', color='red')
plt.title('Потери в ходе обучения')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid()
plt.show()

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