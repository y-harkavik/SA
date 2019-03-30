import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress


def t_St(r, n):
    return (r * math.sqrt(n - 2)) / math.sqrt(1 - r * r)


os.chdir('D:/')

df = pd.read_excel("abalone.data.xlsx", header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3,
                                                    random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

a = df.iloc[:, 0].values
b = df.iloc[:, 1].values

_, _, r_kof, _, _ = linregress(a, b)
# r_kof = math.fabs(r_kof)

print('Математическое ожидание: ' + str(np.mean(a)) + "; " + str(np.mean(b)))
print('Дисперсия: ' + str(np.var(a)) + "; " + str(np.var(b)))
print('СКО: ' + str(np.std(a)) + "; " + str(np.std(b)))
print('Коэффициент корреляции: ' + str(r_kof))
print('T-статистика Cтьюдента: ' + str(t_St(r_kof, a.size)))
print('Доверительная вероятность: ' + str(0.001))  # !
print('Число степеней свободы: ' + str(a.size))
print('Табличное значение Т-Стьюдента: ' + str(3.291))  # !
print('Теснота связи: очень сильная прямая зависимость.')

# print(linregress(a,b))

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('System analysis')
plt.xlabel('Length')
plt.ylabel("Diameter")
plt.show()
