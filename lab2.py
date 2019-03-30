import pandas as pd
from scipy.stats import t, chi2
import numpy as np
from math import sqrt

def check_hypothesis(t):
    t_table = 1.9623415

    if t < t_table:
        print("Гипотеза верна")
    else:
        print("Гипотеза неверна")


df = pd.read_excel("abalone.data.xlsx", header=None)

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

data = np.asarray(x)

bounds = t.interval(0.95, len(data) - 1)
mean = np.mean(data)
stddev = np.std(data)
mean_interval = [mean + bound * stddev / sqrt(len(data)) for bound in bounds]
print('Доверительный интервал для математического ожидания:\t', mean_interval)

variance = np.var(data)
bounds = [chi2.ppf(0.95, len(data) - 1), chi2.ppf(0.05, len(data) - 1)]
var_interval = [variance * len(data) / bound for bound in bounds]
print('Доверительный интервал для дисперсии:\t', var_interval)

first_set = np.asarray(x)
second_set = np.asarray(y)

first_var = np.var(first_set)
second_var = np.var(second_set)

n = 500  # size of first selection
k = 800  # size of second selection
first_selection = first_set[2000:2000 + n]
second_selection = second_set[3000:3000 + k]

first_mean = np.mean(first_selection)
second_mean = np.mean(second_selection)

t_known = (first_mean - second_mean) / sqrt(first_var / n + second_var / k)

print("Для известных дисперсий: ", t_known)
check_hypothesis(t_known)

first_select_var = np.var(first_selection)
second_select_var = np.var(second_selection)

t_unknown = (first_mean - second_mean) / (sqrt(((n - 1) * first_select_var ** 2) + (k - 1) * second_select_var ** 2) /
                                          (n + k) * sqrt(1 / n + 1 / k))

print("Для неизвестных дисперсий: ", t_unknown)
check_hypothesis(t_unknown)
