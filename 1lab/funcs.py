from matplotlib import pyplot as plt

import math
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import chi2

# Критерий Мизеса для уровня значимости = 0.05
MISES = 0.461

# Квантили из таблицы значений функции Лапласа
quantiles = {
    "90": 1.64,
    "92": 1.75,
    "95": 1.96,
    "97": 2.17,
    "99": 2.54,
}


# Вычисление корреляции
def calc_correlation(x_empirical, y_empirical, x_values, y_values, p):
    # Вероятности
    x_probabilities = np.sum(p, axis=1)
    y_probabilities = np.sum(p, axis=0)
    # Мат ожидание
    x_theoretical_average = np.sum(x_values * x_probabilities)
    y_theoretical_average = np.sum(y_values * y_probabilities)
    # Дисперсия
    x_theoretical_deflection = np.sum((x_values ** 2) * x_probabilities) - x_theoretical_average ** 2
    y_theoretical_deflection = np.sum((y_values ** 2) * y_probabilities) - y_theoretical_average ** 2
    theoretical_covariation = 0
    length = len(p[0])

    for index in range(len(p.ravel()) - 1):
        i = index // length
        j = index % length
        # Ковариация
        theoretical_covariation += p[i][j] * (x_values[i] - x_theoretical_average) * (
                    y_values[j] - y_theoretical_average)

    print(f'\nTheoretical covariation: {theoretical_covariation}')
    # Коэффицент корреляции
    print(
        f'Theoretical correlation coefficient: {theoretical_covariation / math.sqrt(x_theoretical_deflection * y_theoretical_deflection)}')
    # Мат ожидание
    x_empirical_average = np.sum(x_empirical) / len(x_empirical)
    y_empirical_average = np.sum(y_empirical) / len(y_empirical)
    # Дисперсия
    x_empirical_deflection = (np.sum(((x_empirical - x_empirical_average) ** 2))) / len(x_empirical)
    y_empirical_deflection = (np.sum(((y_empirical - y_empirical_average) ** 2))) / len(y_empirical)
    # Эмпирическая ковариация
    empirical_covariation = np.sum((x_empirical - x_empirical_average) * (y_empirical - y_empirical_average)) / (
        len(x_empirical))
    # Эмпирический коэффицент корреляии
    empirical_correlation_coefficient = empirical_covariation / (
        math.sqrt(x_empirical_deflection * y_empirical_deflection))

    print(f'\nEmpirical covariation: {empirical_covariation}')
    print(f'Empirical correlation coefficient: {empirical_correlation_coefficient}')

    # t - критерий Стьюдента(табличное значение) для a = 0.05
    # a - уровень значимости
    t = 1.96
    # Доверительный интервал для коэффицента корреляции
    left_border = empirical_correlation_coefficient - t * (
            math.sqrt(1 - empirical_correlation_coefficient ** 2) / len(x_empirical) - 2)
    right_border = empirical_correlation_coefficient + t * (
            math.sqrt(1 - empirical_correlation_coefficient ** 2) / len(x_empirical) - 2)
    print(f'\nConfidence interval of correlation coefficient: '
          f'{left_border} : '
          f'{right_border}\n')


# Вычисление оценок
def calc_estimation(empirical_values, values, p, vector):
    deflection_length = []

    # Математическое ожидание
    empirical_average = np.sum(empirical_values) / len(empirical_values)
    theoretical_average = np.sum(values * p)
    # Дисперсия
    empirical_deflection = (np.sum(((empirical_values - empirical_average) ** 2))) / len(empirical_values)
    theoretical_deflection = np.sum((values ** 2) * p) - theoretical_average ** 2

    print(f'\n{vector} component point estimates:')
    print(f'Theoretical (M({vector})): {theoretical_average}')
    print(f'Empirical (M({vector})): {empirical_average}')

    for quantile in quantiles.items():
        # Точность оценки
        deflection = quantile[1] * math.sqrt(empirical_deflection) / math.sqrt(len(empirical_values))
        # Коэффицент доверия 0.95
        if quantile[0] == "95":
            # Доверительный интервал
            print(f'\nConfidence interval of M({vector})'
                  f' (significance level = 0.{quantile[0]})): '
                  f'({empirical_average - deflection} :'
                  f' {empirical_average + deflection}) ')
            print(f'Confidence interval length: {2 * deflection}')
        deflection_length.append(deflection)

    print(f'\nTheoretical (D({vector})): {theoretical_deflection}')
    print(f'Empirical (D({vector})): {empirical_deflection}')

    # Объём выборки
    n = len(empirical_values)
    # Исправленное среднеквадратичное ожидание
    S = math.sqrt((np.sum(((empirical_values - empirical_average) ** 2))) / n)
    # Критические точки хи-квадрат распределения
    xi_left = fsolve(lambda x: chi2.cdf(x, n - 1) - 0.05, n)[0]
    xi_right = fsolve(lambda x: chi2.cdf(x, n - 1) - (1 - 0.05), n)[0]
    # Границы доверительного интервала
    left_border = (math.sqrt(n - 1) * S) / math.sqrt(xi_right)
    right_border = (math.sqrt(n - 1) * S) / math.sqrt(xi_left)

    # Доверительный интервал дисперсии
    print(
        f'Confidence interval of D({vector}): '
        f'{left_border ** 2} : '
        f'{right_border ** 2}')
    # Среднеквадратичное отклонение
    print(f'\nTheoretical σ({vector}): {math.sqrt(theoretical_deflection)}')
    print(f'Empirical σ({vector}): {math.sqrt(empirical_deflection)}')
    # Доверительный инетрвал среднеквадратичного отклонения
    print(
        f'Confidence interval of σ({vector}): '
        f'{left_border} : '
        f'{right_border}')

    return deflection_length


# Проверка гипотезы
def hypothesis_test(p_theoretical, p_empirical):
    # Объём выборки
    n = len(p_theoretical)
    m = len(p_theoretical[0])
    mises = 0
    for i in range(n):
        for n in range(m):
            # Критерий Крамера — Мизеса — Смирнова
            mises += (p_empirical[i][n] - p_theoretical[i][n]) ** 2
    mises += (1 / (12 * n * m))

    print(f'Empirical mises criteria: {mises}')
    if mises < MISES:
        print(f'No reason to reject H0 hypothesis({mises} < {MISES})')
    else:
        print(f'There is reason to reject H0 hypothesis ({mises} > {MISES})')


def plot_histogram(values, component_name):
    plt.hist(values, color='Red')
    plt.title(component_name + " component")
    plt.xlabel("Value")
    plt.ylabel("Amount")
    plt.legend((component_name + " component", ''))
    plt.show()


def common_histogram(X, Y):
    hist, x_edges, y_edges = np.histogram2d(X, Y)
    hist = hist.T
    X, Y = np.meshgrid(x_edges, y_edges)
    plt.title("Common 2d histogram of empiric values")
    plt.pcolormesh(X, Y, hist)
    plt.show()


def deflection_dependence_plot(x_deviation_lengths, y_deviation_lengths):
    plt.title(f"The dependence of the confidence interval\nvalue on the level of significance")

    plt.ylabel("Level of significance")
    plt.xlabel("Confidence interval length")

    plt.plot(x_deviation_lengths, quantiles.keys(), label=f"X component")
    plt.plot(y_deviation_lengths, quantiles.keys(), label=f"Y component")
    plt.legend()
    plt.show()
