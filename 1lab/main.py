import funcs
from random_variable import *

#####################################
# Константы
#####################################

# Количество генерируемых значений
AMOUNT = 50000

# Вектора возможных значений
vector_x = np.array([0, 1, 3, 7, 9])
vector_y = np.array([0, 2, 4, 6, 8])

# Теоретическая матрица распределения ДВС
#####################################
# Срабатывает не всегда, т.к сумма матрица равна 0.99999999, поэтому ниже указана константная матрица
#####################################
matrix = np.random.randint(1, 6, size=(5, 5))
matrix = matrix/matrix.sum()
matrix = np.array(matrix)

# matrix = np.array([
#     [0.01, 0.08, 0.02, 0.1, 0.02],
#     [0.06, 0.04, 0.02, 0.04, 0.06],
#     [0.02, 0.02, 0.25, 0.03, 0.02],
#     [0.04, 0.03, 0.02, 0.03, 0.04],
#     [0.01, 0.01, 0.01, 0.01, 0.01]
# ])

# Эмпирические значения векторов
empirical_x = list()
empirical_y = list()
if matrix.sum() != 1:
    raise Exception("Incorrect matrix")

# Эмпирическая матрица распределения ДСВ
empirical_matrix = np.zeros_like(matrix)

#####################################
# Основные действия
#####################################


generator = RandomVariable(vector_x, vector_y, matrix)

# Заполнение эмпирической матрицы
for i in range(AMOUNT):
    k, r = generator.generate()
    empirical_x.append(vector_x[k])
    empirical_y.append(vector_y[r])
    empirical_matrix[k][r] += 1

print("\n Probabilities matrix:\n", matrix)
print("\nEmpirical probabilities matrix:\n", empirical_matrix / AMOUNT)

# Отклоенения
x_deflection = funcs.calc_estimation(
    empirical_x, vector_x, np.sum(matrix, axis=1), 'X')

y_deflection = funcs.calc_estimation(
    empirical_y, vector_y, np.sum(matrix, axis=0), 'Y')

funcs.calc_correlation(empirical_x, empirical_y, vector_x, vector_y, matrix)

funcs.hypothesis_test(empirical_matrix / AMOUNT, matrix)

funcs.plot_histogram(empirical_x, "X")
funcs.plot_histogram(empirical_y, "Y")
funcs.common_histogram(empirical_x, empirical_y)
funcs.deflection_dependence_plot(x_deflection, y_deflection)
