import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import norm, t


def generate_random_points_satisfying_sum(n, threshold=0.1):
    while True:
        points = np.random.uniform(-1, 1, n - 1)
        last_point = -points.sum()
        if -1 <= last_point <= 1 and np.abs(last_point) >= threshold:
            points = np.append(points, last_point)
            if all(np.abs(points) >= threshold):
                np.random.shuffle(points)
                return points


def generate_random_points_distributed(n):
    x = np.random.uniform(-1, 1, n)
    return x


def create_adjusted_design_matrix(x, flag=False):
    n = len(x)
    F = np.zeros((n, 6))
    F[:, 0] = 1.0
    for i in range(1, 6):
        F[:, i] = x**i

    if flag:
        F_formatted = np.array2string(F, formatter={'float_kind': lambda y: "%.1f" % y if y == 1.0 else "%.9f" % y}, separator=' ')
        return F_formatted
    else:
        return F


n = 10
points = generate_random_points_satisfying_sum(n)
matrix_formatted = create_adjusted_design_matrix(points, True)
matrix = create_adjusted_design_matrix(points, False)

points_distributed = generate_random_points_distributed(n)
matrix_formatted_distributed = create_adjusted_design_matrix(points_distributed, True)
matrix_distributed = create_adjusted_design_matrix(points_distributed, False)


epsilon = np.random.normal(0, 1, n)

theta = np.array([1, -2, 2, 3.7, 5.1, -1])
Y_theta = matrix.dot(theta) + epsilon
Y_theta_distributed = matrix_distributed.dot(theta) + epsilon

F_transpose = np.transpose(matrix)
F_inverse = np.linalg.inv(F_transpose.dot(matrix))
theta_hat = F_inverse.dot(F_transpose).dot(Y_theta)

F_transpose_distributed = np.transpose(matrix_distributed)
F_inverse_distributed = np.linalg.inv(F_transpose_distributed.dot(matrix_distributed))
theta_hat_distributed = F_inverse_distributed.dot(F_transpose_distributed).dot(Y_theta_distributed)



# Код обчислення другої лаби, зверху це кусок першої лаби
alpha = 0.05
alpha_half = alpha / 2
quantile_value = stats.norm.ppf(alpha_half)

m = 6
t_quantile_value = stats.t.ppf(alpha_half, m)

sigma3 = -2
a = 0
var_theta_hat = 1 * F_inverse[1, 1]
se_theta_hat = np.sqrt(var_theta_hat)
u_stat_corrected = (theta_hat[1] - a) / se_theta_hat

residuals = Y_theta - matrix.dot(theta_hat)
mse = np.sum(residuals**2) / (n - m)
standard_errors = np.sqrt(np.diag(F_inverse) * mse)
t_statistics = (theta_hat - 0) / standard_errors
t_stat_theta_2 = t_statistics[1]
t_critical_value = stats.t.ppf(1 - alpha_half, df=n-6)

p_value_corrected_t = stats.t.sf(np.abs(t_stat_theta_2), df=n-6) * 2
alpha_acceptance = p_value_corrected_t / 2
alpha_rejection = 1 - alpha_acceptance



print(f"Набір точок:\n{points}\n")
print(f"Рівномірно розподілений набір точок:\n{points_distributed}\n\n")
print(f"Матриця плану експерименту:\n{matrix_formatted}\n")
print(f"Матриця плану експерименту рівномірно розподілена:\n{matrix_formatted_distributed}\n\n")
print(f"Вектор похибок eps: {epsilon}\n")
print(f"Вектор результатів спостережень: {Y_theta}\n")
print(f"Вектор результатів спостережень рівномірно розподілений: {Y_theta_distributed}\n\n")
print(f"Матриця для пошуку МНК-оцінки:\n{F_inverse}\n")
print(f"Матриця для пошуку МНК-оцінки рівномірно розподілена:\n{F_inverse_distributed}\n\n")
print(f"МНК-оцінка: {theta_hat}\n")
print(f"МНК-оцінка рівномірно розподілена: {theta_hat_distributed}\n\n")



# Код виводу результуючих значень другої лаби, зверху це кусок виводів першої лаби
print(f"Значення квантиля нормального розподілу: {quantile_value}")
print(f"Значення квантиля розподілу Стьюдента: {t_quantile_value}")
print(f"Значення статистики u: {u_stat_corrected}")
print(f"Значення статистики t: {t_stat_theta_2}")
print(f"Критична точка: {t_critical_value}")
print(f"Гіпотеза приймається, якщо α в проміжку: [{alpha_acceptance}; {alpha_rejection}]")


