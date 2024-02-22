import numpy as np
import matplotlib.pyplot as plt


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

variances_theta_hat = np.diag(F_inverse)
variances_theta_hat_distributed = np.diag(F_inverse_distributed)


x = np.linspace(-1, 1, 100)
eta1 = np.polyval(np.flip(theta), x)
eta2 = np.polyval(np.flip(theta_hat), x)
eta2_distributed = np.polyval(np.flip(theta_hat_distributed), x)
F = matrix[0]
F_distributed = matrix_distributed[0]
eta3 = F[0] + F[1]*x + F[2]*x**2 + F[3]*x**3 + F[4]*x**4 + F[5]*x**5
eta3_distributed = F_distributed[0] + F_distributed[1]*x + F_distributed[2]*x**2 + F_distributed[3]*x**3 + F_distributed[4]*x**4 + F_distributed[5]*x**5

# Plot eta1 and eta2
plt.figure(figsize=(14, 7))
plt.gca().axhline(0, color='black', linewidth=2)
plt.gca().axvline(0, color='black', linewidth=2)
plt.plot(x, eta1, label='eta1: True Polynomial', color='red')
plt.plot(x, eta2, label='eta2: Estimated Polynomial', color='blue')
plt.title('Comparison of eta1 and eta2')
plt.xlabel('x')
plt.ylabel('Polynomial value')
plt.legend()
plt.grid(True)
plt.show()

# Plot eta1 and eta2_distributed
plt.figure(figsize=(14, 7))
plt.gca().axhline(0, color='black', linewidth=2)
plt.gca().axvline(0, color='black', linewidth=2)
plt.plot(x, eta1, label='eta1: True Polynomial', color='red')
plt.plot(x, eta2_distributed, label='eta2: Estimated Polynomial Distributed', color='blue')
plt.title('Comparison of eta1 and eta2_distributed')
plt.xlabel('x')
plt.ylabel('Polynomial value')
plt.legend()
plt.grid(True)
plt.show()

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
print(f"Порівняння дисперсії МНК-оцінок: {variances_theta_hat}\n")
print(f"Порівняння дисперсії МНК-оцінок рівномірно розподілених: {variances_theta_hat_distributed}\n\n")
