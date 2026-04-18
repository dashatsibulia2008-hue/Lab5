import numpy as np


# 1. Функції для роботи з файлами
def save_to_file(filename, data):
    np.savetxt(filename, data, fmt='%0.4f')


def load_from_file(filename):
    return np.loadtxt(filename)


# 2. LU-розклад (Алгоритм з методички)
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)  # Одинична матриця (діагональ = 1)

    for k in range(n):
        for i in range(k, n):
            L[i, k] = A[i, k] - np.dot(L[i, :k], U[:k, k])
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - np.dot(L[k, :k], U[:k, i])) / L[k, k]
    return L, U


# 3. Розв'язок системи AX = B через LU
def solve_lu(L, U, b):
    # 1) Розв'язуємо LY = B (прямий хід)
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # 2) Розв'язуємо UX = Y (зворотний хід)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i + 1:], x[i + 1:])
    return x


# --- ОСНОВНА ЧАСТИНА ПРОГРАМИ ---
n = 10  # Для тесту візьмемо 10, у звіті можна поставити 100

# Геренація матриці (щоб була стійка, додамо значення на діагональ)
A_gen = np.random.rand(n, n) + np.eye(n) * n
x_exact = np.full(n, 2.5)  # Заданий розв'язок за умовою
b_gen = np.dot(A_gen, x_exact)

# Запис у файли
save_to_file('matrix_A.txt', A_gen)
save_to_file('vector_B.txt', b_gen)

print("Файли створено. Починаємо зчитування та розрахунок...")

# Зчитування
A = load_from_file('matrix_A.txt')
b = load_from_file('vector_B.txt')

# Розв'язок
L, U = lu_decomposition(A)
x_calc = solve_lu(L, U, b)

# Оцінка точності
norma = np.linalg.norm(x_exact - x_calc)

print(f"\nПерші 5 значень розв'язку: {x_calc[:5]}")
print(f"Норма похибки: {norma:.2e}")

# Запис результату LU
save_to_file('matrix_L.txt', L)
save_to_file('matrix_U.txt', U)