import numpy as np


# Функції зчитування та обчислень (згідно з п. 2 ходу роботи)
def load_data():
    A = np.loadtxt('matrix_A.txt')
    b = np.loadtxt('vector_b.txt')
    return A, b


def vector_norm(v):
    return np.max(np.abs(v))


def matrix_norm(A):
    return np.linalg.norm(A, ord=np.inf)


# 1. Метод простої ітерації
def simple_iteration(A, b, eps=1e-14):
    n = len(b)
    tau = 1.0 / matrix_norm(A)  # параметр tau
    x = np.ones(n)  # початкове наближення
    for k in range(10000):
        x_next = x - tau * (A @ x - b)
        if vector_norm(x_next - x) < eps:
            return x_next, k
        x = x_next
    return x, 10000


# 2. Метод Якобі
def jacobi_method(A, b, eps=1e-14):
    n = len(b)
    x = np.ones(n)
    for k in range(10000):
        x_next = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_next[i] = (b[i] - s) / A[i, i]
        if vector_norm(x_next - x) < eps:
            return x_next, k
        x = x_next.copy()
    return x, 10000


# 3. Метод Зейделя
def gauss_seidel(A, b, eps=1e-14):
    n = len(b)
    x = np.ones(n)
    for k in range(10000):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if vector_norm(x - x_old) < eps:
            return x, k

    return x, 10000


# ЗАПУСК
if __name__ == "__main__":
    A, b = load_data()

    for name, method in [("Проста ітерація", simple_iteration),
                         ("Якобі", jacobi_method),
                         ("Зейдель", gauss_seidel)]:
        sol, iters = method(A, b)
        print(f"{name}: знайдено за {iters} ітерацій. Перші 3 компоненти: {sol[:3]}")