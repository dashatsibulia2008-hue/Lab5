import numpy as np


def generate_and_save():
    n = 100
    # 1. Генерує матрицю А
    A = np.random.rand(n, n)
    # Забезпечує діагональне переважання (важливо для збіжності)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1

    # 2. Задає точний розв'язок (x = 2.5)
    x_exact = np.full(n, 2.5)

    # 3. Обчислює вектор вільних членів b = A * x
    b = A @ x_exact

    # 4. Записує в текстові файли
    np.savetxt('matrix_A.txt', A)
    np.savetxt('vector_b.txt', b)
    print("Файли matrix_A.txt та vector_b.txt створено успішно!")


if __name__ == "__main__":
    generate_and_save()