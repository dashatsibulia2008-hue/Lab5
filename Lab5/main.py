import numpy as np
import matplotlib.pyplot as plt

# 1. Функція та параметри
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

a, b = 0, 10
n_base = 100

def get_integral(f, a, b, n, method='simpson'):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    if method == 'trap':
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    elif method == 'simpson':
        return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))

# Обчислення для різних n для графіка похибки
n_values = np.arange(10, 500, 10)
errors = [abs(get_integral(M, a, b, n) - get_integral(M, a, b, n*2)) for n in n_values]

# --- Візуалізація (3 ГРАФІКИ) ---
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Графік 1: Функція та інтеграл (площа)
t_plot = np.linspace(a, b, 500)
axs[0].plot(t_plot, M(t_plot), 'b', lw=2)
axs[0].fill_between(t_plot, M(t_plot), color='skyblue', alpha=0.4)
axs[0].set_title("1. Область інтегрування M(t)")
axs[0].grid(True)

# Графік 2: Візуалізація методу трапецій (на малій кількості n)
n_demo = 10
t_demo = np.linspace(a, b, n_demo + 1)
axs[1].plot(t_plot, M(t_plot), 'r--', alpha=0.6)
for i in range(n_demo):
    xs = [t_demo[i], t_demo[i], t_demo[i+1], t_demo[i+1]]
    ys = [0, M(t_demo[i]), M(t_demo[i+1]), 0]
    axs[1].fill(xs, ys, 'orange', edgecolor='black', alpha=0.3)
axs[1].set_title(f"2. Метод трапецій (n={n_demo})")
axs[1].grid(True)

# Графік 3: Похибка від кількості розбиттів n
axs[2].plot(n_values, errors, 'g', lw=2)
axs[2].set_yscale('log')
axs[2].set_title("3. Графік похибки (Log scale)")
axs[2].set_xlabel("Кількість кроків n")
axs[2].grid(True, which="both", ls="-")

plt.tight_layout()
plt.show()

# Вивід результатів у консоль
print(f"Результат (Сімпсон, n={n_base}): {get_integral(M, a, b, n_base):.5f}")