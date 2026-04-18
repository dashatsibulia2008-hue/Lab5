import numpy as np
import matplotlib.pyplot as plt

# 1. Визначення функції та її аналітичної похідної
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_analytical(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# 2. Чисельне диференціювання (Центральна різниця)
def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

t0 = 1.0
exact_val = dM_analytical(t0)

# Дослідження залежності похибки від кроку h
h_values = np.logspace(-6, -1, 50) # кроки від 10^-6 до 10^-1
errors = [abs(central_diff(M, t0, h) - exact_val) for h in h_values]
h_opt = h_values[np.argmin(errors)] # Знаходження оптимального h

# 3-7. Методи покращення точності для h = 0.001
h = 0.001
d_h = central_diff(M, t0, h)      # y'(h)
d_2h = central_diff(M, t0, 2*h)   # y'(2h)
d_4h = central_diff(M, t0, 4*h)   # y'(4h)

# Метод Рунге-Ромберга
d_RR = d_h + (d_h - d_2h) / 3

# Метод Ейткена
d_E = (d_2h**2 - d_4h * d_h) / (2 * d_2h - (d_4h + d_h))
p_order = np.log2(abs((d_4h - d_2h) / (d_2h - d_h))) # Порядок точності

#Візуалізація (як на фото)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# Графік функції M(t)
t_range = np.linspace(0, 20, 400)
axs[0, 0].plot(t_range, M(t_range), 'b-', linewidth=1)
axs[0, 0].set_title("Функція M(t)")
axs[0, 0].grid(True)

# Графік похідної M'(t)
axs[0, 1].plot(t_range, dM_analytical(t_range), 'g-', linewidth=1, label="M'(t) аналітично")
axs[0, 1].axvline(x=t0, color='r', linestyle='--', alpha=0.5)
axs[0, 1].plot(t0, exact_val, 'ro', label=f"M'({t0})={exact_val:.3f}")
axs[0, 1].set_title("Похідна M'(t)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Графік похибки від кроку h (log-log scale)
axs[1, 0].loglog(h_values, errors, 'r-o', markersize=4)
axs[1, 0].set_title("Похибка від кроку h (log-log)")
axs[1, 0].set_xlabel("h (зменшення ->)")
axs[1, 0].grid(True, which="both", ls="-", alpha=0.5)

# Гістограма похибок різних методів
methods = ['Центр. різн.\n(h=0.01)', 'Центр. різн.\n(h/2)', 'Рунге-Ромберг', 'Ейткен']
method_errors = [
    abs(central_diff(M, t0, 0.01) - exact_val),
    abs(d_h - exact_val),
    abs(d_RR - exact_val),
    abs(d_E - exact_val)
]
bars = axs[1, 1].bar(methods, method_errors, color=['tomato', 'peru', 'dodgerblue', 'limegreen'])
axs[1, 1].set_yscale('log')
axs[1, 1].set_title("Похибки методів")
for bar in bars:
    yval = bar.get_height()
    axs[1, 1].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2e}', va='bottom', ha='center', fontsize=8)

plt.show()

print(f"Оптимальний крок h: {h_opt:.2e}")
print(f"Порядок точності за Ейткеном p: {p_order:.2f}")