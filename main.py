import requests
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    coords = "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167332,24.530935|48.167825,24.530044|48.168345,24.529023|48.168874,24.527845|48.169425,24.526543|48.170012,24.525234|48.170643,24.523821|48.171245,24.522412|48.171854,24.520934|48.172456,24.519456|48.173023,24.518012|48.173645,24.516543|48.174212,24.515012|48.174854,24.513543|48.175412,24.512012|48.176043,24.510543|48.176612,24.509012|48.177245,24.507543|48.177812,24.506012|48.178454,24.504543|48.179012,24.503012"
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={coords}"
    data = requests.get(url).json()
    return data["results"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def solve_progonka(x, y):
    n = len(x)
    h = np.diff(x)
    A, B, C, D = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    alpha, beta = np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        m = A[i] * alpha[i - 1] + B[i]
        alpha[i] = -C[i] / m
        beta[i] = (D[i] - A[i] * beta[i - 1]) / m
    c = np.zeros(n)
    for i in range(n - 2, 0, -1):
        c[i] = alpha[i] * c[i + 1] + beta[i]
    a = y[:-1]
    b, d = np.zeros(n - 1), np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    return a, b, c[:-1], d

results = get_data()
lats = [p['latitude'] for p in results]
lons = [p['longitude'] for p in results]
elevs = [p['elevation'] for p in results]
dist = [0]
for i in range(1, len(results)):
    dist.append(dist[-1] + haversine(lats[i - 1], lons[i - 1], lats[i], lons[i]))

x, y = np.array(dist), np.array(elevs)
a, b, c, d = solve_progonka(x, y)

total_dist = x[-1]
total_ascent = sum(max(0, y[i] - y[i - 1]) for i in range(1, len(y)))
total_descent = sum(max(0, y[i-1] - y[i]) for i in range(1, len(y)))
energy = 80 * 9.81 * total_ascent

grad_full = np.gradient(y, x) * 100
max_grad = np.max(grad_full)
min_grad = np.min(grad_full)
avg_grad = np.mean(np.abs(grad_full))
steep_sections = np.sum(np.abs(grad_full) > 15)

print(f"Загальна довжина: {total_dist:.2f} м")
print(f"Загальний підйом: {total_ascent:.2f} м")
print(f"Загальний спуск: {total_descent:.2f} м")
print(f"Максимальний підйом (%): {max_grad:.2f}")
print(f"Максимальний спуск (%): {min_grad:.2f}")
print(f"Середній градієнт (%): {avg_grad:.2f}")
print(f"Ділянки з крутизною > 15%: {steep_sections}")
print(f"Механічна робота: {energy:.2f} Дж")
print(f"Енергія: {energy / 1000:.2f} кДж")
print(f"Енергія (ккал): {energy / 4184:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='Точки GPS')
for i in range(len(x) - 1):
    xs = np.linspace(x[i], x[i + 1], 10)
    ys = a[i] + b[i] * (xs - x[i]) + c[i] * (xs - x[i]) ** 2 + d[i] * (xs - x[i]) ** 3
    plt.plot(xs, ys, 'b-')
plt.title('Профіль маршруту: Заросляк - Говерла')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.show()