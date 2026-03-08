import numpy as np
import matplotlib.pyplot as plt
import requests

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.164983,24.523574|48.166053,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
data = requests.get(url).json()["results"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

elevs = [p['elevation'] for p in data]
lats, lons = [p['latitude'] for p in data], [p['longitude'] for p in data]
dist = [0]
for i in range(1, len(data)):
    dist.append(dist[-1] + haversine(lats[i-1], lons[i-1], lats[i], lons[i]))

X_f, Y_f = np.array(dist), np.array(elevs)

def solve_spline(x, y):
    n = len(x)
    h = np.diff(x)
    A, B = np.zeros((n, n)), np.zeros(n)
    A[0, 0], A[-1, -1] = 1, 1
    for i in range(1, n-1):
        A[i, i-1], A[i, i], A[i, i+1] = h[i-1], 2*(h[i-1]+h[i]), h[i]
        B[i] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    c = np.linalg.solve(A, B)
    a = y[:-1]
    b = (y[1:]-y[:-1])/h - h*(c[1:]+2*c[:-1])/3
    d = (c[1:]-c[:-1])/(3*h)
    return a, b, c[:-1], d

def eval_spline(xk, xq, a, b, c, d):
    idx = np.clip(np.searchsorted(xk, xq)-1, 0, len(a)-1)
    dx = xq - xk[idx]
    return a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3

# Графік 1: Точки з'єднані лініями (лінійна інтерполяція)
plt.figure(figsize=(10, 5))
plt.plot(X_f, Y_f, 'ro-', label='Лінійна інтерполяція (з\'єднані точки)')
plt.title('Графік 1: Вхідні дані (з\'єднані точки)')
plt.xlabel('Відстань (м)'); plt.ylabel('Висота (м)')
plt.legend(); plt.grid(True)
plt.show()

# Графік 2: З'єднані точки + плавний кубічний сплайн
plt.figure(figsize=(10, 5))
sa, sb, sc, sd = solve_spline(X_f, Y_f)
xs = np.linspace(X_f[0], X_f[-1], 500)
ys = eval_spline(X_f, xs, sa, sb, sc, sd)
plt.plot(X_f, Y_f, 'ro--', alpha=0.4, label='Вхідна ламана')
plt.plot(xs, ys, 'b-', linewidth=2, label='Кубічний сплайн (плавна крива)')
plt.title('Графік 2: Порівняння ламаної та кубічного сплайна')
plt.legend(); plt.grid(True)
plt.show()

# Графік 3: Порівняння вузлів 10, 15, 20
plt.figure(figsize=(10, 6))
for n in [10, 15, 20]:
    idx = np.linspace(0, len(X_f)-1, n, dtype=int)
    xk, yk = X_f[idx], Y_f[idx]
    sa, sb, sc, sd = solve_spline(xk, yk)
    ys_n = eval_spline(xk, xs, sa, sb, sc, sd)
    plt.plot(xs, ys_n, label=f'Сплайн ({n} вузлів)')

plt.plot(X_f, Y_f, 'k.', alpha=0.3, label='Точки GPS')
plt.title('Графік 3: Вплив кількості вузлів на гладкість')
plt.legend(); plt.grid(True)
plt.show()