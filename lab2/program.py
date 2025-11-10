import time

import matplotlib.pyplot as plt
import numpy as np


def draw_broken_line(X, f, lower_func, points, L, iteration):
    plt.figure(figsize=(8, 5))
    plt.title(f'Поиск глобального минимума (итерация {iteration})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    plt.plot(X, f(X), label='f(x)')
    plt.plot(X, lower_func(X, points, L), '--', label='нижняя ломаная')
    plt.scatter(points, f(points), color='red', zorder=5, label='выбранные точки')

    plt.legend()
    plt.show()


def global_min_search(f, df, a, b, eps):
    """
    Поиск глобального минимума методом ломаных (Пиявский–Шуберт)
    """
    start = time.time()
    X = np.linspace(a, b, 1000)
    L = np.max(np.abs(df(X)))  # константа Липшица
    print(f"Константа Липшица: {L:.4f}")

    # начальные точки
    x_points = np.array([a, b])

    # нижняя функция
    def lower_func(x, pts, L):
        vals = []
        for xi in x:
            vals.append(np.max(f(pts) - L * np.abs(xi - pts)))
        return np.array(vals)

    iteration = 0
    while True:
        iteration += 1
        f_star = np.min(f(x_points))
        f_lower = lower_func(X, x_points, L)
        # точка, где нижняя оценка минимальна
        x_new = X[np.argmin(f_lower)]

        draw_broken_line(X, f, lower_func, x_points, L, iteration)

        gap = f_star - np.min(f_lower)
        print(f"Итерация {iteration}: x* ≈ {x_new:.4f}, f(x*) ≈ {f(x_new):.4f}, gap = {gap:.6f}")

        if gap < eps:
            xmin = x_new
            fmin = f(xmin)
            break
        x_points = np.append(x_points, x_new)

    elapsed = time.time() - start
    print("\n=== Результат ===")
    print(f"Глобальный минимум: f(x = {xmin:.4f}) = {fmin:.4f}")
    print(f"Число итераций: {iteration}")
    print(f"Время работы: {elapsed:.4f} сек")
    return xmin, fmin


def main():
    # Пример функции с несколькими локальными минимумами
    def f(x): return np.cos(x ** 2 - 3 * x) + 4
    def df(x): return -np.sin(x ** 2 - 3 * x) * (2 * x - 3)

    eps = 0.01
    a, b = -1, 3
    xmin, fmin = global_min_search(f, df, a, b, eps)


if __name__ == "__main__":
    main()