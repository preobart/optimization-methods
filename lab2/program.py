import time

import matplotlib.pyplot as plt
import numpy as np


class MethodPiyavskiy:
    def __init__(self, func, left, right, epsilon, L=None):
        self.func = func
        self.left = left
        self.right = right
        self.epsilon = epsilon
        self.L = L
        self.x_min = None
        self.f_min = None
        self.iter_count = 0
        self.time_elapsed = 0

    # Вычисление константы Липшица
    def compute_L(self):
        if self.L is not None:
            return self.L
        h = 1e-6
        X = np.linspace(self.left, self.right, 1000)
        df = (self.func(X + h) - self.func(X)) / h
        return np.max(np.abs(df))

    # Определение следующей точки на основе формулы пересечения ломанных
    def _next_candidate(self, points, L):
        sorted_pts = np.sort(points)
        candidate = None
        min_p_val = float('inf')

        for i in range(len(sorted_pts)-1):
            x1, x2 = sorted_pts[i], sorted_pts[i+1]
            f1, f2 = self.func(x1), self.func(x2)
            # формула пересечения ломанных
            x_int = (f2 - f1 + L*(x1 + x2)) / (2*L)
            if x1 < x_int < x2:
                p_val = 0.5 * (f1 + f2 - L*(x2 - x1))
                if p_val < min_p_val:
                    min_p_val = p_val
                    candidate = x_int

        # если пересечение вне интервала — взять середину самого длинного сегмента
        if candidate is None:
            max_len = 0
            for i in range(len(sorted_pts)-1):
                seg_len = sorted_pts[i+1] - sorted_pts[i]
                if seg_len > max_len:
                    max_len = seg_len
                    candidate = 0.5 * (sorted_pts[i] + sorted_pts[i+1])
        return candidate

    # Нижняя функция
    def _lower_envelope(self, x_vals, points, L):
        sorted_pts = np.sort(points)
        y = np.full_like(x_vals, -np.inf, dtype=float)
        for i in range(len(sorted_pts)-1):
            x1, x2 = sorted_pts[i], sorted_pts[i+1]
            f1, f2 = self.func(x1), self.func(x2)
            seg_y = np.maximum(f1 - L*np.abs(x_vals - x1), f2 - L*np.abs(x_vals - x2))
            y = np.maximum(y, seg_y)
        return y

    def optimize(self):
        start_time = time.time()
        L = self.compute_L()
        print(f"Константа Липшица: {L:.4f}")

        points = np.array([self.left, self.right])

        while True:
            self.iter_count += 1
            f_min_current = np.min(self.func(points))
            next_pt = self._next_candidate(points, L)
            points = np.append(points, next_pt)

            # вычисляем минимальное значение ломаной для проверки условия остановки
            sorted_pts = np.sort(points)
            min_lower = float('inf')
            for i in range(len(sorted_pts)-1):
                x1, x2 = sorted_pts[i], sorted_pts[i+1]
                f1, f2 = self.func(x1), self.func(x2)
                p_val = 0.5*(f1 + f2 - L*(x2 - x1))
                if p_val < min_lower:
                    min_lower = p_val

            gap = f_min_current - min_lower
            if gap < self.epsilon:
                self.x_min = points[np.argmin(self.func(points))]
                self.f_min = np.min(self.func(points))
                break

        self.time_elapsed = time.time() - start_time

        print(f"Глобальный минимум: f(x = {self.x_min:.6f}) = {self.f_min:.6f}")
        print(f"Число итераций: {self.iter_count}")
        print(f"Время работы: {self.time_elapsed:.4f} сек")

        self._plot(points, L)
        return self.x_min, self.f_min

    def _plot(self, points, L):
        X = np.linspace(self.left, self.right, 1000)
        plt.figure(figsize=(10,5))
        plt.plot(X, self.func(X), color='black', linewidth=2, label='f(x)')
        plt.plot(X, self._lower_envelope(X, points, L), '--', color='red', linewidth=2, label='Нижняя ломаная')
        plt.scatter(points, self.func(points), color='blue', zorder=5, label='Выбранные точки')
        plt.scatter(self.x_min, self.f_min, color='green', s=30, label='Минимум', zorder=10)
        plt.title('Метод Пиявского')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.show()