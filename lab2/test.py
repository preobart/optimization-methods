import numpy as np
from program import MethodPiyavskiy


def rastrigin(x):
    x = np.array(x)
    return 10 + x**2 - 10 * np.cos(2 * np.pi * x)

def ackley(x):
    x = np.array(x)
    return -20 * np.exp(-0.2*np.sqrt(0.5)*abs(x)) - np.exp(0.5*np.cos(2*np.pi*x)) + 20 + np.e

def func(x):
    x = np.array(x)
    return x + np.sin(3.14159*x)

test_cases = [
    ("Функция Растригина", rastrigin, -3, 3, 0.01, 2.0),
    ("Функция Экли", ackley, -3, 3, 0.01, 5.0),
    ("", func, -3, 3, 0.01, 4.0)
]

for name, func, a, b, eps, L in test_cases:
    print(f"\n{name}")
    solver = MethodPiyavskiy(func, a, b, eps, L=L)
    solver.optimize()