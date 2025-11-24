from enum import Enum

import numpy as np


class VariableKind(Enum):
    ORIGINAL = 0      
    SLACK = 1        
    ARTIFICIAL = 2     


class LPStatus(Enum):
    OK = 0
    FOUND = 1
    NO_SOLUTION = 2
    UNBOUNDED = 3


class SimplexSolver:
    def __init__(self, filename, free_vars=None):
        self.filename = filename
        self.free_vars = sorted(set(free_vars or []))
        self.obj = None       # Коэффициенты целевой функции
        self.sense = None     # "min" или "max"
        self.A = None         # Матрица коэффициентов ограничений
        self.b = None         # Правая часть ограничений
        self.rels = None      # Операторы ограничений: <=, >=, =
        self.var_kinds = []   # Типы переменных (ORIGINAL, SLACK, ARTIFICIAL)
        self.base = []        # Индексы базисных переменных

    def load_problem(self):
        """
        Загружает задачу из файла.
        """
        lines = [line.strip() for line in open(self.filename, encoding="utf-8-sig") if line.strip()]
        parts = lines[0].split()
        self.sense = parts[-1].lower()
        self.obj = np.array([float(x) for x in parts[:-1]])

        A, b, rels = [], [], []
        for line in lines[1:]:
            for sign in ("<=", ">=", "="):
                if sign in line:
                    left, right = line.split(sign)
                    A.append([float(x) for x in left.split()])
                    b.append(float(right))
                    rels.append(sign)
                    break
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.rels = rels

    def expand_free_variables(self):
        """
        Преобразуем отрицательные переменные в разность двух неотрицательных
        """
        if not self.free_vars:
            self.var_kinds = [VariableKind.ORIGINAL] * self.A.shape[1]
            return

        m, n = self.A.shape
        new_cols, new_obj, new_kinds = [], [], []
        for j in range(n):
            col = self.A[:, j]
            coef = self.obj[j]
            if j in self.free_vars:
                new_cols.append(col)
                new_cols.append(-col)
                new_obj.append(coef)
                new_obj.append(-coef)
                new_kinds.extend([VariableKind.ORIGINAL, VariableKind.ORIGINAL])
            else:
                new_cols.append(col)
                new_obj.append(coef)
                new_kinds.append(VariableKind.ORIGINAL)
        self.A = np.column_stack(new_cols)
        self.obj = np.array(new_obj)
        self.var_kinds = new_kinds

    def to_canonical_form(self):
        """
        Преобразует задачу к каноническому виду:
        - Все ограничения превращаются в равенства с добавлением slack и artificial переменных.
        - Для <= добавляется slack-переменная.
        - Для >= добавляется -slack + artificial.
        - Для = добавляется artificial.
        """
        m, n = self.A.shape
        if self.sense == "min":
            # Преобразуем min в max для стандартного симплекс-метода
            self.obj = -self.obj

        base = [-1] * m
        next_col = n
        for i, rel in enumerate(self.rels):
            if rel == "<=":
                # Добавляем slack-переменную
                col = np.zeros(m)
                col[i] = 1
                self.A = np.column_stack((self.A, col))
                self.obj = np.append(self.obj, 0)
                self.var_kinds.append(VariableKind.SLACK)
                base[i] = next_col
                next_col += 1
            elif rel == ">=":
                # Добавляем surplus-переменную и artificial
                col1 = np.zeros(m)
                col1[i] = -1
                self.A = np.column_stack((self.A, col1))
                self.obj = np.append(self.obj, 0)
                self.var_kinds.append(VariableKind.SLACK)
                next_col += 1
                col2 = np.zeros(m)
                col2[i] = 1
                self.A = np.column_stack((self.A, col2))
                self.obj = np.append(self.obj, 0)
                self.var_kinds.append(VariableKind.ARTIFICIAL)
                base[i] = next_col - 1
                next_col += 1
            elif rel == "=":
                # Добавляем artificial переменную
                col = np.zeros(m)
                col[i] = 1
                self.A = np.column_stack((self.A, col))
                self.obj = np.append(self.obj, 0)
                self.var_kinds.append(VariableKind.ARTIFICIAL)
                base[i] = next_col
                next_col += 1

        self.rels = ["="] * m
        self.base = base

    def pivot(self, T, base, r, c):
        """
        Выполняет pivot по элементу (r, c):
        - Делает столбец c базисным в строке r
        - Обновляет таблицу симплекс-метода
        """
        T[r] /= T[r, c]
        for i in range(len(T)):
            if i != r:
                T[i] -= T[i, c] * T[r]
        base[r] = c

    def choose_row(self, col, rhs):
        """
        Выбирает строку для pivot по правилу минимального отношения.
        Выбираем минимальное положительное отношение rhs[i] / col[i].
        """
        valid = [i for i, v in enumerate(col) if v > 1e-9]
        if not valid:
            return None
        ratios = rhs[valid] / col[valid]
        idx = np.argmin(ratios)
        return valid[idx]

    def simplex(self, T, base):
        """
        Основной цикл симплекс-метода:
        1. Находим положительный коэффициент в строке цели (rc)
        2. Выбираем соответствующую строку pivot
        3. Выполняем pivot
        4. Повторяем до оптимума или неограниченности
        """
        m = len(base)
        while True:
            rc = T[-1, :-1]
            pos = np.where(rc > 1e-9)[0]
            if len(pos) == 0:
                return LPStatus.FOUND
            c = pos[0]
            r = self.choose_row(T[:m, c], T[:m, -1])
            if r is None:
                return LPStatus.UNBOUNDED
            self.pivot(T, base, r, c)

    def build_auxiliary(self):
        """
        Строим вспомогательную задачу для искусственных переменных,
        чтобы получить допустимый базис.
        """
        m, n = self.A.shape
        T = np.zeros((m + 1, n + 1))
        T[:m, :n] = self.A
        T[:m, -1] = self.b
        T[-1, :n] = [-1 if k == VariableKind.ARTIFICIAL else 0 for k in self.var_kinds]
        base = self.base.copy()
        for i, j in enumerate(base):
            if j >= 0 and self.var_kinds[j] == VariableKind.ARTIFICIAL:
                T[-1] += T[i]
        return T, base

    def solve_auxiliary(self, T, base):
        """
        Решаем вспомогательную задачу.
        Проверяем наличие допустимого решения.
        """
        self.simplex(T, base)
        if abs(T[-1, -1]) > 1e-8:
            return LPStatus.NO_SOLUTION, None, None
        return LPStatus.OK, T, base

    def solve(self):
        """
        Полное решение задачи ЛП:
        1. Загружаем задачу
        2. Преобразуем свободные переменные
        3. Приводим к канонической форме
        4. Решаем вспомогательную задачу для базиса
        5. Основной симплекс-метод
        6. Вывод оптимального значения F и точек x_i
        """
        self.load_problem()
        self.expand_free_variables()
        self.to_canonical_form()

        # Вспомогательная задача
        T, base = self.build_auxiliary()
        status, T, base = self.solve_auxiliary(T, base)
        if status != LPStatus.OK:
            print("Нет допустимого решения")
            return

        # Основная задача
        keep = [i for i, k in enumerate(self.var_kinds) if k != VariableKind.ARTIFICIAL]
        m = T.shape[0] - 1
        tab = np.zeros((m + 1, len(keep) + 1))
        tab[:m, :-1] = T[:m, keep]
        tab[:m, -1] = T[:m, -1]
        tab[-1, :-1] = self.obj[keep]

        base2 = [keep.index(j) for j in base if j in keep]
        for i, j in enumerate(base2):
            tab[-1] -= tab[-1, j] * tab[i]

        status = self.simplex(tab, base2)
        if status != LPStatus.FOUND:
            print("Решение не найдено")
            return

        x = np.zeros(len(keep))
        for i, j in enumerate(base2):
            x[j] = tab[i, -1]

        print(f"F = {tab[-1, -1]:g}")
        print("Точки:", " | ".join(f"x_{i+1}={val:g}" for i, val in enumerate(x)))


if __name__ == "__main__":
    SimplexSolver("input1.txt").solve()