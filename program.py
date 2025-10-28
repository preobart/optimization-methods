from enum import Enum

import numpy as np


class VariableType(Enum):
    ORIGINAL = 0
    ADDITIONAL = 1
    TEMPORARY = 2

class Status(Enum):
    OK = 0
    FOUND = 1
    NO_SOLUTION = 2
    INFINITE = 3

class SimplexSolver:
    def __init__(self, filename):
        self.filename = filename
        self.sense = None
        self.obj = None
        self.A = None
        self.b = None
        self.rels = None
        self.base = None
        self.var_types = None

    def load_problem(self):
        """Загрузка задачи из файла. Формат: первая строка — коэффициенты цели + 'min'/'max', далее ограничения"""
        lines = [line.strip() for line in open(self.filename, encoding="utf-8-sig") if line.strip()]
        parts = lines[0].split()
        self.sense = parts[-1].lower()
        self.obj = np.array([float(x) for x in parts[:-1]])

        A, b, rels = [], [], []
        for ln in lines[1:]:
            for r in ("<=", ">=", "="):
                if r in ln:
                    left, right = ln.split(r)
                    A.append([float(x) for x in left.split()])
                    b.append(float(right))
                    rels.append(r)
                    break
        self.A, self.b, self.rels = np.array(A), np.array(b), rels

    def prepare_canonical(self):
        """Преобразует задачу линейного программирования к каноническому виду."""
        m, n = self.A.shape
        if self.sense == "min":
            self.obj = -self.obj

        objective_coefficients = list(self.obj.copy())
        constraint_matrix = [list(row.copy()) for row in self.A]
        constraint_rhs = list(self.b)
        constraint_senses = list(self.rels)

        self.base = [-1] * m
        self.var_types = [VariableType.ORIGINAL] * n
        next_var = n

        for i in range(m):
            if constraint_senses[i] == "=":
                for j in range(m):
                    constraint_matrix[j].append(1 if j == i else 0)
                objective_coefficients.append(0)
                self.var_types.append(VariableType.TEMPORARY)
                self.base[i] = next_var
                next_var += 1

            elif constraint_senses[i] == "<=":
                for j in range(m):
                    constraint_matrix[j].append(1 if j == i else 0)
                objective_coefficients.append(0)
                self.var_types.append(VariableType.ADDITIONAL)
                self.base[i] = next_var
                next_var += 1
                constraint_senses[i] = "="

            elif constraint_senses[i] == ">=":
                for j in range(m):
                    constraint_matrix[j].append(-1 if j == i else 0)
                objective_coefficients.append(0)
                self.var_types.append(VariableType.ADDITIONAL)
                next_var += 1

                for j in range(m):
                    constraint_matrix[j].append(1 if j == i else 0)
                objective_coefficients.append(0)
                self.var_types.append(VariableType.TEMPORARY)
                self.base[i] = next_var
                next_var += 1
                constraint_senses[i] = "="

        canonical_problem_table = [objective_coefficients.copy()]
        canonical_problem_table[0].append(self.sense)
        for i in range(m):
            row = constraint_matrix[i].copy()
            row.append(constraint_rhs[i])
            canonical_problem_table.append(row)

        self.canonical_problem_table = canonical_problem_table
        self.A = np.array([row[:-1] for row in canonical_problem_table[1:]], dtype=float)
        self.b = np.array([row[-1] for row in canonical_problem_table[1:]], dtype=float)
        self.obj = np.array(objective_coefficients, dtype=float)
        self.rels = ["="] * m
   
    def pivot(self, table, bidx, r, c):
        """Поворот по элементу (r,c)"""
        table[r] /= table[r,c]
        for i in range(len(table)):
            if i != r:
                table[i] -= table[i,c]*table[r]
        bidx[r] = c

    def select_row(self, col, rhs):
        """Выбор строки по правилу минимального отношения"""
        candidates = [i for i,v in enumerate(col) if v>1e-9]
        if not candidates: 
            return None
        ratios = rhs[candidates]/col[candidates]
        min_val = ratios.min()
        for i in candidates:
            if abs(rhs[i]/col[i]-min_val)<1e-12: 
                return i
        return None

    def run_simplex_core(self, table, bidx):
        """Основной цикл симплекс-метода"""
        m = len(bidx)
        while True:
            rc = table[-1,:-1]
            cols = np.where(rc>1e-9)[0]
            if len(cols)==0: 
                return Status.FOUND
            col = int(cols[0])
            row = self.select_row(table[:m,col], table[:m,-1])
            if row is None: 
                return Status.INFINITE
            self.pivot(table, bidx, row, col)

    def build_auxiliary(self):
        """Формирование вспомогательной задачи"""
        m, n = self.A.shape
        table = np.zeros((m+1,n+1))
        table[:m,:n] = self.A
        table[:m,-1] = self.b
        table[-1,:n] = [-1 if t==VariableType.TEMPORARY else 0 for t in self.var_types]

        bidx = self.base.copy()
        for i,j in enumerate(bidx):
            if 0<=j<len(self.var_types) and self.var_types[j]==VariableType.TEMPORARY:
                table[-1] += table[i]
        return table, bidx

    def solve_auxiliary(self, table, bidx):
        """Решение вспомогательной задачи"""
        status = self.run_simplex_core(table,bidx)
        if status==Status.INFINITE: 
            return Status.INFINITE,None,None
        if abs(table[-1,-1])>1e-7: 
            return Status.NO_SOLUTION,None,None
        return Status.OK,table,bidx

    def solve_main(self, table, bidx):
        keep = [i for i, t in enumerate(self.var_types) if t != VariableType.TEMPORARY]
        m = table.shape[0] - 1
        n = len(keep)

        tab = np.zeros((m + 1, n + 1))
        tab[:m, :n] = table[:m, keep]
        tab[:m, -1] = table[:m, -1]
        tab[-1, :n] = self.obj[keep]

        bidx2 = [keep.index(j) for j in bidx if j in keep]
        for i, j in enumerate(bidx2):
            tab[-1, :] -= tab[-1, j] * tab[i, :]

        status = self.run_simplex_core(tab, bidx2)
        if status != Status.FOUND:
            return status, None, None

        x = np.zeros(len(keep))
        for i, j in enumerate(bidx2):
            x[j] = tab[i, -1]

        return Status.FOUND, tab[-1, -1], x

    def run(self):
        """Запуск решения задачи"""
        self.load_problem()
        self.prepare_canonical()
        aux_tab, aux_base = self.build_auxiliary()
        status, aux_tab, aux_base = self.solve_auxiliary(aux_tab, aux_base)
        if status!=Status.OK:
            print("Задачу решить невозможно:", status)
            return
        status, z, x = self.solve_main(aux_tab, aux_base)
        if status==Status.FOUND:
            print(f"F = {z:g}")
            print("Точки:", " | ".join(f"x_{i}={val:g}" for i, val in enumerate(x, start=1)))
        else:
            print("Результат:", status)

if __name__=="__main__":
    SimplexSolver('input.txt').run()