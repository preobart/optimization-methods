from data import FEE, INIT, MINV, STAGES, STEP


class InvestmentPlanner:
    def __init__(self):
        self.cache = {}

    def total(self, s):
        """Возвращает стоимость портфеля в состоянии s."""
        return s["cb1"] + s["cb2"] + s["dep"] + s["cash"]

    def apply_action(self, s, a):
        """
        Применяет действие a к состоянию s:
        - переводит активы по шагам STEP
        - учитывает комиссии FEE
        - обновляет деньги (cash)
        - соблюдает минимальные объемы MINV
        """
        cost = 0
        for k, d in a.items():
            c = d * STEP[k]
            c *= (1 + FEE[k]) if c > 0 else (1 - FEE[k])
            cost += c

        return dict(
            cb1=max(MINV["cb1"], s["cb1"] + a["cb1"] * STEP["cb1"]),
            cb2=max(MINV["cb2"], s["cb2"] + a["cb2"] * STEP["cb2"]),
            dep=max(MINV["dep"], s["dep"] + a["dep"] * STEP["dep"]),
            cash=max(0, s["cash"] - cost)
        )

    def apply_sit(self, s, k, idx):
        """
        Применяет один сценарий изменений стоимости активов
        из STAGES[k][idx] к состоянию s.
        """
        _, k1, k2, kd = STAGES[k][idx]
        return dict(
            cb1=s["cb1"] * k1,
            cb2=s["cb2"] * k2,
            dep=s["dep"] * kd,
            cash=s["cash"]
        )

    def actions(self, s):
        """
        Генерирует все допустимые действия из состояния s:
        - шаг изменения от -3 до 3
        - проверка минимальных объемов
        - проверка достаточности денег под комиссией
        """
        res = []
        max_steps = 3
        for d1 in range(-max_steps, max_steps + 1):
            for d2 in range(-max_steps, max_steps + 1):
                for dd in range(-max_steps, max_steps + 1):
                    a = {"cb1": d1, "cb2": d2, "dep": dd}

                    if s["cb1"] + d1 * STEP["cb1"] < MINV["cb1"]:
                        continue
                    if s["cb2"] + d2 * STEP["cb2"] < MINV["cb2"]:
                        continue
                    if s["dep"] + dd * STEP["dep"] < MINV["dep"]:
                        continue

                    cost = 0
                    for k, d in a.items():
                        c = d * STEP[k]
                        c *= (1 + FEE[k]) if c > 0 else (1 - FEE[k])
                        cost += c

                    if cost > s["cash"] + 1e-9:
                        continue

                    res.append(a)

        if not any(a["cb1"] == a["cb2"] == a["dep"] == 0 for a in res):
            res.append({"cb1": 0, "cb2": 0, "dep": 0})

        return res
    
    def bellman(self, k, s):
        """
        Основной метод динамического программирования.
        Для этапа k и состояния s:
        - перебирает все допустимые действия
        - считает ожидаемую прибыль с учетом вероятностей событий
        - использует кэш для ускорения
        """
        key = (
            k,
            round(s["cb1"] / 50) * 50,
            round(s["cb2"] / 100) * 100,
            round(s["dep"] / 50) * 50,
            round(s["cash"] / 100) * 100
        )

        if key in self.cache:
            return self.cache[key]

        # база рекурсии - просто стоимость
        if k == len(STAGES):
            return self.total(s), None

        best_v = -1e18
        best_a = None

        for a in self.actions(s):
            ns = self.apply_action(s, a)
            ev = 0

            # Взвешивание будущей стоимости по вероятностям событий
            for sit in range(3):
                p, _, _, _ = STAGES[k][sit]
                ns2 = self.apply_sit(ns, k, sit)
                fv, _ = self.bellman(k + 1, ns2)
                ev += p * fv

            if ev > best_v:
                best_v = ev
                best_a = a

        self.cache[key] = (best_v, best_a)
        return best_v, best_a

    def solve(self):
        s = INIT.copy()
        path = []

        for k in range(len(STAGES)):
            _, a = self.bellman(k, s)
            ns = self.apply_action(s, a)

            # Формируем ожидаемое состояние после применения вероятностей
            exp = dict(cb1=0, cb2=0, dep=0, cash=ns["cash"])
            for p, k1, k2, kd in STAGES[k]:
                exp["cb1"] += p * ns["cb1"] * k1
                exp["cb2"] += p * ns["cb2"] * k2
                exp["dep"] += p * ns["dep"] * kd

            path.append((a, ns))
            s = exp

        return self.total(s), path

def format_action(action):
    parts = []
    names = {"cb1": "ЦБ1", "cb2": "ЦБ2", "dep": "Депозиты"}
    for k, v in action.items():
        if v != 0:
            sign = "+" if v > 0 else ""
            parts.append(f"{names[k]}: {sign}{v} шагов ({sign}{v*STEP[k]:.2f} д.е.)")
    if not parts:
        return "Без изменений"
    return ", ".join(parts)


if __name__ == "__main__":
    planner = InvestmentPlanner()
    value, path = planner.solve()

    print("ОПТИМАЛЬНАЯ СТРАТЕГИЯ УПРАВЛЕНИЯ:")
    print()

    for i, (action, state) in enumerate(path, 1):
        print(f"Этап {i}: {format_action(action)}")
        print(f"  Стоимость портфеля: {planner.total(state):.2f} д.е.")
        print()

    print("РЕЗУЛЬТАТЫ:")
    print()

    initial_value = planner.total(INIT)
    growth = value - initial_value
    growth_pct = growth / initial_value * 100

    print(f"  Начальная стоимость портфеля: {initial_value:.2f} д.е.")
    print(f"  Максимальный ожидаемый доход: {value:.2f} д.е.")
    print(f"  Прирост: {growth:.2f} д.е. ({growth_pct:.2f}%)")
    print(f"  Всего обработано состояний: {len(planner.cache)}")