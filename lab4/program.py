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
        - шаг изменения от -2 до 2
        - проверка минимальных объемов
        - проверка достаточности денег под комиссией
        """
        res = []
        for d1 in range(-2, 3):
            for d2 in range(-2, 3):
                for dd in range(-2, 3):
                    a = {"cb1": d1, "cb2": d2, "dep": dd}

                    if s["cb1"] + d1 * STEP["cb1"] < MINV["cb1"]:
                        continue
                    if s["cb2"] + d2 * STEP["cb2"] < MINV["cb2"]:
                        continue
                    if s["dep"] + dd * STEP["dep"] < MINV["dep"]:
                        continue

                    ok = True
                    for k, d in a.items():
                        c = d * STEP[k]
                        c *= (1 + FEE[k]) if c > 0 else (1 - FEE[k])
                        if c > s["cash"] + 1e-9:
                            ok = False
                            break

                    if ok:
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

if __name__ == "__main__":
    planner = InvestmentPlanner()
    value, path = planner.solve()
    print(f"Начальный портфель: {planner.total(INIT):.2f} д.е.\n")
    for i, (action, state) in enumerate(path, 1):
        action_str = ", ".join(f"{k.upper()}:{v*STEP[k]:.2f}" for k, v in action.items() if v != 0)
        if not action_str:
            action_str = "Без изменений"
        print(f"Этап {i}: {action_str}")
        print(f"  Стоимость портфеля: {planner.total(state):.2f} д.е.\n")
    growth = value - planner.total(INIT)
    growth_pct = growth / planner.total(INIT) * 100
    print(f"Максимальный ожидаемый доход: {value:.2f} д.е. (Прирост: {growth:.2f} д.е., {growth_pct:.2f}%)")
    print(f"Всего обработано состояний: {len(planner.cache)}")