import re
import math

EPS = 1e-9

# Парсинг
def parse_equation(eq_line):
    s = eq_line.replace(' ', '')
    left, right = re.split('<=|>=|=', s)
    sign = re.findall('<=|>=|=', s)[0]
    coeffs = {}
    for num, var in re.findall(r'([+-]?\d*)(x\d+)', left):
        if num in ('', '+'): val = 1
        elif num == '-': val = -1
        else: val = int(num)
        coeffs[var] = val
    return coeffs, sign, float(right)

def read_lp(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    goal = lines[0].lower()
    obj_str = lines[1]
    obj = {}
    for num, var in re.findall(r'([+-]?\d*)(x\d+)', obj_str):
        if num in ('', '+'): v = 1
        elif num == '-': v = -1
        else: v = int(num)
        obj[var] = v
    constraints = [parse_equation(ln) for ln in lines[2:]]
    return goal, obj, constraints

# Приведение к каноническому виду (с учётом ограничений)
def build_canonical(goal, obj, constraints):
    # исходные переменные (x1, x2, ...)
    orig_vars = sorted({v for c,_,_ in constraints for v in c} | set(obj.keys()),
                       key=lambda x: int(re.findall(r'\d+', x)[0]))
    A_rows = []
    b = []
    row_types = []   # сохраняем тип ограничения (<=, >=, =)
    slack_vars = []
    art_vars = []
    added_per_row = []  # какие добавочные переменные появились в каждой строке (нужно для базиса)
    for i,(coeffs, sign, rhs) in enumerate(constraints):
        row = [coeffs.get(v, 0) for v in orig_vars]
        added = []
        if sign == '<=':
            s = f"s{i+1}"
            added.append(s)
            slack_vars.append(s)
            # добавляем переменную запаса +1
            row += [1]
        elif sign == '>=':
            s = f"s{i+1}"
            a = f"a{i+1}"
            # добавляем избыточную (-1) и искусственную (+1)
            added.extend([s, a])
            slack_vars.append(s)
            art_vars.append(a)
            row += [-1, 1]
        elif sign == '=':
            a = f"a{i+1}"
            added.append(a)
            art_vars.append(a)
            row += [1]
        A_rows.append(row)
        b.append(rhs)
        row_types.append(sign)
        added_per_row.append(added)

    # выравниваем длину строк матрицы
    added_all = []
    for added in added_per_row:
        for var in added:
            if var not in added_all:
                added_all.append(var)
    all_vars = orig_vars + added_all

    # дополняем строки нулями, если в них нет некоторых добавочных переменных
    m = len(A_rows)
    n_orig = len(orig_vars)
    for i in range(m):
        current_len = len(A_rows[i])
        target_len = n_orig + len(added_all)
        if current_len < target_len:
            # создаём словарь добавочных переменных для текущей строки
            this_added = added_per_row[i]
            added_vals = {}
            for j, var in enumerate(this_added):
                added_vals[var] = A_rows[i][n_orig + j]
            full_added_part = [added_vals.get(var, 0) for var in added_all]
            A_rows[i] = A_rows[i][:n_orig] + full_added_part

    # создаём вектор коэффициентов целевой функции
    c = [obj.get(v, 0) for v in orig_vars] + [0]*len(added_all)

    return all_vars, A_rows, b, c, slack_vars, art_vars, row_types

# Таблица / операции
def print_tableau(tableau, basic_vars, all_vars, phase, step):
    m = len(tableau)-1
    header = ["Базис"] + all_vars + ["Свободн."]
    print(f"\n===== {phase} — Итерация {step} =====")
    print(" | ".join(f"{h:>8}" for h in header))
    print("-" * (10 * (len(header))))
    for i in range(m):
        row = tableau[i]
        print(f"{basic_vars[i]:>8} | " + " | ".join(f"{row[j]:8.3f}" for j in range(len(row))))
    print("-" * (10 * (len(header))))
    last = tableau[-1]
    print(f"{'Obj':>8} | " + " | ".join(f"{last[j]:8.3f}" for j in range(len(last))))
    print("=" * (10 * (len(header))))

def pivot(tableau, row_idx, col_idx):
    piv = tableau[row_idx][col_idx]
    if abs(piv) < EPS:
        raise ValueError("Опорный элемент (pivot) близок к нулю")
    # нормализуем строку с опорным элементом
    tableau[row_idx] = [v / piv for v in tableau[row_idx]]
    # обнуляем остальные строки в этом столбце
    for i in range(len(tableau)):
        if i == row_idx:
            continue
        factor = tableau[i][col_idx]
        tableau[i] = [tableau[i][j] - factor * tableau[row_idx][j] for j in range(len(tableau[0]))]

def find_entering_col(obj_row):
    # для задачи на максимум: если в последней строке есть отрицательные коэффициенты — улучшаем
    candidates = [(j, val) for j,val in enumerate(obj_row[:-1])]
    min_val = min(val for j,val in candidates)
    if min_val >= -EPS:
        return None
    for j,val in candidates:
        if val == min_val:
            return j
    return None

def find_leaving_row(tableau, col):
    # метод минимального отношения для выбора выходящей переменной
    ratios = []
    for i,row in enumerate(tableau[:-1]):
        coeff = row[col]
        if coeff > EPS:
            ratios.append((i, row[-1] / coeff))
        else:
            ratios.append((i, float('inf')))
    min_ratio = min(r for i,r in ratios)
    if math.isinf(min_ratio):
        return None
    for i,r in ratios:
        if abs(r - min_ratio) < 1e-12:
            return i
    return None

def simplex_iterations(tableau, basic_vars, all_vars, phase_name):
    step = 0
    while True:
        print_tableau(tableau, basic_vars, all_vars, phase_name, step)
        enter = find_entering_col(tableau[-1])
        if enter is None:
            # достигнуто оптимальное решение
            break
        leave = find_leaving_row(tableau, enter)
        if leave is None:
            raise ValueError(f"{phase_name}: Неограниченная область (нет выходящей переменной).")
        print(f"Pivot: базисная переменная '{basic_vars[leave]}' заменяется на '{all_vars[enter]}' (строка {leave}, столбец {enter})")
        pivot(tableau, leave, enter)
        basic_vars[leave] = all_vars[enter]
        step += 1
    return tableau, basic_vars

# Фаза 1: вспомогательная задача
def phase_one(all_vars, A, b, c, art_vars, row_types):
    m = len(A)
    n = len(all_vars)
    # формируем начальный базис: для каждой строки выбираем s (если есть), иначе a
    basic_vars = []
    for i,rt in enumerate(row_types):
        found = None
        for j,v in enumerate(all_vars):
            if v.startswith('s') and abs(A[i][j]) > EPS:
                found = v
                break
        if found:
            basic_vars.append(found)
        else:
            found_a = None
            for v in art_vars:
                j = all_vars.index(v)
                if abs(A[i][j]) > EPS:
                    found_a = v
                    break
            if found_a is None:
                for j,v in enumerate(all_vars):
                    col = [A[r][j] for r in range(m)]
                    if abs(A[i][j] - 1) < EPS and sum(abs(x) for k,x in enumerate(col) if k!=i) < EPS:
                        found_a = v
                        break
            if found_a is None:
                raise RuntimeError("Не удалось подобрать начальный базис для строки " + str(i))
            basic_vars.append(found_a)

    # строим таблицу для вспомогательной задачи (фаза 1)
    tableau = []
    for i in range(m):
        tableau.append([A[i][j] for j in range(n)] + [b[i]])
    # функция цели фазы 1: сумма искусственных переменных
    obj_aux = [1.0 if v in art_vars else 0.0 for v in all_vars] + [0.0]
    tableau.append(obj_aux)

    # делаем целевую строку приведённой (вычитаем строки с искусственными базисами)
    for i, bv in enumerate(basic_vars):
        if bv in art_vars:
            tableau[-1] = [tableau[-1][k] - tableau[i][k] for k in range(len(tableau[0]))]

    # выполняем симплекс-итерации для фазы 1
    tableau, basic_vars = simplex_iterations(tableau, basic_vars, all_vars, "Phase I")

    W = tableau[-1][-1]
    if abs(W) > 1e-6:
        raise ValueError(f"Phase I: допустимого решения нет (W* = {W})")

    # исключаем искусственные переменные из таблицы
    for i, bv in enumerate(basic_vars[:]):
        if bv in art_vars:
            found_j = None
            for j,v in enumerate(all_vars):
                if v in art_vars: continue
                if abs(tableau[i][j]) > EPS:
                    found_j = j
                    break
            if found_j is not None:
                pivot(tableau, i, found_j)
                basic_vars[i] = all_vars[found_j]

    keep_idx = [j for j,v in enumerate(all_vars) if v not in art_vars]
    new_all_vars = [all_vars[j] for j in keep_idx]
    new_tableau = []
    for i in range(len(tableau)):
        new_row = [tableau[i][j] for j in keep_idx] + [tableau[i][-1]]
        new_tableau.append(new_row)

    new_basic = []
    for bv in basic_vars:
        if bv in art_vars:
            new_basic.append(bv)
        else:
            new_basic.append(bv)
    for i in range(len(new_basic)):
        if new_basic[i] not in new_all_vars:
            row = new_tableau[i]
            found = False
            for j,v in enumerate(new_all_vars):
                col = [new_tableau[r][j] for r in range(len(new_tableau)-1)]
                if abs(row[j] - 1) < EPS and sum(abs(col[k]) for k in range(len(col)) if k!=i) < EPS:
                    new_basic[i] = v
                    found = True
                    break
            if not found:
                new_basic[i] = new_all_vars[0] if new_all_vars else f"dummy{i}"
    return new_all_vars, new_tableau, new_basic

# фаза 2: подстановка исходной целевой функции и симплекс
def phase_two(all_vars, tableau, basic_vars, orig_c, goal):
    n = len(all_vars)
    # задаём строку цели (в виде -c для максимизации)
    obj_row = [-orig_c[j] for j in range(n)] + [0.0]
    tableau[-1] = obj_row[:]

    # вычисляем приведённые стоимости (уменьшаем на c_B * A_B)
    cB = []
    for bv in basic_vars:
        if bv in all_vars:
            cB.append(orig_c[all_vars.index(bv)])
        else:
            cB.append(0.0)
    reduced = tableau[-1][:]
    for i,cb in enumerate(cB):
        if abs(cb) > EPS:
            reduced = [reduced[j] - cb * tableau[i][j] for j in range(len(reduced))]
    tableau[-1] = reduced

    # выполняем симплекс-итерации (фаза 2)
    tableau, basic_vars = simplex_iterations(tableau, basic_vars, all_vars, "Phase II")
    return tableau, basic_vars

# Извлечение решения
def extract_solution(tableau, basic_vars, all_vars):
    sol = {v:0.0 for v in all_vars}
    m = len(tableau)-1
    for i in range(m):
        sol[basic_vars[i]] = tableau[i][-1]
    Z = tableau[-1][-1]
    return sol, Z


def main():
    goal, obj, constraints = read_lp("input.txt")
    print("Цель:", goal)
    print("Целевая функция:", obj)
    print("Ограничения:")
    for c in constraints:
        print(" ", c)
    # приводим задачу к каноническому виду
    all_vars, A, b, c_vec, slacks, arts, row_types = build_canonical(goal, obj, constraints)
    print("\nПеременные (порядок столбцов):", all_vars)
    print("Искусственные переменные:", arts)
    # фаза 1
    if arts:
        all_vars_p, tableau_p, basic_p = phase_one(all_vars, A, b, c_vec, arts, row_types)
    else:
        # если искусственных переменных нет — формируем базис по slack-переменным
        basic_p = []
        for i in range(len(A)):
            found = None
            for j,v in enumerate(all_vars):
                if v.startswith('s') and abs(A[i][j])>EPS:
                    found = v
                    break
            if found is None:
                for j,v in enumerate(all_vars):
                    col = [A[r][j] for r in range(len(A))]
                    if abs(A[i][j]-1)<EPS and sum(abs(col[k]) for k in range(len(col)) if k!=i)<EPS:
                        found = v
                        break
            if found is None:
                found = f"b{i}"
            basic_p.append(found)
        tableau_p = [row[:] + [b_i] for row,b_i in zip(A,b)]
        tableau_p.append([0.0]* (len(all_vars)) + [0.0])
        all_vars_p = all_vars[:]

    # подготавливаем коэффициенты исходной целевой функции для фазы 2
    orig_c = []
    for v in all_vars_p:
        if v in all_vars:
            orig_c.append(c_vec[all_vars.index(v)])
        else:
            orig_c.append(0.0)

    # фаза 2
    tableau_final, basic_final = phase_two(all_vars_p, tableau_p, basic_p, orig_c, goal)

    # извлекаем итоговое решение
    sol_all, Z = extract_solution(tableau_final, basic_final, all_vars_p)
    x_vars = sorted([v for v in sol_all.keys() if v.startswith('x')],
                     key=lambda name: int(re.findall(r'\d+', name)[0]))
    print("\n====== Результат ======")
    for v in x_vars:
        print(f"{v:>4} = {sol_all.get(v,0.0):8.4f}")
    s_vars = [v for v in sol_all.keys() if v.startswith('s')]
    for v in sorted(s_vars, key=lambda name: int(re.findall(r'\d+', name)[0])):
        print(f"{v:>4} = {sol_all.get(v,0.0):8.4f}")
    print(f"Z  = {Z:8.4f}")

if __name__ == "__main__":
    main()
