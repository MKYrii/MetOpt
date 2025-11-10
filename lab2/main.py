import numpy as np
import matplotlib.pyplot as plt
import time

# Функция парсинга пользовательского ввода
def parse_function(func_str):
    """
    Преобразует строку функции (например, "x + np.sin(3.14*x)")
    в исполняемую Python-функцию f(x)
    """
    def f(x):
        return eval(func_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi})
    return f


# Автоматическая оценка константы Липшица
def estimate_lipschitz(f, a, b, n_points=1000):
    """
    Приближённая оценка константы Липшица L:
    L ≈ max(|f(x_i) - f(x_{i-1})| / |x_i - x_{i-1}|) по равномерной сетке
    """
    xs = np.linspace(a, b, n_points)
    ys = [f(x) for x in xs]
    diffs = np.abs(np.diff(ys)) / np.diff(xs)
    L_est = np.max(diffs)
    return L_est


# Метод ломаных
def broken_line_method(func, a, b, L=None, eps=0.01, max_iter=1000):
    start_time = time.time()

    # Если L не задан — оцениваем автоматически
    if L is None:
        L = estimate_lipschitz(func, a, b) * 1.05
        print(f"Оцененная константа Липшица: L ≈ {L:.4f}")

    X = [a, b]
    Y = [func(a), func(b)]
    iterations = 0

    while True:
        iterations += 1
        R = []
        for i in range(len(X) - 1):
            xi, xj = X[i], X[i + 1]
            yi, yj = Y[i], Y[i + 1]
            x_mid = 0.5 * (xi + xj) - 0.5 * (yj - yi) / L
            m_val = 0.5 * (yi + yj) - (L / 2) * (xj - xi)
            R.append((x_mid, m_val, xi, xj))

        # минимальная нижняя оценка
        x_next, m_next, seg_left, seg_right = min(R, key=lambda t: t[1])
        y_next = func(x_next)

        X.append(x_next)
        Y.append(y_next)
        X, Y = map(list, zip(*sorted(zip(X, Y))))

        # обновлённые границы
        f_best = min(Y)
        m_min = min(r[1] for r in R)

        # критерий остановки: разница между верхней и нижней оценкой <= eps
        if f_best - m_min <= eps or iterations >= max_iter:
            break

    total_time = time.time() - start_time
    xmin = X[np.argmin(Y)]
    fmin = min(Y)

    return {
        "x_min": xmin,
        "f_min": fmin,
        "iterations": iterations,
        "time": total_time,
        "X": np.array(X),
        "Y": np.array(Y),
        "L": L
    }


# Пример использования
if __name__ == "__main__":
    # функция Растригина (1D)
    func_str = "x**2 - 10*np.cos(2*np.pi*x) + 10"
    # func_str = "x**4 - 3*x**3 + 2"
    # func_str = "np.sin(5*x) + np.sin(2*x) + 0.1*x"
    f = parse_function(func_str)

    a, b = -5, 5 # область на графике
    eps = 0.01

    result = broken_line_method(f, a, b, L=None, eps=eps)  # L автоматически

    # Визуализация
    x_vals = np.linspace(a, b, 1000)
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {func_str}", color="blue")
    plt.plot(result["X"], result["Y"], 'ro--', label="Точки ломаной", alpha=0.8)
    plt.scatter(result["x_min"], result["f_min"], color="red", s=100, label="Найденный минимум")
    plt.title("Метод Ломаных")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.text(a + 0.1*(b - a), min(y_vals) + 0.1*(max(y_vals)-min(y_vals)),
             f"L ≈ {result['L']:.3f}\nε = {eps}\nИтерации = {result['iterations']}",
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.show()

    print(f"\n==== РЕЗУЛЬТАТЫ ====")
    print(f"Функция: f(x) = {func_str}")
    print(f"Приближённый минимум: x = {result['x_min']:.5f}, f(x) = {result['f_min']:.5f}")
    print(f"Оценка Липшица: L ≈ {result['L']:.4f}")
    print(f"Число итераций: {result['iterations']}")
    print(f"Время работы: {result['time']:.4f} сек.")
