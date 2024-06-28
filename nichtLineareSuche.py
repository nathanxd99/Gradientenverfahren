import numpy as np

class GradientDescentWithFibonacci:
    """
    Implementiert den Gradientenabstieg mit der Fibonacci-Suche zur Optimierung einer Funktion.

    Args:
        func (callable): Die zu minimierende Funktion.
        grad (callable): Die Gradientenfunktion der zu minimierenden Funktion.
        tolerance (float, optional): Die Toleranz für den Abbruch des Gradientenabstiegs. Standardwert ist 1e-4.
        max_iters (int, optional): Die maximale Anzahl der Iterationen für den Gradientenabstieg. Standardwert ist 1000.
        fib_tolerance (float, optional): Die Toleranz für die Fibonacci-Suche. Standardwert ist 1e-4.
    """

    def __init__(self, func, grad, tolerance=1e-4, max_iters=1000, fib_tolerance=1e-4):
        self.func = func
        self.grad = grad
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.fib_tolerance = fib_tolerance

    def fibonacci_search(self, func, a, b, tolerance):
        """
        Führt die Fibonacci-Suche zur Minimierung einer Funktion auf einem Intervall durch.

        Args:
            func (callable): Die zu minimierende Funktion.
            a (float): Die linke Grenze des Intervalls.
            b (float): Die rechte Grenze des Intervalls.
            tolerance (float): Die Toleranz für die Suche.

        Returns:
            tuple: Das Minimum der Funktion und der Funktionswert an diesem Punkt.
        """
        fib_n_2 = 0
        fib_n_1 = 1
        fib_n = fib_n_1 + fib_n_2

        while fib_n < (b - a) / tolerance:
            fib_n_2 = fib_n_1
            fib_n_1 = fib_n
            fib_n = fib_n_1 + fib_n_2

        k = 0
        x1 = a + (fib_n_2 / fib_n) * (b - a)
        x2 = a + (fib_n_1 / fib_n) * (b - a)
        f1 = func(x1)
        f2 = func(x2)

        for _ in range(fib_n - 1):
            if f1 > f2:
                a = x1
                x1 = x2
                x2 = a + (fib_n_1 / fib_n) * (b - a)
                f1 = f2
                f2 = func(x2)
            else:
                b = x2
                x2 = x1
                x1 = a + (fib_n_2 / fib_n) * (b - a)
                f2 = f1
                f1 = func(x1)
            fib_n_1 -= 1
            fib_n_2 -= 1
            fib_n -= 1

        if f1 < f2:
            return x1, f1
        else:
            return x2, f2

    def minimize(self, x0):
        """
        Führt den Gradientenabstieg durch, um das Minimum der Funktion zu finden.

        Args:
            x0 (array-like): Der Startpunkt für den Gradientenabstieg.

        Returns:
            tuple: Der Punkt des lokalen Minimums und der Funktionswert an diesem Punkt.
        """
        x = np.array(x0, dtype=float)
        for i in range(self.max_iters):
            gradient = np.array(self.grad(*x))
            if np.linalg.norm(gradient) < self.tolerance:
                break

            def line_search_func(alpha):
                return self.func(*(x - alpha * gradient))

            alpha_opt, _ = self.fibonacci_search(line_search_func, 0, 1, self.fib_tolerance)
            new_x = x - alpha_opt * gradient

            if np.linalg.norm(new_x - x) < self.tolerance:
                break
            x = new_x
        return x, self.func(*x)

# Definition der Funktionen und ihrer Gradienten
def f1(x, y):
    """
    Eine Beispiel-Zielfunktion.

    Args:
        x (float): Die x-Koordinate.
        y (float): Die y-Koordinate.

    Returns:
        float: Der Wert der Funktion an Punkt (x, y).
    """
    return (x + 5) * (x + 1) * (x - 2) * (x + 4) * x * (y - 1) * (y + 2) * (y - 3) * (y + 5)

def grad_f1(x, y):
    """
    Berechnet den Gradienten der Funktion f1 numerisch.

    Args:
        x (float): Die x-Koordinate.
        y (float): Die y-Koordinate.

    Returns:
        ndarray: Der Gradient der Funktion an Punkt (x, y).
    """
    h = 1e-5
    df_dx = (f1(x + h, y) - f1(x, y)) / h
    df_dy = (f1(x, y + h) - f1(x, y)) / h
    return np.array([df_dx, df_dy])

def f2(x, y):
    """
    Eine weitere Beispiel-Zielfunktion.

    Args:
        x (float): Die x-Koordinate.
        y (float): Die y-Koordinate.

    Returns:
        float: Der Wert der Funktion an Punkt (x, y).
    """
    return np.sin(x**2 + y) - np.cos(y**2 - x)

def grad_f2(x, y):
    """
    Berechnet den Gradienten der Funktion f2 numerisch.

    Args:
        x (float): Die x-Koordinate.
        y (float): Die y-Koordinate.

    Returns:
        ndarray: Der Gradient der Funktion an Punkt (x, y).
    """
    h = 1e-5
    df_dx = (f2(x + h, y) - f2(x, y)) / h
    df_dy = (f2(x, y + h) - f2(x, y)) / h
    return np.array([df_dx, df_dy])

# Instanziierung und Verwendung des Minimierers
minimizer_f1 = GradientDescentWithFibonacci(f1, grad_f1)
minimizer_f2 = GradientDescentWithFibonacci(f2, grad_f2)

start_points_f1 = [(-2, -2), (-1, 1), (0, 0), (1, -1), (2, 2)]
start_points_f2 = [(-5 * np.pi, -5 * np.pi), (-2.5 * np.pi, 0), (0, 0), (2.5 * np.pi, 2.5 * np.pi), (5 * np.pi, -5 * np.pi)]

print("Minimierung von f1")
for sp in start_points_f1:
    minimum, value = minimizer_f1.minimize(sp)
    print(f"Startpunkt: {sp}, Minimum: {minimum}, Funktionswert: {value}")

print("\nMinimierung von f2")
for sp in start_points_f2:
    minimum, value = minimizer_f2.minimize(sp)
    print(f"Startpunkt: {sp}, Minimum: {minimum}, Funktionswert: {value}")
