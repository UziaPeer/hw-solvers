import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


def solve_with_root(a, b):
    """
    Solves the linear system ax=b using scipy.optimize.root.

    >>> a = np.array([[2, 0], [0, 4]])
    >>> b = np.array([6, 8])
    >>> solve_with_root(a, b)
    array([3., 2.])
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    def f(x):
        return a @ x - b

    sol = root(f, np.zeros(len(b)))

    if not sol.success:
        raise ValueError("root did not find a solution")

    return sol.x


def test_random_inputs():
    """
    Tests solve_with_root on random inputs by comparing it to numpy.linalg.solve.
    """
    for n in range(1, 11):
        for i in range(5):
            a = np.random.rand(n, n)
            a = a + n * np.eye(n)
            b = np.random.rand(n)

            x1 = solve_with_root(a, b)
            x2 = np.linalg.solve(a, b)

            assert np.allclose(x1, x2)

    print("All random tests passed")


def average_time(func, n, repeats=3):
    """
    Returns the average running time of func on random input of size n.
    """
    total = 0

    for i in range(repeats):
        a = np.random.rand(n, n)
        a = a + n * np.eye(n)
        b = np.random.rand(n)

        start = time.time()
        func(a, b)
        end = time.time()

        total += end - start

    return total / repeats


def compare_times():
    """
    Compares the running times of solve_with_root and numpy.linalg.solve
    and saves the graph to comparison.png.
    """
    sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    root_times = []
    numpy_times = []

    for n in sizes:
        print("checking size", n)

        numpy_times.append(average_time(np.linalg.solve, n))
        root_times.append(average_time(solve_with_root, n))

    plt.plot(sizes, numpy_times, marker="o", label="numpy.linalg.solve")
    plt.plot(sizes, root_times, marker="o", label="scipy.optimize.root")

    plt.xlabel("matrix size")
    plt.ylabel("average time in seconds")
    plt.title("Comparison of solvers")
    plt.legend()
    plt.grid(True)

    plt.savefig("comparison.png")
    plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    test_random_inputs()
    compare_times()# Put your code here 
