from scipy.misc import derivative


def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx=1e-6)

if __name__ == "__main__":
    def foo(x, y):
        return (x ** 2 + y ** 3)

    print(partial_derivative(foo, 0, [3,1]))
    print(partial_derivative(foo, 1, [3,1]))

    from sympy import symbols, diff
    x, y = symbols('x y', real=True)
    f = x**2 + y**3
    print(diff(f, x))
    print(diff(f, y))

