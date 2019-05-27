from scipy.misc import derivative

def foo(x, y):
    return (x**2 + y**3)


def partial_derivative(func, var=1, point=[]):
    args = point[:]
    print("args", args)
    def wraps(x):
        print("x", x)
        args[var] = x
        print("wrap args", args)
        return func(*args)
    print(point[var])
    return "ee", derivative(wraps, point[var])

print(partial_derivative(foo, 0, [3,1]))
print(partial_derivative(foo, 1, [3,1]))

from sympy import symbols, diff
x, y = symbols('x y', real=True)
f = x**2 + y**3
print(diff(f, x))
print(diff(f, y))

