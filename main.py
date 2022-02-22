from math import sqrt
from math import sin
import numpy as np
from matplotlib import pyplot as plt


def gaussian_quadrature(f, a, b): #liczenie całki przy użyciu metody kwadratury Gaussa
    x0 = -1 / sqrt(3)
    x1 = 1 / sqrt(3)
    tmp = (b - a) / 2
    val0 = f(((b - a) / 2) * x0 + (a + b) / 2)
    val1 = f(((b - a) / 2) * x1 + (a + b) / 2)
    return tmp * (val0 + val1)


def x_from_i(i, n): #zwracanie współrzędnej x i-tego punktu w przedziale
    return (2.0 * i) / n


def e(i, x, n): #wartośc i-tej funkcji e (numerowanych od e0) dla x współrzędnej
    if x < x_from_i(i - 1, n): return 0
    if x_from_i(i - 1, n) <= x < x_from_i(i, n): return (x - x_from_i(i-1,n)) / (2/n) #gdy funkcja e jest rosnąca
    if x_from_i(i, n) <= x <= x_from_i(i + 1, n): return (x_from_i(i+1,n) - x) / (2/n) # gdy funkcja e jest malejąca
    else: return 0


def e_prim(i,x,n): #wartość pochodnej i-tej funkcji e (numerowanych od e0) dla x współrzędnej
    if x <= x_from_i(i - 1, n): return 0
    if x_from_i(i - 1, n) < x < x_from_i(i, n): return n/2 #gdy funkcja e jest rosnąca
    if x_from_i(i, n) < x < x_from_i(i + 1, n): return -1 * (n/2) # gdy funkcja e jest malejąca
    else: return 0


def B(w,v,n): #wartość funkcji B
    f1 = lambda x: e_prim(v,x,n)*e_prim(w,x,n)
    f2 = lambda x: e(v,x,n)*e(w,x,n)

    count_from = max(0, x_from_i(w-1,n), x_from_i(v-1,n))
    count_to = min(x_from_i(w+1, n), x_from_i(v+1, n),2)
    return e(v,2,n)*e(w,2,n) + gaussian_quadrature(f1,count_from,count_to) - gaussian_quadrature(f2,count_from,count_to)


def L(v,n): #wartość funkcji L
    count_from = max(0, x_from_i(v - 1, n))
    count_to = min(x_from_i(v + 1, n), 2)
    # return (gaussian_quadrature(lambda x: sin(x)*e(v,x,n),count_from,count_to) - 2*e(v,2,n) + gaussian_quadrature(lambda x: 2*e(v,x,n),count_from,count_to)) #Warunek Dirichleta zmieniony na u(0)=2
    return (gaussian_quadrature(lambda x: sin(x)*e(v,x,n),count_from,count_to))


if __name__ == '__main__':
    n = 100 # n - elementów
    p = 101 # dokładność wykresu

    A = [[B(v+1,w+1,n) for v in range(n)] for w in range(n)]
    B = [L(v+1,n) for v in range(n)]

    X = np.linalg.solve(A,B)

    # val = [sum(X[j] * e(j+1,x_from_i(i+1,p),n) for j in range(n)) + 2 for i in range(p)] # warunek Dirichleta zmieniony na u(0)=2
    val = [sum(X[j] * e(j + 1, x_from_i(i + 1, p), n) for j in range(n)) for i in range(p)]
    x_axis = np.linspace(0,2,p)

    for _ in range(n): print(A[_])
    print()
    print(B)
    print()
    print(X)
    plt.plot(x_axis,val)
    plt.show()
