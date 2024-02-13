import copy
import random
from lib.array import Array
from lib.mathLib import sin, linspace
import matplotlib.pyplot as plt

"""
C. H. Reinsch's cubic smoothing spline algorithm.
"""


def reinschProcedure(data, ss=0):
    """
    smooth = dict {n1,n2}
        n1,n2 number of first and last data point n2>n1
    data = dict {x,y,dy,s}
        arrays with x[i],y[i 1, dy[i]  as abscissa, ordinate and relative
        weight of i-th data point, respectively, ( i = n1 (1) n2). The components
        of the array x must be strictly increasing.
        a non-negative parameter which controls the extent of smoothing: the
        spline function / is determined such that
        n2
        ∑   ((f(x[i]) - y[i])/dy[i]) -> 2 <= s,
        i=n1
        where equality holds unless f describes a straight line.
    @:return
        a, b, c, d arrays, collecting the coefficients of the cubic spline f
    """

    x = [i[0] for i in data]
    y = [i[1] for i in data]
    dy = [i[2] for i in data]
    s = ss
    n1 = 0
    n2 = len(x) - 1
    if s == 0:
        s = n2 + 1 - (n2 + 1) ** 0.5

    for i in range(len(x) - 1):
        if (x[i + 1] - x[i]) < 0: raise ValueError("the sequence of x[i] must be strictly increasing")
    a, b, c, d, r, r1, r2, t, t1, u, v = ([0 for i in range(len(x))] for i in range(11))

    r2.append(0)
    u.append(0)
    p = 0

    m1 = n1 + 1
    m2 = n2 - 1

    h: float = x[m1] - x[n1]
    f: float = (y[m1] - y[n1]) / h
    for i in range(m1, m2 + 1):
        g = h
        h = x[i + 1] - x[i]
        e = f
        try:
            f = (y[i + 1] - y[i]) / h
        except:
            print("h is equal 0")
        a[i] = f - e
        t[i] = 2 * (g + h) / 3
        t1[i] = h / 3
        r2[i] = dy[i - 1] / g
        r[i] = dy[i + 1] / h
        r1[i] = -dy[i] / g - dy[i] / h
    for i in range(m1, m2 + 1):
        b[i] = r[i] ** 2 + r1[i] ** 2 + r2[i] ** 2
        c[i] = r[i] * r1[i + 1] + r1[i] * r2[i + 1]
        d[i] = r[i] * r2[i + 2]
    f2 = -s
    n = 0

    while n < 100:
        n += 1
        for j in range(m1, m2 + 1):
            r1[j - 1] = f * r[j - 1]
            r2[j - 2] = g * r[j - 2]
            r[j] = 1 / (p * b[j] + t[j] - f * r1[j - 1] - g * r2[j - 2])
            u[j] = a[j] - r1[j - 1] * u[j - 1] - r2[j - 2] * u[j - 2]
            f = p * c[j] + t1[j] - h * r1[j - 1]
            g = h
            h = d[j] * p
        for k in reversed(range(m1, m2 + 1)):
            u[k] = r[k] * u[k] - r1[k] * u[k + 1] - r2[k] * u[k + 2]
        e = 0
        h = 0
        for j in range(n1, m2 + 1):
            g = h
            h = (u[j + 1] - u[j]) / (x[j + 1] - x[j])
            v[j] = (h - g) * dy[j] * dy[j]
            e = e + v[j] * (h - g)
        v[n2] = -h * dy[n2] * dy[n2]
        g = v[n2]
        e = e - g * h
        g = f2
        f2 = e * p ** 2
        if f2 >= s or f2 <= g: break
        f = 0
        h = (v[m1] - v[n1]) / (x[m1] - x[n1])
        for j in range(m1, m2 + 1):
            g = h
            h = (v[j + 1] - v[j]) / (x[j + 1] - x[j])
            g = h - g - r1[j - 1] * r[j - 1] - r2[j - 2] * r[j - 2]
            f = f + g * r[j] * g
            r[j] = g
        h = e - p * f
        if h <= 0: break
        p = p + (s - f2) / (((s / e) ** 0.5 + p) * h)
    for i in range(n1, n2 + 1):
        a[i] = y[i] - p * v[i]
        c[i] = u[i]
    for i in range(n1, m2 + 1):
        h = x[i + 1] - x[i]
        d[i] = (c[i + 1] - c[i]) / (3 * h)
        b[i] = (a[i + 1] - a[i]) / h - (h * d[i] + c[i]) * h
    return {
        # a - a
        # b - x
        # c - x**2
        # d - x**3
        "d": a,
        "c": b,
        "b": c,
        "a": d,
    }


def kSmooth(XYin_, Xout_, sigma):
    """
     (*XYin: macierz wspolrzednych punktow funkcji dyskretnej XY *)
     (*Xout: wektor odcietych X, dla których wyznaczamy wartość wygładzonej funkcji,
     długość Xout moze być różna od długości XYin *)
     (*σ: liczba lub wektor o długości równej długości XYin *)
     (*σ to wagi zdefiniowane dla odciętych funkcji XY -- to co Reinsch oznacza przez δy -2 *)
     (*deriv: =k=1, 2, 3 gdy na wyjściu potrzebujemy pochodnych do rzędu k włącznie *)
     """
    XYin = copy.deepcopy(XYin_)
    Xout = copy.deepcopy(Xout_)
    if type(sigma) == int or type(sigma) == float:
        XYin.sort()
        for i in XYin:
            i.append(sigma)
    else:
        for index, pointCoordinates in enumerate(XYin):
            pointCoordinates.append(sigma[index])
        XYin.sort()
    DAT = copy.deepcopy(XYin)
    cf = reinschProcedure(DAT, 1 + len(DAT))
    xi = [i[0] for i in DAT]  # OK
    xmin = xi[0]
    xmax = xi[-1]

    def F(x):
        i = 0
        for j in range(len(xi)):
            if xi[j] <= x:
                i = j
        if xi[i] == xmax:
            i = i - 1
        h = x - xi[i]
        h2 = h * h
        tmp = [
            [1, h, h2, h * h2],
            [0, 1, 2 * h, 3 * h2],
            [0, 0, 2, 6 * h],
            [0, 0, 0, 6]
        ]
        smooth_and_der = []

        coeff = Array([cf["d"][i], cf["c"][i], cf["b"][i], cf["a"][i]])

        for k in tmp:
            smooth_and_der.append(coeff.dotProd(Array(k)))
        return smooth_and_der

    valid_Xout = []
    for k in Xout:
        acc = 13
        if round(k, 15) >= round(xmin, acc) and round(k, 15) <= round(xmax, acc):
            valid_Xout.append(k)
        else:
            print(
                "Element : " + str(round(k, 1)) + " from reference points (Xout) is not within x range : < " + str(
                    round(xmin, acc)) +
                " , " + str(round(xmax, acc)) + " >")
    valid_Xout = list(dict.fromkeys(valid_Xout))
    valid_Xout.sort()
    result = []
    for k in valid_Xout:
        result.append(F(k))

    """
    find all elements in Xout that meet xmin ≤  ≤ xmax &
    sort selected items and drop duplicates
    apply F[] for every element in list
    """

    return {
        "smoothData": Array(copy.deepcopy(result)).transpose(),
        "coeff": cf
    }
