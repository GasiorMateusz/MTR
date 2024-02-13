import copy
from decimal import Decimal, getcontext
from random import randint, random

from lib.array import Array, trace

pi = 355 / 113
e = 2.718281828459045
getcontext().prec += 1
K_pi = Decimal('3.14159265358979323846264')
K_pipol = Decimal(K_pi / 2)
K_2pi = Decimal(K_pi * 2)
K_speed = None


def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5


def euclidean_norm(vector):
    squared_sum = sum(coord ** 2 for coord in vector)
    norm = squared_sum ** 0.5
    return norm


class Solver:
    prec = 1e-14

    def __init__(self, prec, coeffs):
        self.coeffs = coeffs  # list of coefficients
        self.prec = prec
        pass

    def __fun(self, x):
        result = 0
        for ind, c in enumerate(self.coeffs):
            if ind == len(self.coeffs) - 1:
                result += c
            else:
                pow = (len(self.coeffs) - ind - 1)
                result += c * x ** pow
        return result

    def cubic(self):
        a, b, c, d = self.coeffs[0], self.coeffs[1], self.coeffs[2], self.coeffs[3]
        result = 0
        x0 = a ** (-1 / 3)  # start point
        n = 400
        while (n > 0):
            n -= 1
            f0 = x0 * (c + x0 * (b + x0 * a)) + d  # function(x0) f(x0)
            if abs(f0) < self.prec:
                result = 1
                break
            f1 = x0 * (2 * b + 3 * a * x0) + c  # derivative f'(x0)
            x1 = x0
            x0 -= f0 / f1  # f(x0)/f'(x0)
            if abs(x1 - x0) < self.prec:
                result = 1
                break
        if result:
            return self.quadratic(x0)
        return None

    def quadratic(self, x0=None):
        result = 1
        a, b, c = self.coeffs[0], self.coeffs[1], self.coeffs[2]
        if x0 != None:
            b = b + a * x0
            c = c + b * x0
        delta = b * b - 4 * a * c
        if abs(delta) < self.prec:
            x1 = x2 = - b / 2 / a
        elif delta < 0:
            result = 0
            x1 = -b / 2 / a
            x2 = (-delta) ** (0.5) / 2 / a
        else:
            delta = (delta) ** (0.5)
            x1 = (-b - delta) / 2 / a
            x2 = (-b + delta) / 2 / a
        if result:
            if x0 != None:
                return [x0, x1, x2]
            else:
                return [x1, x2]
        else:
            if x0 != None:
                return [x0, complex(x1, -x2), complex(x1, x2)]
            else:
                return [complex(x1, -x2), complex(x1, x2)]


class Polynomial:

    def __init__(self, *coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0
        """
        self.coefficients = list(coefficients)  # tuple is turned into a list

    def __call__(self, x):
        res = 0
        for index, coeff in enumerate(self.coefficients[::-1]):
            res += coeff * x ** index
        return res

    def __repr__(self):
        def x_expr(degree):
            if degree == 0:
                res = ""
            elif degree == 1:
                res = "x"
            else:
                res = "x^" + str(degree)
            return res

        degree = len(self.coefficients) - 1
        res = ""

        for i in range(0, degree + 1):
            coeff = self.coefficients[i]
            # nothing has to be done if coeff is 0:
            if abs(coeff) == 1 and i < degree:
                # 1 in front of x shouldn't occur, e.g. x instead of 1x
                # but we need the plus or minus sign:
                res += f"{'+' if coeff > 0 else '-'}{x_expr(degree - i)}"
            elif coeff != 0:
                res += f"{coeff:+g}{x_expr(degree - i)}"

        return res.lstrip('+')  # removing leading '+'


def polyDer(poly: Polynomial, order: int = 1):
    coeff = poly.coefficients
    newCoeffs = copy.deepcopy(coeff)
    for j in range(1, order + 1):
        del newCoeffs[-1]
        for i in range(len(newCoeffs)):
            newCoeffs[i] = newCoeffs[i] * (len(newCoeffs) - i)
    return Polynomial(*newCoeffs)  # [4, 1]


def sin(item):
    if isinstance(item, list) or isinstance(item, Array):
        return [sin(x) for x in item]
    else:
        return _sin(item)


def _sin(x):
    """K_sin(x) wyznacza sinus(x) dla x typu float lub int."""
    # zgodnie z teoria w pliku TeoriaTrysekcjiSinusa
    # błąd nie powinien przekroczyc 10E-17
    sgn = 1
    delta = float(x)
    if delta < 0.0:
        delta, sgn = -delta, -1
    delta = delta % K_2pi
    if delta > K_pi:
        sgn, delta = -sgn, delta - K_pi
    if delta > K_pipol:
        delta = K_pi - delta
    sqr = False
    if delta > K_pipol / 2.0:
        sqr, delta = True, K_pipol - delta
    if delta < 2.085771560746: k = 4  # delta i tak nie wieksza niz pi/4
    if delta < 0.785523097658: k = 3
    if delta < 0.295836106200: k = 2
    if delta < 0.111414931009: k = 1
    if delta < 0.041960012965: k = 0
    delta = delta / (3.0 ** k)
    deltasq = delta * delta
    s = delta * (1. + (deltasq * (-1. + (deltasq * (1. - deltasq / 42.)) / 20.)) / 6.)
    for i in range(1, k + 1):
        s = 4.0 * s * (0.75 - s * s)
    if sqr: s = (1.0 - s * s) ** (0.5)
    return sgn * s


def cos(item):
    if isinstance(item, list) or isinstance(item, Array):
        return [cos(x) for x in item]
    else:
        return _cos(item)


def _cos(x):
    return sin(Decimal(x) + K_pipol)


def arccos(item):
    if isinstance(item, list) or isinstance(item, Array):
        return [cos(x) for x in item]
    else:
        return _arccos(item)


def _arccos(xin):  # blad wzgledny sporadycznie przekracza 1.0E-15
    x = float(xin)
    y0 = K_pipol * Decimal(1. - x)
    if abs(x) >= 1. or x == 0.:
        return y0
    sgn = 1.
    if x < 0.: sgn = -sgn
    hmax = 10. ** (-4)
    h = 1. - abs(x)
    if h <= hmax:  # Taylor w niewielkim otoczeniu biegunow +/-1
        # blad wzgledny = eps --> h<5.64 eps**(1/3)
        # blad wzgledny 5.64E-15 dla hmax=10**(-4)
        dy = (2. * h) ** (0.5) * (1. + (h * (1. + (9. * (1. + (25. * h) / 84.) * h) / 40.)) / 12.)
        # print(K_pipol*(1.-sgn)+sgn*dy)
        return K_pipol * (1. - sgn) + sgn * dy
    else:  # poszukiwanie zera poza otoczeniem bieguna: odwzorowanie iterowane
        ynew = y0
        dy = 1.
        n = 0
        while dy > 10. ** (-15) and n < 24:  # zazwyczaj n=kilka
            yold = ynew
            siny = sin(yold)
            cosy = sgn * (1 - siny ** 2) ** 0.5
            corr1 = (cosy - x) / siny
            # approx hybrydowa
            if n > 0:
                corr = corr1  # approx liniowa
            else:  # wykonaj dokladniejsze oszacowanie wstepnego przyblizenia
                tany = siny / cosy
                corr2 = tany * ((1. + 2. * corr1 / tany) ** 0.5 - 1.)
                corr = corr2  # approx kwadratowa
                # wystarczy zastosowac ją w pierwszym kroku,
                # co znacznie przyspiesza zbieznosc dla duzych x
                # np x=0.9 z n=9 do n=5
                # np x=0.999 z n=10 do n=4
                # (approx kwadratowa dalaby n=4,
                # ale wymaga bardziej zlozonych  obliczeń
                # dla x=0.999:
                # mathematica     0.044725087168733|4312496962
                # python acos     0.044725087168733|454
                # n=10 approx1    0.044725087168733|2
                # n=3  approx2    0.044725087168734|21
                # n=4  approxHybr 0.044725087168733|67
            ynew = yold + Decimal(corr)
            dy = abs(corr)
            n = n + 1
    return ynew


def findPerpendicularVector(V):
    """
    find a random vector perpendicular to the given one
    :param V: vector [x,y,z]
    :return: perpendicular vector
    """
    L = randomVector()
    X = L + (-1 * L.transpose()).dotProd(V) / V.transpose().dotProd(V.transpose()) * V  #
    norX = X.dotProd(X) ** (0.5)
    o2 = X * norX ** (-1)
    return o2


def rotateByAngle(vector, angle, rotAxis):
    rii = lambda a, b: a ** 2 * (
            1 - cos(b) + cos(b))  # function for calculating diagonal element of rotatiom matrix (rotMatrix)
    # a is axis[] value, b i rotation angle
    rotMat = Array([
        [rii(rotAxis[0], angle),
         rotAxis[0] * rotAxis[1] * (1 - cos(angle)) - rotAxis[2] * sin(angle),
         rotAxis[0] * rotAxis[2] * (1 - cos(angle)) + rotAxis[1] * sin(angle)],
        [rotAxis[0] * rotAxis[1] * (1 - cos(angle)) + rotAxis[2] * sin(angle),
         rii(rotAxis[1], angle),
         rotAxis[1] * rotAxis[2] * (1 - cos(angle)) - rotAxis[0] * sin(angle)],
        [rotAxis[0] * rotAxis[2] * (1 - cos(angle)) - rotAxis[1] * sin(angle),
         rotAxis[1] * rotAxis[2] * (1 - cos(angle)) + rotAxis[0] * sin(angle),
         rii(rotAxis[2], angle)],
    ]
    )
    rotatedVector = vector * rotMat
    return Array(rotatedVector)


def linspace(start, end, numberOfPoints):
    delta = (end - start)
    div = numberOfPoints - 1
    list = []
    for i in range(0, numberOfPoints):
        list.append(i * delta / div + start)
    return Array(list)


def findEigenvalues(I):
    """Returns eigenValues of given array I"""
    # niezmienniki tensora
    I1 = I[0][0] + I[1][1] + I[2][2]
    I2 = I[0][0] * I[1][1] - I[1][0] * I[0][1] + \
         I[0][0] * I[2][2] - I[0][2] * I[2][0] + \
         I[1][1] * I[2][2] - I[1][2] * I[2][1]
    I3 = I[0][0] * I[1][1] * I[2][2] + \
         I[0][1] * I[1][2] * I[2][0] + \
         I[1][0] * I[2][1] * I[0][2] - \
         I[1][1] * I[0][2] * I[2][0] - \
         I[0][0] * I[1][2] * I[2][1] - \
         I[0][1] * I[1][0] * I[2][2]
    values = Solver([1, -I1, I2, -I3]).cubic()
    values = sorted(values, reverse=True)
    X = []
    for i in range(0, 2):
        X.append([])
        for j in range(0, 3):
            X[i].append(
                Array([I[j][0] - values[i] * (j == 0), I[j][1] - values[i] * (j == 1), I[j][2] - values[i] * (j == 2)])
            )
    W1 = X[0][0] @ X[0][1]
    W2 = X[1][2] @ X[1][0]
    e1 = W1 * (W1[0] ** 2 + W1[1] ** 2 + W1[2] ** 2) ** (-0.5)
    e2 = W2 * (W2[0] ** 2 + W2[1] ** 2 + W2[2] ** 2) ** (-0.5)
    e3 = e1 @ e2
    return {
        "versors": [e1, e2, e3],
        "values": values
    }


def matrixEigenvalues(A):
    """
    procedure for determining the eigenvalues of a symmetrical 3 * 3 matrix

    :param A: array
    :return: eigenvalue
    """
    t1 = trace(A)
    t2 = trace(A * A)
    t3 = trace(A * A * A)  ##### LB: tu powinno być mnożenie macierzowe
    ##### MG: zastosowane jest tutaj mnożenie macierzowe, odpowiadające numpy.matmul
    a = t1
    b = 1 / 2 * (t1 ** 2 - t2)
    c = 1 / 6 * (t1 ** 3 - 3 * t1 * t2 + 2 * t3)
    beta = (3 * t2 - t1 ** 2) ** 0.5
    y = 2 ** 0.5 * (9 * t3 - 9 * t1 * t2 + 2 * t1 ** 3) / beta ** 3
    gamma = Decimal(1 / 3) * arccos(y)
    lst = [-1, 0, 1]
    x = []
    for i in lst:
        tmp = cos(gamma + Decimal(2 * pi / 3 * i))  ##### LB: zamienic na K_cos
        ##### MG: metoda cos jest własną funkcją K_cos
        x.append(1 / 3 * a + (2 / 9) ** 0.5 * beta * tmp)
    x.sort()
    return x


def taylorSeriesExpansion(f, a, n, x):
    """
    Taylor's expansion around point a up to n degree for x.

    :param f: function
    :param a: point
    :param n: Taylor series degree
    :param x: value for which series value is calculated
    :return: result of taylor series expansion for x.
    """
    result = f(a)
    for i in range(1, n + 1):
        ith_derivative = calculateDerivative(f, i, a)
        result += (ith_derivative / factorial(i)) * (x - a) ** i
    return result


def calculateDerivative(f, order, point, h=1e-5):
    """
    Derivative of n degree of function f in point.

    :param f: function.
    :param order: derivative degree.
    :param point: point for which derivative is being calculated.
    :param h: Differentiation step.
    :return: diferation result.
    """
    if order == 0:
        return f(point)
    else:
        return (calculateDerivative(f, order - 1, point + h) - calculateDerivative(f, order - 1, point)) / h


def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def randNorm(b=100):
    """return random number within range <-1, 1>"""
    # xi=[xi−min(x)]/[max(x)−min(x)]
    return 2 * (randint(-b, b) + b) / (b + b) - 1


def sampleVectors(number):
    """return list of n 3d vectors"""
    vectors = [[], [], []]
    for d in range(0, number):
        tmp = randomVector()
        vectors[0].append(tmp[0])
        vectors[1].append(tmp[1])
        vectors[2].append(tmp[2])
    return vectors


def randomVectorBase():
    """return random vector base"""
    e1 = Array(randomVector())
    e2 = findPerpendicularVector(e1)
    e3 = e1 @ e2
    return [e1, e2, e3]


def randomVector():
    "randomize a point"
    theta = arccos(1 - 2 * random())
    phi = float(2 * K_pi * Decimal(random()))
    e3 = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
    return Array(e3)


def randomBasis():
    "randomize a base"
    theta = arccos(1 - 2 * random())
    phi = float(2 * K_pi * Decimal(random()))
    alfa = float(2 * K_pi * Decimal(random()))
    e3 = [sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)]
    eTheta = [cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)]
    ePhi = [-sin(phi), cos(theta), 0]
    e1 = []
    e2 = []
    for i in range(0, 3):
        e1.append(cos(alfa) * eTheta[i] + sin(alfa) * ePhi[i])
        e2.append(-sin(alfa) * eTheta[i] + cos(alfa) * ePhi[i])
    return [e1, e2, e3]
