from __future__ import annotations

import copy


# Array and methods operating on array

class Array(list):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = self.__shape()
        self.ndim = self.__dimension()  # pyplot calls for ndim

    def _mul(self, item, factor):
        """Recursive method for multiplying last item of nested lists by factor"""
        if isinstance(item, list) or isinstance(item, Array):
            return Array([self._mul(x, factor) for x in item])
        else:
            return item * factor

    def __rmul__(self, other):
        """method implements the reverse multiplication operation that is multiplication with reflected,
         swapped operands. Its called when on the left operand __mul__ is not implemented"""
        out = copy.deepcopy(self)
        return out._mul(out, other)

    def __mul__(self, other):
        """Calculate scalar product of two 2D tensors (sign '*') or tensor and scalar"""
        out = []
        if not isinstance(other, (int, float, complex)):
            # multiply arrays
            if self.ndim == 1:
                out = 0
                for i in range(0, len(self)):
                    out += self[i] * other[i]
            else:
                if len(self.shape) > 1:
                    if self.shape[1] != other.shape[0]:
                        raise IndexError(
                            f"inconsistent arrays shapes (size {other.shape[0]} is different from {self.shape[1]})"
                        )
                    elif self.shape[0] != other.shape[1]:
                        raise IndexError(
                            f"inconsistent arrays shapes (size {self.shape[0]} is different from {other.shape[1]})"
                        )
                    for i in range(len(self)):
                        out.append([])
                        for j in range(len(other[0])):
                            out[i].append(0)
                            for k in range(len(other)):
                                out[i][j] += self[i][k] * other[k][j]
        else:
            # multiply array with scalar
            out = copy.deepcopy(self)
            out = out._mul(out, other)
        if not type(out) == float or type(out) == int:
            out = Array(out)
        return Array(out)

    def __matmul__(self, other):
        """Calculate vector product sign('@')"""
        try:
            return Array([self[1] * other[2] - self[2] * other[1], self[2] * other[0] - self[0] * other[2],
                          self[0] * other[1] - self[1] * other[0]]
                         )
        except:
            raise ValueError("can only calcualte product of 3D vectors ")

    def __radd__(self, other):
        """method implements the reverse sum operation that is sum with reflected,
         swapped operands. Its called when on the left operand __add__ is not implemented"""
        out = copy.deepcopy(self)
        return out._add(out, other)

    def _add(self, item, component):
        """Recursive method for addition last item of nested lists to component"""
        if isinstance(item, list) or isinstance(item, Array):
            return [self._add(x, component) for x in item]
        else:
            return item + component

    def __add__(self, other):
        """Calculate sum of two 2D tensors or tensor and scalar"""
        if type(other) == Array:
            if self.shape != other.shape:
                raise IndexError(f"inconsistent arrays shapes (size {other.shape} is different from {self.shape})")
            if len(self.shape) == 1:
                return Array([self[i] + other[i] for i in range(len(self))])
            out = []
            for i in range(len(self)):
                out.append([])
                for j in range(len(other[0])):
                    out[i].append(0)
                    out[i][j] = self[i][j] + other[i][j]
            return Array(out)
        else:
            if len(self.shape) == 1:
                return Array([self[i] + other for i in range(len(self))])
            out = []
            for i in range(len(self)):
                out.append([])
                for j in range(len(self[0])):
                    out[i].append(0)
                    out[i][j] = self[i][j] + other
            return Array(out)

    def __sub__(self, other):
        return Array([a - b for a, b in zip(self, other)])

    def dotProd(self, other: Array):
        """Returns dot product of two arrays"""
        """
            - If both `a` and `b` are 1-D arrays, it is inner product of vectors
            (without complex conjugation).
        """
        if (len(self) != len(other)):
            raise Exception("shapes and not aligned")
        dotProdRes = 0
        if (self.ndim == 1 and other.ndim == 1):
            for i in range(0, len(self)):
                dotProdRes += self[i] * other[i]
            return dotProdRes
        """
            - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
            but using :func:`matmul` or ``a @ b`` is preferred.
        """
        if (self.ndim == 2 and other.ndim == 2):
            return self * other
        """
            - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
             and using ``numpy.multiply(a, b)`` or ``a * b`` is preferred.
        """
        if (other.ndim == 1 and len(other) == 1):
            return self * other
        """
            - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
            the last axis of `a` and `b`.
        """
        if (self.ndim > 2 and other.ndim == 1):
            return Array(self[..., -self.ndim:]) + other
        """
        - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
        sum product over the last axis of `a` and the second-to-last axis of `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        """
        if (self.ndim > 2 and other.ndim > 1):
            raise Exception("not implemented yet")

    def __shape(self):
        """Calculates shape of array"""
        i = self
        shp = []
        while type(i) == Array and len(i) != 0:
            shp.append(len(i))
            if type(i[0]) == list or type(i[0]) == Array:
                i = Array(i[0])
            else:
                i = i[0]
        return tuple(shp)

    def __dimension(self) -> int:
        """Calculates array's ndim"""
        if len(self) != 0:
            i = self[0]
            counter = 1
            while type(i) == list or type(i) == Array:
                counter += 1
                i = i[0]
            return counter
        else:
            return 0

    def __replace(self, item, value):
        """Recursive method to substitute last item of nested lists with value"""
        if isinstance(item, list) or isinstance(item, Array):
            return [self.__replace(x, value) for x in item]
        else:
            return value

    def __pow__(self, other):
        """power operand"""
        if type(other) == Array:
            raise IndexError(f"unsupported operand type for ** : Array and Array")
        else:
            out = copy.deepcopy(self)
            out = out._pow(out, other)
        if not type(out) == float or type(out) == int:
            out = out
        return Array(out)

    def __rpow__(self, other):
        """method implements the reverse power operation that is power with reflected,
         swapped operands. Its called when on the left operand __pow__ is not implemented"""
        out = copy.deepcopy(self)
        return Array(out._pow(out, other))

    def _pow(self, item, power):
        """Recursive method for raising last item of nested lists to a power"""
        if isinstance(item, list) or isinstance(item, Array):
            return [self._pow(x, power) for x in item]
        else:
            return item ** power

    def tolist(self):
        return list(self)

    def transpose(self):
        """
        transpose matrix
        """
        if not isinstance(self[0], list):
            return Array(self.copy())
        tmp = [[self[j][i] for j in range(len(self))] for i in range(len(self[0]))]
        out = []

        for row in tmp:
            out.append(row)
        return Array(out)


def det(matrix):
    n = len(matrix)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    d = 0
    for i in range(n):
        submatrix = [row[:i] + row[i + 1:] for row in matrix[1:]]
        d += matrix[0][i] * (-1) ** i * det(submatrix)

    return d


def outer(array1, array2):
    """Compute the outer product of two arrays"""
    if not type(array1) == Array:
        array1 = Array(array1)
    out = []
    if array1.ndim >= 2:
        for i in array1:
            for j in i:
                out.append(Array([]))
                for k in array2:
                    for k2 in k:
                        out[-1].append(j * k2)
    else:
        for i in array1:
            out.append(Array([]))
            for k in array2:
                out[-1].append(i * k)
    return Array(out)


def ones_like(arr):
    """returns array of same ndim and filled with '1'"""
    c = copy.deepcopy(arr)
    return _replace(c, 1)


def _replace(item, value):
    if isinstance(item, list) or isinstance(item, Array):
        return [_replace(x, value) for x in item]
    else:
        return value


def full(array, value):
    """returns array of same ndim and filled with 'value'"""
    c = copy.deepcopy(array)
    return _replace(c, value)


def trace(array):
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        trace = 0
        for i, row in enumerate(array):
            trace += row[i]
        return trace
    else:
        raise Exception("array has to be 2-dimensional and square")


def mul(one, other):
    """Calculate product of same elements beetwen two 2D tensors"""
    out = []
    if one.shape != other.shape:
        raise ValueError(f"inconsistent arrays shapes (size {one.shape} is different from {other.shape})")
    else:
        if not isinstance(other, (int, float, complex)):
            # multiply arrays
            out = copy.deepcopy(one)
            if one.ndim == 1:
                for i in range(0, len(one)):
                    out[i] = one[i] * other[i]
            else:
                if len(one.shape) > 1:
                    for i in range(len(one)):
                        for j in range(len(one[0])):
                            out[i][j] = one[i][j] * other[i][j]
    return out
