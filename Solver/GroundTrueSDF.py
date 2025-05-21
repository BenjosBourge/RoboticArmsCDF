import numpy as np

# func to make sdf closer to GLSL

def vec2(x, y):
    return np.array([x, y])

def length(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2)

def length2(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def min(a, b):
    return np.minimum(a, b)

def max(a, b):
    return np.maximum(a, b)


# params must contain at [0] the radius of the circle
def sdCircle(x, y, params):
    return length2(x, y) - params[0]

# params must contain at [0] the width and at [1] the height of the box
def sdBox(x, y, params):
    d = vec2(abs(x) - params[0], abs(y) - params[1])
    return length(max(d, 0)) + min(max(d[0], d[1]), 0)

# params must contain at [0] the radius of the star
def sdStar(x, y, params):
    k1x = 0.809016994  # cos(π/ 5) = ¼(√5+1)
    k2x = 0.309016994  # sin(π/10) = ¼(√5-1)
    k1y = 0.587785252  # sin(π/ 5) = ¼√(10-2√5)
    k2y = 0.951056516  # cos(π/10) = ¼√(10+2√5)
    k1z = 0.726542528  # tan(π/ 5) = √(5-2√5)
    v1 = vec2(k1x, -k1y)
    v2 = vec2(-k1x, -k1y)
    v3 = vec2(k2x, -k2y)

    p = vec2(x, y)
    p[0] = abs(p[0])
    p -= 2.0 * max(np.dot(v1, p), 0.0) * v1
    p -= 2.0 * max(np.dot(v2, p), 0.0) * v2
    p[0] = abs(p[0])
    p[1] -= params[0]
    return length(p - v3 * np.clip(np.dot(p, v3), 0.0, k1z * params[0])) * np.sign(p[1] * v3[0] - p[0] * v3[1]) - params[1]


class GroundTrueSDF:
    def __init__(self):
        self._func = None
        self._params = []

    def setCircle(self, radius):
        self._func = sdCircle
        self._params = [radius]

    def setBox(self, width, height):
        self._func = sdBox
        self._params = [width, height]

    def setStar(self, radius, width):
        self._func = sdStar
        self._params = [radius, width]

    # solver function needed
    def copy(self):
        gts = GroundTrueSDF()
        gts._func = self._func
        gts._params = self._params.copy()
        return gts

    def solve(self, x, y):
        if self._func is None:
            return 0
        return self._func(x, y, self._params)

    def getLoss(self):
        return 0

    def get_wb_as_1D(self):
        return np.array(self._params).flatten()

    def set_wb_from_1D(self, datas):
        self._params = datas