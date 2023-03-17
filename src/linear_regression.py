import numpy as np

def choose_closest(r, x):
    final = r[...,0]
    delta = np.absolute(r[...,0] - x)
    for i in range(1, r.shape[-1]):
        pick = np.absolute(r[...,i] - x) < delta
        final[pick] = r[...,i][pick]
        delta[pick] = np.absolute(r[...,i] - x)[pick]
    return final

# where all inputs are np.array objects
# https://en.wikipedia.org/wiki/Cubic_equation#General_cubic_formula
def cubic_roots(a, b, c, d):
    p = c/a - b*b/(3*a*a)
    q = 2*b*b*b/(27*a*a*a) - b*c/(3*a*a) + d/a
    ds = q*q/4 + p*p*p/27

    D = np.ones(ds.shape)*b/(3*a)

    root = np.zeros(ds.shape + (3,))

    s = ds > 0
    if np.count_nonzero(s) > 0:
        root[s,:] = (np.cbrt(-q[s]/2 - np.sqrt(ds[s])) + np.cbrt(-q[s]/2 + np.sqrt(ds[s])) - D[s])[...,None] * np.ones((root.shape))[s,:]

    m = np.logical_and(ds == 0, p != 0)
    if np.count_nonzero(m) > 0:
        root[m,0] = 3*q[m]/p[m]
        root[m,1:3] = -3*q[m]/(2*p[m])[...,None] * np.ones(ds.shape + (2,))[m,:]

    t = np.logical_and(ds < 0, p < 0)
    if np.count_nonzero(t) > 0:
        for k in range(3):
            root[t,k] = 2 * np.sqrt(-p[t]/3) * np.cos(1/3*np.arccos(3*q[t]/(2*p[t])*np.sqrt(-3/p[t])) - 2*math.pi*k/3) - D[t]

    return root

def quadratic_roots(a, b, c):
    C = b * b - 4 * a * c

    res = np.zeros(C.shape + (2,))
    res[...,0] = (-b + np.sqrt(C)) / (2 * a)
    res[...,1] = (-b - np.sqrt(C)) / (2 * a)
    return res

def roots(c, x):
    if c.shape[-1] == 4:
        rts = cubic_roots(c[...,3], c[...,2], c[...,1], c[...,0])
        return choose_closest(rts, x)
    elif c.shape[-1] == 3:
        rts = quadratic_roots(c[...,2], c[...,1], c[...,0])
        return choose_closest(rts, x)
    elif c.shape[-1] == 2:
        return -c[...,0] / c[...,1]

# expects NxM for a and b.
# expects M for d.
# rerturns a flag N that indicates whether each element should be kept
def trim_outliers_by_diff(a, b, d):
        diff = a - b
        mn = np.mean(diff, axis=0)
        std = np.std(diff, axis=0)

        inc = np.logical_and(mn - d*std < diff, diff < mn + d*std).all(axis=1)
        return inc

# is in a N by 1 vector.
def trim_outliers(i, d):
    m = np.mean(i, axis=0)
    std = np.std(i, axis=0)

    inc = np.logical_and(m - d*std < i, i < m + d*std).all(axis=-1)
    return inc

class LinearRegression():
    # order an array of X integers for the order of each independent variable.
    # an order of 2 indicates that terms x^2 and x will be used along with a constant.
    def __init__(self, order=np.array([2]), cross=False):
        self._order = order
        self._cross = cross
        self._remove_outliers = False
        self._vars = order.shape[0]

        if cross:
            C = 1
            for o in range(self._vars):
                self._order[o] = order[o]+1
                C *= self._order[o]
            self._num_terms = C
        else:
            self._num_terms = np.sum(order) + 1

        self._coeffs = np.zeros((self._num_terms, 1))

        # the powers of each dependent variable for each term
        self._powers = np.zeros((self._num_terms, self._vars))
        for i in range(self._num_terms):
            self._powers[i] = self._index_to_powers(i)

    def remove_outliers(self, enable):
        self._remove_outliers = enable
        return self

    def regression(self, x, y):
        terms = self._terms(x)
        Q, R = np.linalg.qr(terms)
        self._coeffs = np.linalg.inv(R).dot(np.transpose(Q)).dot(y)

        approx = terms.dot(self._coeffs)
        err = approx - y

        if self._remove_outliers:
            keep = trim_outliers(err, 1.5)
            err = x - y
            terms = self._terms(x[keep])
            Q, R = np.linalg.qr(terms)
            self._coeffs = np.linalg.inv(R).dot(np.transpose(Q)).dot(y[keep])
            approx = terms.dot(self._coeffs)
            err[keep] = approx - y[keep]
            return err, keep

        return err, np.ones((x.shape[0],), bool)

    def evaluate(self, x):
        terms = self._terms(x)
        return terms.dot(self._coeffs)

    # only applies if order is 3 or less
    # only finds the solution for a single variable (0-indexed)
    def reverse(self, x, var):
        # note: at this point self._order is the number of coeffs, not the order
        if self._order[var] > 4:
            return np.zeros(y.shape)

        C = np.zeros(x.shape[:-1] + (self._order[var],))
        C[...,0] = -1 * x[...,var]
        for i in range(self._num_terms):
            t = np.ones(x.shape[:-1])
            for v in range(self._vars):
                if v == var: continue
                t *= np.power(x[...,v], self._powers[i,v])
            ci = int(self._powers[i,var])
            C[..., ci] += t * self._coeffs[i,0]

        return roots(C, x[...,var])

    def to_dict(self):
        order_adj = -1 if self._cross else 0
        return {
            "order": (self._order + order_adj).tolist(),
            "coeffs": self._coeffs.tolist(),
            "cross": self._cross,
            "removeOutliers": self._remove_outliers
        }

    def from_dict(self, d):
        order = np.array(d["order"])
        cross = d["cross"]
        self.__init__(order, cross)
        self._coeffs = np.array(d["coeffs"])
        if "removeOutliers" in d:
            self._remove_outliers = d["removeOutliers"]
        return self

    def _index_to_powers(self, i):
        p = np.zeros((1, self._vars))
        for oi, o in enumerate(self._order):
            if self._cross:
                p[0,oi] = i % o
                i = int(i / o)
            else:
                if i > o:
                    p[0,oi] = 0
                    i -= o
                else:
                    p[0,oi] = i
                    break
        return p

    def _terms(self, x):
        terms = np.ones(x.shape[:-1] + (self._num_terms,), np.float32)
        for i in range(self._num_terms):
            for v in range(self._vars):
                terms[...,i] *= np.power(x[...,v], self._powers[i,v])
        return terms
