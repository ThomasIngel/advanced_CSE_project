import scipy.optimize as spop
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def termval(n_y, n_z):
    G: np.ndarray = np.zeros((n_y, n_z))
    xi: np.ndarray = np.zeros((n_y, n_z))

    integ: np.ndarray = np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    A_eq: np.ndarray = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 
        -1, -1])

    lb: np.ndarray = np.zeros(15) 
    lb[0] = 250
    lb[1] = 100
    lb[2] = 60

    ub: np.ndarray = np.array([500, 400, 600,
        15.82, 17.49, 19, 20.55, 19.63, 7.51,
        16.81, 16.59, 19.94, 18.45, 20.32, 7.89])

    y_d = np.linspace(-100, 200, n_y, True)
    z_d = np.linspace(-1600, 100, n_z, True)
    hT = 0.4559

    c: np.ndarray = np.zeros(15)
    c[0] = 25
    c[1] = 35
    c[2] = 60

    bp = np.array([0, 1.05, 1.94, 2.95, 3.95, 5.15])
    sp = np.array([0, 0.96, 1.75, 2.74, 3.86, 5.2])

    bounds = spop.Bounds(lb, ub)

    k = 0
    for i in range(n_y):
        # sell 
        c[3:9] = y_d[i] + hT + sp

        # buy
        c[9:] = -(y_d[i] - hT - bp)
        for j in range(n_z):
            k = k + 1

            # equality constraint
            b_eq = -z_d[j]
            const = spop.LinearConstraint(A_eq, b_eq, b_eq)

            # solve problem
            res = spop.milp(c, bounds=bounds, constraints=const, 
                integrality=integ, options={"disp": False})

            # gather results
            G[i,j] = -res.fun
            xi[i,j] = res.x[0] + res.x[1] + res.x[2]
            print("k = ", k)
    
    return G