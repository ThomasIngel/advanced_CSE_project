from source.hjb.hjb_problem import HjbProblem
import numpy as np
from source.hjb.milp_termval import termval
from source.hjb.hjb_solver import solve_hjb
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use("QtAgg")

def f(t, x, q, intraday):
    y = x[1]
    h = intraday.h(t)
    k = intraday.k(t)
    phi = intraday.phi(k, q)

    return (-(y + np.sign(q) * h + phi) * q)

def dx(t, x, q, intraday):
    dxv = []
    dxv.append(q)

    b = intraday.b
    psi = intraday.psi(b, q)
    dxv.append(intraday.mu_y + psi)

    return dxv

def dxx(t, x, q, intraday):
    dxxv = []
    ddy = 0.5 * intraday.sig_y**2
    ddz = 0.5 * intraday.sig_D**2
    dxxv.append(ddz * np.ones(np.shape(x[0])))
    dxxv.append(ddy * np.ones(np.shape(x[1])))

    return dxxv

def u_0(T, x, intraday):
    if intraday.n_x != 301 or intraday.n_y != 101:
        return termval(intraday.n_y, intraday.n_x)
    else:
        return np.load("termval.npy")

def u_D(t, x, intraday):
    dir_val = []
    G = intraday.G
    dir_val.append(G[0,:] * 0 - 100000)
    dir_val.append(G[:,-1] * 0 - 100000)
    dir_val.append(G[-1,:] * 0 - 100000)
    dir_val.append(G[:,0] * 0 - 100000)
    return dir_val

def q_optim(t, x, du, intraday):
    dufx = du[0][0]
    dufy = du[0][1]
    dubx = du[1][0]
    duby = du[1][1]

    dux = (dufx - dubx) / 2
    duy = (dufy - duby) / 2

    y = intraday.x[1]
    h = intraday.h(t)
    k = intraday.k(t)
    b = intraday.b

    q_minus = (-y + h + b * duby + dubx) / (2 * k)
    q_plus = (-y - h + b * dufy + dufx) / (2 * k)

    q_minus[q_minus < -50] = -50
    q_minus[q_minus > -0.3] = -0.3
    q_plus[q_plus < 0.3] = 0.3
    q_plus[q_plus > 50] = 50

    f_val = np.zeros((intraday.n_y, intraday.n_x, 3))
    f_val[:,:,0] = -(-(y - h + k * q_minus) * q_minus) + (b * q_minus) * duby \
        + q_minus * dubx
    f_val[:,:,2] = -(-(y + h + k * q_plus) * q_plus) + (b * q_plus) * dufy \
        + q_plus * dufx

    ind_q = np.argmin(f_val, 2)

    q_tmp = np.zeros((intraday.n_y, intraday.n_x))
    q_tmp[ind_q == 0] = q_minus[ind_q == 0]
    q_tmp[ind_q == 1] = 0
    q_tmp[ind_q == 2] = q_plus[ind_q == 2]

    return q_tmp

def run_intraday():
    # initialize problem class
    intraday = HjbProblem(2, 2, "min", "user_func", 1, 1)

    # time disc
    intraday.T = 17.5
    intraday.n_t = 1001

    # x disc
    intraday.x_min = -1600
    intraday.x_max = 100
    intraday.n_x = 301

    # y disc
    intraday.y_min = -100
    intraday.y_max = 200
    intraday.n_y = 101

    # howard parameters
    intraday.tol = 1e-6
    intraday.max_iter = 1000
    intraday.tol_gmres = 1e-4

    # control disc
    intraday.q_max = 50
    intraday.q_min = -50
    intraday.n_q = 101

    # hjb parameters
    intraday.h = lambda t: 2.11e1 - 7.46 * t + 1.36 * t**2 - 1.01e-1 * t**3 \
        - 1.06e-3 * t**4 + 6.3e-4 * t**5 - 3.59e-5 * t**6 + 6.59e-7 * t**7

    intraday.k = lambda t: -0.5 * (-8.72e-2 + 1.67e-2 * t - 5.28e-4 * t**2 
        - 4.15e-4 * t**3 + 5.012e-5 * t**4 - 1.95e-6 * t**5 + 2.36e-8 * t**6)

    intraday.b = 0.0017

    intraday.phi = lambda k, q: k * q
    intraday.psi = lambda b, q: b * q

    intraday.mu_y = 0.0433
    intraday.sig_y = 1.06
    intraday.sig_D = 5

    # user defined function
    intraday.u_0 = u_0
    intraday.u_D = u_D
    intraday.f = f
    intraday.dx = dx
    intraday.ddx = dxx
    intraday.q_opt_user = q_optim

    intraday.G = intraday.u_0([], [], intraday) 

    # check inputs discretize and solve
    intraday.check_input_and_discretize()
    solve_hjb(intraday)

    # plot solutioan at t = 0
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    plt.ion()
    plt.show()
    for i in range(len(intraday.sol)):
        ax1.clear()
        ax1.plot_surface(intraday.x[0][5:-5,5:-5], intraday.x[1][5:-5,5:-5], intraday.sol[i][5:-5,5:-5], 
            cmap = cm.jet, edgecolor="black")
        plt.draw()
        plt.pause(0.0001)

        ax2.clear()
        ax2.plot_surface(intraday.x[0][5:-5,5:-5], intraday.x[1][5:-5,5:-5], intraday.q_opt[i][5:-5,5:-5], 
            cmap = cm.jet, edgecolor="black")
        plt.draw()
        plt.pause(0.0001)

        print(i)