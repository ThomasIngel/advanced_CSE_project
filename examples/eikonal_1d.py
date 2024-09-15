from source.hjb.hjb_problem import HjbProblem
import numpy as np
from source.hjb.hjb_solver import solve_hjb
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use("QtAgg")

# term in front of the first derivative
def dx(t, x, q, problem):
    dx_val = []
    dx_val.append(q * np.ones(np.shape(x[0])))

    return dx_val

def run_eikonal_1d():
    # initiazation of the problem class
    eikonal_1d = HjbProblem(1, 1, "max", "brute_all", 1, 0)

    # bound on the 1d domain and discretization parameters
    eikonal_1d.x_min = -1
    eikonal_1d.x_max = 1
    eikonal_1d.n_x = 201

    # end time and time discretization
    eikonal_1d.T = 1
    eikonal_1d.n_t = 101

    # bounds on the control and discretization
    eikonal_1d.q_min = -1
    eikonal_1d.q_max = 1
    eikonal_1d.n_q = 2

    # rhs of the equation
    eikonal_1d.f = lambda t, x, q, problem: np.ones(np.shape(x[0]))

    # initial conditions
    eikonal_1d.u_0 = lambda t, x, problem: np.zeros(np.shape(x[0]))

    # term in front of first derivative
    eikonal_1d.dx = dx

    # dirichlet boundaries
    eikonal_1d.u_D = lambda t, x, problem: [0, 0]

    # discretize and solve the problem
    eikonal_1d.check_input_and_discretize()
    print("start solving")
    solve_hjb(eikonal_1d)
    print("end solving")

    # print solution

    for i in range(len(eikonal_1d.sol)):
        plt.figure()

        plt.plot(eikonal_1d.x[0], eikonal_1d.sol[i])
        plt.grid()

        plt.figure()

        plt.plot(eikonal_1d.x[0], eikonal_1d.q_opt[i])
        plt.grid()

        plt.show()
        print(i)

        plt.close("all")