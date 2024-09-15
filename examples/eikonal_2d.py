from source.hjb.hjb_problem import HjbProblem
from source.hjb.hjb_solver import solve_hjb
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
matplotlib.use("QtAgg")

# right side of equation
def f(t, x, q, problem):
	return np.ones(np.shape(x[0]))

# first derivative
def dx(t, x, q, problem):
	dxv = []
	dxv.append(q * np.ones(np.shape(x[0])))
	dxv.append(q * np.ones(np.shape(x[0])))
	return dxv

# dirichlet boundary conditions
def u_D(t, x, problem):
	u_d = []
	u_d.append(np.zeros(problem.n_x))
	u_d.append(np.zeros(problem.n_y))
	u_d.append(np.zeros(problem.n_x))
	u_d.append(np.zeros(problem.n_y))

	return u_d

def run_eikonal_2d():
	eikonal_2d = HjbProblem(2, 1, "max", "brute", 1, 0.5)
	eikonal_2d.n_x = 201
	eikonal_2d.n_t = 101
	eikonal_2d.n_y = 201
	eikonal_2d.u_0 = lambda t, x, problem: np.zeros(np.shape(x[0]))
	eikonal_2d.u_D = u_D
	eikonal_2d.f = f
	eikonal_2d.dx = dx
	eikonal_2d.q_min = -1
	eikonal_2d.q_max = 1
	eikonal_2d.n_q = 2
	eikonal_2d.q_d = np.array([-1, 1])
	eikonal_2d.max_iter = 1000
	eikonal_2d.check_input_and_discretize()
	solve_hjb(eikonal_2d)

	fig1, ax1 = plt.subplots(subplot_kw={"projection":"3d"})
	for i in range(len(eikonal_2d.t_d)):
		ax1.clear()
		ax1.plot_surface(eikonal_2d.x[0], eikonal_2d.x[1], eikonal_2d.sol[i], cmap=cm.jet, edgecolor="black")
		plt.draw()
		plt.pause(0.0001)