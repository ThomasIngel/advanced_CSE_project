r"""
Implementation of the Howard algorithm
"""
import numpy as np
from .hjb_problem import HjbProblem
from typing import List
import scipy.sparse.linalg as spspla
from .optimization import get_optimal_control
from .hjb_operator import HjbOperator
from .solver_helper import Solver

def incoporate_dirichtlet_bcs(solver: Solver, Problem: HjbProblem, 
hjb_operator: HjbOperator) -> None:
    """
    Manipulates the right hand side of the linear system of the euqations for
    the dirichlet boundary values

    Parameters
    ----------

    solver : Solver
        Solver class where all the parameters for the howard algorithm are saved
    Problem : HjbProblem
        HjbProblem class where all the parameters from the problem are saved
    hjb_operator : HjbOperator
        HjbOperator class where the discretization is performed
    """
    # get boundaries
    u_D: np.ndarray = Problem.u_D(solver.t, Problem.x, Problem)

    # minus for the diags because they are negative
    if Problem.dimension == 1:
        u_D_l: float = u_D[0]
        u_D_r: float = u_D[1]
        solver.rhs[1] = solver.rhs[1] - Problem.h_t * Problem.theta \
            * hjb_operator.diags[0][1] * u_D_l
        solver.rhs[-2] = solver.rhs[-2] - Problem.h_t * Problem.theta \
            * hjb_operator.diags[2][-2] * u_D_r
        
    else:
        # collect top, right, bot and left bcs in this order
        u_D_t: np.ndarray = u_D[0]
        u_D_r: np.ndarray = u_D[1]
        u_D_b: np.ndarray = u_D[2]
        u_D_l: np.ndarray = u_D[3]

        # top boundarys
        solver.rhs[1,:] = solver.rhs[1,:] - Problem.h_t * Problem.theta \
            * hjb_operator.diags[0][1,:] * u_D_t

        # right boundaries
        solver.rhs[:,-2] = solver.rhs[:,-2] - Problem.h_t * Problem.theta \
            * hjb_operator.diags[3][:,-2] * u_D_r

        # bottom boundaries
        solver.rhs[-2,:] = solver.rhs[-2,:] - Problem.h_t * Problem.theta \
            * hjb_operator.diags[4][-2,:] * u_D_b

        # left boundaries
        solver.rhs[:,1] = solver.rhs[:,1] - Problem.h_t * Problem.theta \
            * hjb_operator.diags[1][:,1] * u_D_l

def gather_bcs(solver: Solver, Problem: HjbProblem) -> None:
    """
    Write boundary conditions in the right place of the solution

    Parameters
    ----------
        solver : Solver
            Solver class where all the parameters for the howard algorithm are saved

        Problem : HjbProblem
            HjbProblem class where all the parameters from the problem are saved
    """
    # gather bcs
    u_D: List[np.ndarray] = Problem.u_D(solver.t, Problem.x, Problem)

    if Problem.dimension == 1:
        solver.u_sol[0]= u_D[0]
        solver.u_sol[-1]= u_D[1]
    else:
        # write bcs in solution; order top, right, bottom, left
        solver.u_sol[0,:] = u_D[0]
        solver.u_sol[:,-1] = u_D[1]
        solver.u_sol[-1,:] = u_D[2]
        solver.u_sol[:,0] = u_D[3]

# def get_first_control(solver: Solver, Problem: HjbProblem, 
# hjb_operator: HjbOperator) -> None:
#     # set theta to 0 because we want the first control with the explicit method
#     theta: float = Problem.theta
#     Problem.theta = 0

#     # small timestep for stability reasons
#     h_t: float = 1e-12

#     # get initial conditions
#     solver.u_old_t = Problem.sol[0]
#     solver.u_old = solver.u_old_t

#     # get first control
#     solver.q_old = get_optimal_control(solver, Problem, hjb_operator)
#     solver.q = solver.q_old

#     for k in range(Problem.max_iter):

#         # get discretized operator
#         hjb_operator.construct_matrix(solver, Problem)

#         # get volume force
#         solver.f = Problem.f(solver.t, Problem.x, solver.q, Problem)
        
#         # calculate new function values
#         solver.u_new = (solver.I - h_t * hjb_operator.L) @ solver.u_old_t.ravel()  \
#             + h_t * solver.f.ravel()

#         # get new optimal control
#         solver.q_new = get_optimal_control(solver, Problem, hjb_operator)
#         solver.q = solver.q_new

#         norm: float = np.linalg.norm(solver.u_old.ravel() - solver.u_new)
#         if norm < Problem.tol:
#             solver.q_old_t = solver.q_new
#             solver.f_old_t = solver.f
#             break

#         solver.u_old = solver.u_new

#     if k == Problem.max_iter - 1:
#         raise ValueError("Howard algorithm did not converge")
#     Problem.q_opt.append(solver.q_new)
#     Problem.theta = theta

def solve_hjb(Problem: HjbProblem) -> None:
    """Implementation of the howard algorithm for the solution of the HJB

    Parameters
    ----------
    Problem : HjbProblem
        HjbProblem class where all the parameters from the problem are saved
    """
    # initialise Solver and HjbOperator class
    solver = Solver(Problem)
    hjb_operator = HjbOperator(Problem)

    # get initial conditions
    solver.u_old_t = Problem.sol[0]
    solver.u_old = solver.u_old_t
    solver.u_sol = solver.u_old_t

    # get first control
    solver.q_old = get_optimal_control(solver, Problem, hjb_operator)
    solver.q = solver.q_old
    solver.q_old_t = solver.q_old

    # time loop starting from the second time step
    for t in Problem.t_d[1:]:
        # actual time for the solver 
        solver.t = t

        # gather old solution for the termination criteria
        if Problem.dimension == 1:
            solver.u_old = solver.u_old_t[1:-1].ravel()
        else:
            solver.u_old = solver.u_old_t[1:-1,1:-1].ravel()

        solver.q_old = solver.q_old_t
        solver.q = solver.q_old

        # solution variable
        if Problem.dimension == 1:
            solver.u_sol: np.ndarray = np.empty(Problem.n_x)
        else:
            solver.u_sol: np.ndarray = np.empty((Problem.n_y, Problem.n_x))

        # get bcs
        gather_bcs(solver, Problem)

        # optimization loop
        for k in range(Problem.max_iter):
            # get discretized operator
            hjb_operator.construct_matrix(solver, Problem)

            # get implicit and explicit part
            solver.L_imp = solver.I + Problem.h_t * Problem.theta * hjb_operator.L
            solver.L_exp = solver.I - Problem.h_t * (1 - Problem.theta) \
                * hjb_operator.L

            # get volume force for the rhs
            if k == 0:
                solver.f = Problem.f(solver.t, Problem.x, solver.q, Problem)
                solver.f_old = Problem.f(solver.t - Problem.h_t, Problem.x, 
                solver.q, Problem)

            # get complete rhs
            solver.rhs =  solver.L_exp @ solver.u_old_t.ravel() \
                + Problem.h_t * Problem.theta * solver.f.ravel() \
                + Problem.h_t * (1 - Problem.theta) * solver.f_old.ravel()

            # incoporate bcs in the rhs
            if Problem.dimension == 2:
                solver.rhs = solver.rhs.reshape((Problem.n_y, Problem.n_x))
            incoporate_dirichtlet_bcs(solver, Problem, hjb_operator)

            # right format for the rhs
            if Problem.dimension == 1:
                b: np.ndarray = solver.rhs[1:-1]
            else:
                b: np.ndarray = solver.rhs[1:-1, 1:-1].ravel()

            # starting value for the gmres and outer v 
            if k == 0:
                prep_outer: bool = False
                if Problem.dimension == 1:
                    x0: np.ndarray = solver.u_old_t[1:-1].ravel()
                else:
                    x0: np.ndarray = solver.u_old_t[1:-1,1:-1].ravel()
                outer_v: List[tuple] = []
            else:
                x0: np.ndarray = solver.u_old.ravel()
                prep_outer: bool = True

            # solve lgs
#            solver.u_new, _ = spspla.lgmres(solver.L_imp[Problem.inds], b,
#            x0, tol=Problem.tol_gmres, atol=Problem.tol_gmres, outer_v=outer_v, 
#            prepend_outer_v=prep_outer)
#            print(_)
            solver.u_new, _ = spspla.gmres(solver.L_imp[Problem.inds], b, x0, 
            rtol=Problem.tol_gmres, atol=Problem.tol_gmres, maxiter=100)

            # save new solution in solution vector
            if Problem.dimension == 1:
                solver.u_sol[1:-1] = solver.u_new
            else:
                solver.u_sol[1:-1,1:-1] = solver.u_new.reshape((Problem.n_y - 2,
                Problem.n_x - 2))

            # get optimal control
            solver.q_new = get_optimal_control(solver, Problem, hjb_operator)
            solver.q = solver.q_new

            # termination criterion (L_\infty norm)
            norm: float = np.linalg.norm(solver.u_new - solver.u_old, ord=np.inf)
            if Problem.info == 1:
                print("k = ", k+1, "\t norm = ", norm)
                
            if norm < Problem.tol:
                print("Howard terminated after ", k+1, "iterations at time ", t)
                Problem.sol.append(solver.u_sol)
                Problem.q_opt.append(solver.q_new)
                solver.u_old_t = solver.u_sol
                solver.q_old_t = solver.q_new
                break

            # old iterations values are the new values from the current iteration
            solver.u_old = solver.u_new
            solver.q_old = solver.q_new