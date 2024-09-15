import numpy as np
from .hjb_problem import HjbProblem
from typing import List
import scipy.sparse as spsp
from .solver_helper import Solver
from .hjb_operator import HjbOperator
import scipy.optimize as spop


def get_optimal_control(Solver: Solver, Problem: HjbProblem, 
HjbOperator: HjbOperator) -> np.ndarray:
    """Wrapper to call the right optimization routine

    Parameters
    ----------
    Solver : Solver
        Solver class
    Problem : HjbProblem
        HjbProblem class
    HjbOperator : HjbOperator
        HjbOperator class

    Returns
    -------
    np.ndarray
        Return the optimal control
    """
    
    if Problem.optim_method == "brute":
        q_opt: np.ndarray = brute(Solver, Problem)
    elif Problem.optim_method == "scipy_scalar":
        q_opt: np.ndarray = scalar(Solver, Problem)
    elif Problem.optim_method == "brute_all":
        q_opt: np.ndarray = brute_all(Solver, Problem, HjbOperator)
    elif Problem.optim_method == "user_func":
        q_opt: np.ndarray = optim_user_func(Solver, Problem)

    return q_opt

def get_derivatives(u: np.ndarray, Problem: HjbProblem) -> List[np.ndarray]:
    """Function to get the different finite differences and bring them into the 
       right structure

    Parameters
    ----------
    u : np.ndarray
        current solution of the hjb
    Problem : HjbProblem
        HjbProblem class

    Returns
    -------
    List[List[np.ndarray]]
        Returns a list of a list of ndarrays in the following order

        First dimension:

            1. forward differences

            2. backward differences

            3. second order differences (if necessary)

        Second dimension:

            1. first spatial dimension

            2. second spatial dimension
    """
    du: List[List[np.ndarray]] = []
    du.append(forward_differences(u, Problem))
    du.append(backward_differences(u, Problem))
    if Problem.order == 2:
        du.append(second_order_differences(u, Problem))
    
    return du

def forward_differences(u: np.ndarray, Problem: HjbProblem) -> List[np.ndarray]:
    """calculate the forward finite differences of the current solution

    Parameters
    ----------
    u : np.ndarray
        current solution of the hjb
    Problem : HjbProblem
        HjbProblem class

    Returns
    -------
    List[np.ndarray]
        Returns the forward differences with the first spatial dimension in the 
        first list entry and the second spatial dimension in the secon if the
        dimension of the problem is two
    """
    # initialisze list for the forward differences 
    duf: List[np.ndarray] = []

    # check dimension
    if Problem.dimension == 1:
        # need the first point as would we do the finite differences with a 
        # matrix vector multiplikation
        _duf: np.ndarray = np.zeros(Problem.n_x)
        _duf[-1] = u[-1] / Problem.h_x
        _duf[:-1] = (u[:-1] - u[1:]) / Problem.h_x
        duf.append(_duf)

    if Problem.dimension == 2:
        # boundarys as above and forward differences in x direction
        _dufx: np.ndarray = np.zeros((Problem.n_y, Problem.n_x))
        _dufx[:,-1] = u[:,-1] / Problem.h_x
        _dufx[:,:-1] = (u[:,:-1] - u[:,1:]) / Problem.h_x
        duf.append(_dufx)

        # y direction
        _dufy: np.ndarray = np.zeros((Problem.n_y, Problem.n_x))
        _dufy[-1,:] = u[-1,:] / Problem.h_y
        _dufy[:-1,:] = (u[:-1,:] - u[1:,:]) / Problem.h_y
        duf.append(_dufy)

    return duf

def backward_differences(u: np.ndarray, Problem: HjbProblem) -> \
    List[np.ndarray]:
    """calculate the backwards finite differences of the current solution

    Parameters
    ----------
    u : np.ndarray
        current solution of the hjb
    Problem : HjbProblem
        HjbProblem class

    Returns
    -------
    List[np.ndarray]
        Returns the backwards differences with the first spatial dimension in the 
        first list entry and the second spatial dimension in the secon if the
        dimension of the problem is two
    """
    # initialize list for backward differences 
    dub: List[np.ndarray] = []

    # check dimension
    if Problem.dimension == 1:
        # as in forward differences 
        _dub: np.ndarray = np.zeros(Problem.n_x)
        _dub[0] = u[0] / Problem.h_x	
        _dub[1:] = (u[1:] - u[:-1]) / Problem.h_x
        dub.append(_dub)

    if Problem.dimension == 2:
        # x direction
        _dubx: np.ndarray = np.zeros((Problem.n_y, Problem.n_x))
        _dubx[:,0] = u[:,0] / Problem.h_x
        _dubx[:,1:] = (u[:,1:] - u[:,:-1]) / Problem.h_x
        dub.append(_dubx)

        # y direction
        _duby: np.ndarray = np.zeros((Problem.n_y, Problem.n_x))
        _duby[0,:] = u[0,:] / Problem.h_y
        _duby[1:,:] = (u[1:,:] - u[:-1,:]) / Problem.h_y
        dub.append(_duby)
    return dub

def second_order_differences(u: np.ndarray, Problem: HjbProblem) -> \
    List[np.ndarray]:
    """calculate the second order finite differences of the current solution

    Parameters
    ----------
    u : np.ndarray
        current solution of the hjb
    Problem : HjbProblem
        HjbProblem class

    Returns
    -------
    List[np.ndarray]
        Returns the second order differences with the first spatial dimension in the 
        first list entry and the second spatial dimension in the secon if the
        dimension of the problem is two
    """
    # initialize list for second order differences 
    ddu: List[np.ndarray] = []

    # check dimensions
    if Problem.dimension == 1:
        # forward and backward differences for boundarys
        _ddu: np.ndarray = np.zeros(Problem.n_x)
        _ddu[0] = (-u[0] + 2 * u[1] - u[2]) / (Problem.h_x ** 2)
        _ddu[-1] = (-u[-1] + 2 * u[-2] - u[-3]) / (Problem.h_x ** 2)
        _ddu[1:-1] = (-u[:-2] + 2 * u[1:-1] - u[2:]) / (Problem.h_x ** 2)
        ddu.append(_ddu)

    if Problem.dimension == 2:
        # x direction
        _ddux: np.ndarray = np.zeros((Problem.n_y, Problem.n_x))
        _ddux[:,0] = (-u[:,0] + 2 * u[:,1] - u[:,2]) / (Problem.h_x ** 2)
        _ddux[:,-1] = (-u[:,-1] + 2 * u[:,-2] - u[:,-3]) / (Problem.h_x ** 2)
        _ddux[:,1:-1] = (-u[:,:-2] + 2 * u[:,1:-1] - u[:,2:]) \
            / (Problem.h_x ** 2)
        ddu.append(_ddux)

        # y direction
        _dduy: np.ndarray = np.zeros((Problem.n_y, Problem.n_x))
        _dduy[0,:] = (-u[0,:] + 2 * u[1,:] - u[2,:]) / (Problem.h_y ** 2)
        _dduy[-1,:] = (-u[-1,:] + 2 * u[-2,:] - u[-3,:]) / (Problem.h_y ** 2)
        _dduy[1:-1,:] = (-u[:-2,:] + 2 * u[1:-1,:] + u[2:,:]) \
            / (Problem.h_y ** 2)
        ddu.append(_dduy)

    return ddu

def brute_all(Solver: Solver, Problem: HjbProblem, HjbOperator: HjbOperator) \
    -> np.ndarray:
    """Calculates the optimal control for the current solution with a line search
       performed over the discretized control values. The function performs the 
       line search by calculating all the values and finding the minimum or 
       maximum over all this values. This means that this function needs much 
       space in the RAM if the discretization in the spatial and/or the discretization
       of the control is fine. For example in two dimensions we need
       n_x * n_y * n_q * sizeof(double) amoaunt of RAM.

    Returns
    -------
    ndarray
        Current optimal control values
    """
    # allocate space to save all values
    if Problem.dimension == 1:
        val: np.ndarray = np.empty((Problem.n_x, Problem.n_q))
    else:
        val: np.ndarray = np.empty((Problem.n_y * Problem.n_x, Problem.n_q))

    for i in range(Problem.n_q):
        # get current control
        Solver.q: float = Problem.q_d[i]

        # get volume force for the current control
        fv: np.ndarray = Problem.f(Solver.t, Problem.x, Solver.q, Problem)
        fv_old: np.ndarray = Problem.f(Solver.t + Problem.h_t, Problem.x,
        Solver.q, Problem)

        # get disretized operator
        HjbOperator.construct_matrix(Solver, Problem)

        # if we are explicit we don't need to calculate the implicit part and 
        # the function value is evaluated at the current time
        if Problem.theta == 0:
            val[:,i] = HjbOperator.L @ Solver.u_old_t.ravel() - fv_old.ravel()
        else:
            # implicit and explicit part of operator 
            L_imp: spsp.csr_array = HjbOperator.L * Problem.theta
            L_exp: spsp.csr_array = HjbOperator.L * (1 - Problem.theta)

            # get value for current controll
            val[:,i] = L_imp @ Solver.u_sol.ravel() \
                + L_exp @ Solver.u_old_t.ravel() \
                - Problem.theta * fv.ravel() \
                - (1 - Problem.theta) * fv_old.ravel()

    # get indice of optimal controll
    if Problem.max_min == "min":
        q_opt_ind: np.ndarray = np.argmin(val, 1)
    elif Problem.max_min == "max":
        q_opt_ind: np.ndarray = np.argmax(val, 1)

    if Problem.dimension == 2:
        q_opt_ind = q_opt_ind.reshape((Problem.n_y, Problem.n_x))

    q_opt: np.ndarray = Problem.q_d[q_opt_ind]

    # save function evaluation for the current best control
    Solver.f = Problem.f(Solver.t, Problem.x, q_opt, Problem)
    Solver.f_old = Problem.f(Solver.t + Problem.h_t, Problem.x, q_opt, Problem)

    return q_opt

def optim_user_func(solver: Solver, Problem: HjbProblem) -> np.ndarray:
    """Calculates the optimal control based on a user function to calculate the
       optimal control

    Parameters
    ----------
    solver : Solver
        Solver class
    Problem : HjbProblem
        HjbProblem class

    Returns
    -------
    ndarray
        Returns the current optimal control values
    """
    if Problem.theta == 0:
        du: np.ndarray = get_derivatives(solver.u_old_t, Problem)
    else:
        du: np.ndarray = get_derivatives(solver.u_sol, Problem)

    return Problem.q_opt_user(solver.t, Problem.x, du, Problem)

def brute(solver: Solver, problem: HjbProblem) -> np.ndarray:
    """Calculates the optimal control for the current solution with a line search
       performed over the discretized control values. The function performs the 
       line search by calculating the values at each discretization point of the 
       control values. This means is uses less space than
       :func: 'optimization.brute_all' but can be considerably slower

    Parameters
    ----------
    solver : Solver
        Solver class
    problem : HjbProblem
        HjbProblem class

    Returns
    -------
    ndarray
        Returns the current optimal control
    """
    # get derivatives of the implicit and explicit part
    if problem.theta == 0:
        du_imp: np.ndarray = get_derivatives(solver.u_old_t, problem)
    else:
        du_imp: np.ndarray = get_derivatives(solver.u_sol, problem)
    du_exp: np.ndarray = get_derivatives(solver.u_old_t, problem)
    if problem.order == 1:
        du_imp.append(np.zeros(problem.n_x))
        du_exp.append(np.zeros(problem.n_x))

    # preallocate
    if problem.dimension == 1:
        q_opt: np.ndarray = np.empty(problem.n_x)
        q_tmp: np.ndarray = np.empty(problem.n_x)
        q_ind: np.ndarray = np.zeros(problem.n_x)
    else:
        q_opt: np.ndarray = np.empty((problem.n_y, problem.n_x))
        q_tmp: np.ndarray = np.empty((problem.n_y, problem.n_x))
        q_ind: np.ndarray = np.zeros((problem.n_y, problem.n_x))

        # loop over the discretized control values
        for i in range(problem.n_q):
            q: float = problem.q_d[i]
            # get the values in front of the derivatives terms for readability
            # first order terms
            dxv: List[np.ndarray] = (problem.dx(solver.t, problem.x, q, 
            problem))

            # second order terms
            if problem.order == 2:
                ddxv: List[np.ndarray] = (problem.ddx(solver.t, problem.x, 
                q, problem))
            else:
                ddxv: List[np.ndarray] = []
                for j in range(problem.dimension): 
                    ddxv.append(np.zeros((problem.n_y, problem.n_x)))

            # volume force
            fv_imp: np.ndarray = (problem.f(solver.t, problem.x, q, problem)) \
                * problem.theta
            fv_exp: np.ndarray = (problem.f(solver.t + problem.h_t, problem.x, 
            q, problem)) * (1 - problem.theta)
            
            if problem.dimension == 1:
                val: np.ndarray = np.zeros(problem.n_x)
            else:
                val: np.ndarray = np.zeros((problem.n_y, problem.n_x))

            # loop over the dimensions and get the sup/inf values
            for j in range(problem.dimension):
                val += (np.maximum(dxv[j], 0) * du_imp[0][j]) * problem.theta \
                    + (np.maximum(dxv[j], 0) * du_exp[0][j]) * (1 - problem.theta) \
                    + (np.maximum(-dxv[j], 0) * du_imp[1][j]) * problem.theta \
                    + (np.maximum(-dxv[j], 0) * du_exp[1][j]) * (1 - problem.theta) \
                    + (ddxv[j] * du_imp[2][j]) * problem.theta \
                    + (ddxv[j] * du_exp[2][j]) * (1 - problem.theta)

            val = val - fv_imp - fv_exp

            # we need no check for the first control value
            if i == 0:
                q_opt = val
                continue
            
            q_tmp = val

            if problem.max_min == "min":
                ind = q_tmp < q_opt
                q_ind += ind
                q_opt[ind] = q_tmp[ind]
            else:
                ind = q_tmp > q_opt
                q_ind += ind
                q_opt[ind] = q_tmp[ind]

        return problem.q_d[q_ind.astype(int)]

def scalar(solver: Solver, problem: HjbProblem) -> np.ndarray:
    """Calculates the optimal control with the scipy.optimize.minimize_scalar
    function. This can be quite slow because we have to perform an optimization
    at each discretization point for each time step. If HjbProblem.q_min or 
    HjbProblem.q_max are specified they are given as bounds to the 
    scipy.optimize.minimize_scalar solver

    Parameters
    ----------
    solver : Solver
        Solver class
    problem : HjbProblem
        HjbProblem class

    Returns
    -------
    ndarray
        Current optimal control values
    """
    # define cost function for readability since its relative long
    def cost_function(q: float) -> float:
        if problem.dimension == 1:
            # current spatial point
            x_d = [x[i]]

            # forward and backwards differences of the implict part for the current point
            du_if = [du_imp[0][0][i]]
            du_ib = [du_imp[1][0][i]]
            
            # second order differences for the current point if necessary
            if problem.order == 2:
                ddu_i = [du_imp[2][0][i]]
            else:
                ddu_i = [0, 0] # set to zeros if dont needed
            
            # forward and backwards differences of the explicit part for the current point
            du_ef = [du_exp[0][0][i]]
            du_eb = [du_exp[1][0][i]]

            # second oder differences for the current point if needed
            if problem.order == 2:
                ddu_e = [du_exp[2][0][i]]
            else:
                ddu_e = [0, 0]
        else:
            # same procedure for the two dimensional case
            x_d = [x[i,j], y[i,j]]
            du_if = [du_imp[0][0][i,j], du_imp[0][1][i,j]]
            du_ib = [du_imp[1][0][i,j], du_imp[1][1][i,j]]

            if problem.order == 2:
                ddu_i = [du_imp[2][0][i,j], du_imp[2][1][i,j]]
            else:
                ddu_i = [0, 0]

            du_ef = [du_exp[0][0][i,j], du_exp[0][1][i,j]]
            du_eb = [du_exp[1][0][i,j], du_exp[1][1][i,j]]

            if problem.order == 2:
                ddu_e = [du_exp[2][0][i,j], du_exp[2][1][i,j]]
            else:
                ddu_e = [0,0]

        # implicit and explicit part of the volume force
        fv_imp: float = problem.f(solver.t, x_d, q, problem) * problem.theta
        fv_exp: float = problem.f(solver.t, x_d, q, problem) \
            * (1 - problem.theta)
        
        # get the terms in front of the first oder derivative
        dxv: List[float] = problem.dx(solver.t, x_d, q, problem)

        # get the terms in front of the secon order dervative if necessary
        if problem.order == 2:
            ddxv: List[float] = problem.ddx(solver.t, x_d, q, problem)
        else:
            ddxv: List[float] = [0 for _ in range(problem.dimension)]

        # initialize val to 0 so we can sum up all the necessary components
        val: float = 0
        
        # sum up all the components
        for k in range(problem.dimension):
            val += (np.maximum(dxv[k], 0) * du_if[k]) * problem.theta \
                + (np.maximum(dxv[k], 0) * du_ef[k]) * (1 - problem.theta) \
                + (np.minimum(dxv[k], 0) * du_ib[k]) * problem.theta \
                + (np.minimum(dxv[k], 0) * du_eb[k]) * (1 - problem.theta) \
                + (ddxv[k] * ddu_i[k]) * problem.theta \
                + (ddxv[k] * ddu_e[k]) * (1 - problem.theta) \

        # cost function splitted in implicit and explicit part
        val = val - fv_imp * problem.theta - fv_exp * (1 - problem.theta)
        
        # check if we want to maximize
        if problem.max_min == "max":
            return -val
        else:
            return val

    # get derivatives and gather the spatial discretization for readability
    if problem.theta == 0:
        du_imp: np.ndarray = get_derivatives(solver.u_old_t, problem)
    else:
        du_imp: np.ndarray = get_derivatives(solver.u_sol, problem)
    du_exp: np.ndarray = get_derivatives(solver.u_old_t, problem)
    x = problem.x[0]
    
    if problem.dimension == 1:
        q_opt: np.ndarray = np.zeros(problem.n_x)

        # loop over all discretization points
        for i in range(problem.n_x):
            # check if we have bounds on the control
            if not problem.q_min or not problem.q_max:
                res = spop.minimize_scalar(cost_function,
                bounds=(problem.q_min, problem.q_max))
            else:
                res = spop.minimize_scalar(cost_function)
            q_opt[i,j] = res.x
    else:
        y = problem.x[1]
        q_opt: np.ndarray = np.zeros((problem.n_y, problem.n_x))

        # loop over all discretization points in the two dimensional case
        for i in range(problem.n_y):
            for j in range(problem.n_x):
                # check if we have bounds on the control
                if not problem.q_min or problem.q_max:
                    res = spop.minimize_scalar(cost_function,
                    bounds=(problem.q_min, problem.q_max))
                else:
                    res = spop.minimize_scalar(cost_function)
                q_opt[i,j] = res.x

    return q_opt