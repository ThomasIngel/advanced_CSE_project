from typing import List
import numpy as np
import scipy.sparse as spsp
from HjbProblem import HjbProblem

def construct_matrix(diags: List[np.ndarray], Problem: HjbProblem) \
    -> spsp.csr_array:

    # shape diagonal dependent of the hjb
    diags_shaped: List[np.ndarray] = shape_diagonals(diags, Problem)

    # construct discretized operator matrix
    if Problem.dimension == 1:
        L: spsp.csr_array = spsp.diags(diags_shaped, [-1,0,1], format="csr")
    else:
        L: spsp.csr_array = spsp.diags(diags_shaped, [-Problem.n_x, -1, 0, 1, 
            Problem.n_x], format="csr")

    return L

def incoporate_dirichtlet_bcs(t: float, q: np.ndarray, rhs: np.ndarray,
    diags: List[np.ndarray], Problem: HjbProblem) -> np.ndarray:
    # get boundaries
    u_D: np.ndarray = Problem.u_D(t, Problem.x, Problem)

    # minus for the diags because they are negative
    if Problem.dimension == 1:
        u_D_l: float = u_D[0]
        u_D_r: float = u_D[1]
        rhs[1] = rhs[1] - Problem.h_t * Problem.theta * diags[0][1] * u_D_l
        rhs[-2] = rhs[-2] - Problem.h_t * Problem.theta * diags[2][-2] * u_D_r
        
        return rhs[1:-1]

    else:
        u_D_t: np.ndarray = u_D[0]
        u_D_r: np.ndarray = u_D[1]
        u_D_b: np.ndarray = u_D[2]
        u_D_l: np.ndarray = u_D[3]

        # top boundarys
        rhs[1,:] = rhs[1,:] - Problem.h_t * Problem.theta * diags[0][1,:] \
            * u_D_t

        # right boundaries
        rhs[:,-2] = rhs[:,-2] - Problem.h_t * Problem.theta * diags[3][:,-2] \
            * u_D_r

        # bottom boundaries
        rhs[-2,:] = rhs[-2,:] - Problem.h_t * Problem.theta * diags[4][-2,:] \
            * u_D_b

        # left boundaries
        rhs[:,1] = rhs[:,1] - Problem.h_t * Problem.theta * diags[1][:,1] \
            * u_D_l

        return rhs[1:-1,1:-1]

def gather_bcs(u_sol:np.ndarray, t: float, Problem: HjbProblem) \
    -> np.ndarray:
    # gather bcs
    u_D: List[np.ndarray] = Problem.u_D(t, Problem.x, Problem)

    if Problem.dimension == 1:
        u_sol[0] = u_D[0]
        u_sol[-1] = u_D[1]
    else:
        # write bcs in solution; order top, right, bottom, left
        u_sol[0,:] = u_D[0]
        u_sol[:,-1] = u_D[1]
        u_sol[-1,:] = u_D[2]
        u_sol[:,0] = u_D[3]

    return u_sol

def get_unraveld_diagonals(dxv: np.ndarray, ddxv: np.ndarray, 
    Problem: HjbProblem) -> List[np.ndarray]:
    # get step size factors
    sts_fac: List[float] = get_stepsize_factors(Problem)

    if Problem.dimension == 1:
        # sub and sup diagonal
        diag_m1: np.ndarray = sts_fac[0] * np.minimum(dxv, 0)
        diag_p1: np.ndarray = -sts_fac[0] * np.maximum(dxv, 0)
        diag_0: np.ndarray = -diag_m1 - diag_p1
        if Problem.order == 2:
            diag_m1 = diag_m1 - sts_fac[1] * ddxv
            diag_p1 = diag_p1 - sts_fac[1] * ddxv
            diag_0 = diag_0 + 2 * sts_fac[1] * ddxv 

        return [diag_m1, diag_0, diag_p1]
    else:
        # sub diagonals
        diag_m1: np.ndarray = sts_fac[0] * np.minimum(dxv[0], 0)
        diag_mm: np.ndarray = sts_fac[1] * np.minimum(dxv[1], 0)

        # sup diagonals
        diag_p1: np.ndarray = -sts_fac[0] * np.maximum(dxv[0], 0)
        diag_pm: np.ndarray = -sts_fac[1] * np.maximum(dxv[1], 0)

        # main diagonal
        diag_0: np.ndarray = -diag_m1 - diag_mm - diag_p1 - diag_pm 
        if Problem.order == 2:
            diag_m1 = diag_m1 - sts_fac[2] * ddxv[0]
            diag_p1 = diag_p1 - sts_fac[2] * ddxv[0]
            diag_mm = diag_mm - sts_fac[3] * ddxv[1]
            diag_pm = diag_pm - sts_fac[3] * ddxv[1]
            diag_0 = diag_0 + 2 * sts_fac[2] * ddxv[0] + 2 * sts_fac[3] \
                * ddxv[1]

        return [diag_mm, diag_m1, diag_0, diag_p1, diag_pm]

def get_stepsize_factors(Problem: HjbProblem) -> \
    List[float]:
    # initialize list
    sts_fac: List[np.ndarray] = []

    # check if we are time dependent
    if Problem.dimension == 1:
        sts_fac.append(1 / Problem.h_x)
        if Problem.order == 2:
            sts_fac.append(1 / Problem.h_x**2)
            
    else: 
        sts_fac.append(1 / Problem.h_x)
        sts_fac.append(1 / Problem.h_y)
        if Problem.order == 2:
            sts_fac.append(1 / (Problem.h_x ** 2))
            sts_fac.append(1 / (Problem.h_y ** 2))
    
    return sts_fac

def shape_diagonals(diags: List[np.ndarray], Problem: HjbProblem) \
    -> List[np.ndarray]:
    diags_shaped: List[np.ndarray] = []
    # check if time dependent then we construct the matrix without the bc nodes
    if Problem.dimension == 1:
        diags_shaped.append(diags[0][1:])
        diags_shaped.append(diags[1])
        diags_shaped.append(diags[2][:-1])
    else:
        for diag in diags:
            diags_shaped.append(diag.ravel())
        diags_shaped[0] = diags_shaped[0][Problem.n_x:] 
        diags_shaped[1][::Problem.n_x] = 0
        diags_shaped[1] = diags_shaped[1][1:]
        diags_shaped[3][::Problem.n_x] = 0
        diags_shaped[3] = diags_shaped[3][:-1]
        diags_shaped[4] = diags_shaped[4][:-Problem.n_x]

    return diags_shaped
