from .hjb_problem import HjbProblem
import scipy.sparse as spsp
import numpy as np

class Solver:
    """
        Solver class where all the necessary data for the howard algorithm is stored.

        Attributes
        ----------
        t : float
            current time step 
        f_old : ndarray
            volume force for the explicit part
        q_old_t : ndarray
            controll values from the time step bevor
        q_old : ndarray
            old control values in one howard step
        q_new : ndarray
            new control values after one howard step
        f : ndarray
            current volume forces
        u_new : ndarray
            new solution of the value function after one howard step
        u_old : ndarray
            old solution of the value function in one howard step
        u_old_t : ndarray
            solution from the time step bevor for the explicit part
        u_sol : ndarray
            current solution after one howard step with boundary conditions
        rhs : ndarray
            right hand side of the linear system of equation
        rhs_exp : ndarray
            explicit part of the right hand side
        L_exp : scipy.sparse.csr_array
            explicit part of the spatial derivative operator
        L_imp : scipy.sparse.csr_array
            implicit part of the spatial derivative operator
        q : ndarray
            controll values wich are used in the optimization algorithms and for
            the evaluation of the user defined functions
        I : ndarray
            identity matrix of the appropriate size for the time discretization
    """

    def __init__(self, Problem: HjbProblem) -> None:
        # declare some variables wich are needed in the howard algorithm
        self.t = Problem.T
        self.f_old: np.ndarray = None
        self.q_old_t: np.ndarray = None
        self.q_old: np.ndarray = None
        self.q_new: np.ndarray = None
        self.f: np.ndarray = None
        self.u_new: np.ndarray = None
        self.u_old: np.ndarray = None
        self.u_old_t: np.ndarray = None
        self.u_sol: np.ndarray = None
        self.rhs: np.ndarray = None
        self.rhs_exp: np.ndarray = None
        self.L_exp: spsp.csr_array = None
        self.L_imp: spsp.csr_array = None
        self.q: np.ndarray = None

        # dimension of the identity matrix for the time stepping
        if Problem.dimension == 1:
            self.I: spsp.csr_array = spsp.csr_array(spsp.identity(Problem.n_x))
        else:
            self.I: spsp.csr_array = spsp.csr_array(spsp.identity(Problem.n_x * 
                Problem.n_y))