r"""
=====================================
:mod:`hjb.hjb_operator`
=====================================
Class for the Kushner-Dupuis discretization scheme

Functions
--------------------

.. autosummary::
   :toctree: generated/

   HjbOperator
   get_diagonals
   get_unraveld_diagonals
   shape_diagonals
   construct_matrix

.. sectionauthor:: Thomas Ingel, Feb 2023
"""

from .hjb_problem import HjbProblem
from .solver_helper import Solver
from typing import List
import numpy as np
import scipy.sparse as spsp

class HjbOperator:
    """
    Class for the construction of the discretization scheme.

    Attributes
    ----------
    dimension : int
        dimension of the problem (1 or 2 dimensions supported) received from
        the :obj:`HjbProblem` class
    order : int
        order of the highest derivative term in the hjb received from the 
        :obj:`HjbProblem` class
    n_x : int
        spatial discretization in the first dimension received from the 
        :obj:`HjbProblem` class
    sts_fac : List[float]
        step size factors for the spatial dimensions 
    """
    def __init__(self, Problem: HjbProblem) -> None:
        # collect parameters from the problem class
        self.dimension: int = Problem.dimension
        self.order: int = Problem.order
        self.n_x: int = Problem.n_x
        self.sts_fac: List[float] = []

        # distinguish between one and two dimensions and order one and two
        if self.dimension == 1:
            self.sts_fac.append(1 / Problem.h_x)
            if Problem.order == 2:
                self.sts_fac.append(1 / Problem.h_x**2)
        else:
            self.sts_fac.append(1 / Problem.h_x)
            self.sts_fac.append(1 / Problem.h_y)
            if self.order == 2:
                self.sts_fac.append(1 / Problem.h_x**2)
                self.sts_fac.append(1 / Problem.h_y**2)

    def get_diagonals(self, Solver: Solver, Problem: HjbProblem) -> None:
        """
        Function to gather the terms in front of the derivatives

        Parameters
        ----------
        Solver : Solver
            Solver class where all the parameters for the howard algorithm are saved
        Problem : HjbProblem
            HjbProblem class where all the parameters from the problem are saved
        """
        # get values infront of the derivatives
        self.dxv: List[np.ndarray] = Problem.dx(Solver.t, Problem.x,
        Solver.q, Problem)

        if self.order == 2:
            self.ddxv: List[np.ndarray] = Problem.ddx(Solver.t, Problem.x, 
            Solver.q, Problem)

        self.get_unraveld_diagonals()

    def get_unraveld_diagonals(self) -> None:
        """
        Constructs the diagonals for the different matrices in the different spatial dimensions
        """
        # dimension check
        if self.dimension == 1:
            # sub and sup diagonals
            diag_m1: np.ndarray = self.sts_fac[0] * np.minimum(self.dxv[0], 0)
            diag_p1: np.ndarray = -self.sts_fac[0] * np.maximum(self.dxv[0], 0)

            # main diagonal
            diag_0: np.ndarray = -diag_m1 - diag_p1

            # second order terms
            if self.order == 2:
                diag_m1 = diag_m1 - self.sts_fac[1] * self.ddxv[0]
                diag_p1 = diag_p1 - self.sts_fac[1] * self.ddxv[0]
                diag_0 = diag_0 + 2 * self.sts_fac[1] * self.ddxv[0]

            self.diags = [diag_m1, diag_0, diag_p1]
        else:
            # sub diagonals
            diag_m1: np.ndarray = self.sts_fac[0] * np.minimum(self.dxv[0], 0)
            diag_mm: np.ndarray = self.sts_fac[1] * np.minimum(self.dxv[1], 0)

            # sub diagonals
            diag_p1: np.ndarray = -self.sts_fac[0] * np.maximum(self.dxv[0], 0)
            diag_pm: np.ndarray = -self.sts_fac[1] * np.maximum(self.dxv[1], 0)

            # main diagonal
            diag_0: np.ndarray = -diag_m1 - diag_mm - diag_p1 - diag_pm
            if self.order == 2:
                diag_m1 = diag_m1 - self.sts_fac[2] * self.ddxv[0]
                diag_p1 = diag_p1 - self.sts_fac[2] * self.ddxv[0]
                diag_mm = diag_mm - self.sts_fac[3] * self.ddxv[1]
                diag_pm = diag_pm - self.sts_fac[3] * self.ddxv[1]
                diag_0 = diag_0 + 2 * self.sts_fac[2] * self.ddxv[0] \
                    + 2 * self.sts_fac[3] * self.ddxv[1]

            self.diags = [diag_mm, diag_m1, diag_0, diag_p1, diag_pm]

    def shape_diagonals(self) -> None:
        """Brings the diagonals in the right shape. 
        
        Considers e. g. the zeros in the diagonals where the boundary conditions
        are located
        """
        self.diags_shaped: List[np.ndarray] = []

        # cut of the last points for one spatial dimension
        if self.dimension == 1:
            self.diags_shaped.append(self.diags[0][1:])
            self.diags_shaped.append(self.diags[1])
            self.diags_shaped.append(self.diags[2][:-1])
        else:
            for diag in self.diags:
                self.diags_shaped.append(diag.ravel())
            
            # cuts of the right length of the diagonals and places zeros on the
            # nodes where boundary conditions are located
            self.diags_shaped[0] = self.diags_shaped[0][self.n_x:]
            self.diags_shaped[1][::self.n_x] = 0
            self.diags_shaped[1] = self.diags_shaped[1][1:]
            self.diags_shaped[3][::self.n_x] = 0
            self.diags_shaped[3] = self.diags_shaped[3][:-1]
            self.diags_shaped[4] = self.diags_shaped[4][:-self.n_x]

    def construct_matrix(self, Solver: Solver, Problem: HjbProblem) -> None:
        """
        Constructs the conrecte matrices from the spatial differential operator
        in the *csr* format

        Parameters
        ----------
        Solver : Solver
            Solver class where all the parameters for the howard algorithm are saved
        Problem : HjbProblem
            HjbProblem class where all the parameters from the problem are saved
        """
        self.get_diagonals(Solver, Problem)
        self.shape_diagonals()

        if self.dimension == 1:
            self.L: spsp.csr_array = spsp.diags(self.diags_shaped, [-1, 0, 1], 
            format="csr")
        else:
            self.L: spsp.csr_array = spsp.diags(self.diags_shaped, 
            [-self.n_x, -1, 0, 1, self.n_x], format="csr")