r"""
============================
HjbProblem :mod:`HjbProblem`
============================

Class for all the parameters of the problem to be solved

.. sectionauthor::
	Thomas Ingel, Feb 2023
"""
import numpy as np
from typing import Callable, List, Union, Tuple
import warnings

class HjbProblem:
	""" 
	Class for all the parameters of the specified problem.

	This class is responsible for the discretziation and the book keeping 
	of the parameters of the problem.

	Attributes
	----------

		x_min : float
			Lower bound for the first spatial dimension. Default: -1.0

		x_max : float
			Upper bound for the first spatial dimension. Default: 1.0

		q_min : float
			Lower bound for the control values. If the control values is not
			bounded in the -infinity direction you have to specify it as an 
			empty list. Default: -1.0

		q_max : float
			Upper bound for the control values. If the control values is not
			bounded in the +infinity direction you have to specify it as an 
			empty list. Default: 1.0

		T : float
			Endtime of the time dependent problem. Default: 1.0

		f : Callable
			User defined function for the volume force.

		u_D : Callable
			User defined dirichlet boundary conditions. If the problem is one 
			dimensional you have to specify a list with two entrys where the 
			value on left bound is in the first entry of the list and the value
			on the right bound is in the second entry. 

			If the Problem is two dimensional you have to specify it as a list
			with **four** :obj:`numpy.ndarray`\ s in the following order:

				1. Top boundary conditions

				2. Right boundary conditions

				3. Bottom boundary conditions

				4. Left boundary conditions

		u_0 : Callable
			User defined initial conditions. Have to return a :obj:`numpy.ndarray` of the 
			appropriate size.

		order : int {1, 2}
			Order of the highest derivative term

		dimension : int {1, 2}
			Spatial dimension of the problem

		max_min : str {``"min"``, ``"max"``}
			Either ``max`` or ``min`` depending wheter you have a supremum or a infimum
			in your HJB equation. This option is only necessary if you don't 
			specify a user defined optimization routine.

		theta : float or int [0, 1]
			Theta parameter in the time discretization

		tol : float
			Termination criterium for the howard algorithm

		max_iter : float
			Maximum amount of iterations allowed in the howard algorithm

		tol_gmres : float
			Tolerance for the GMRES solver for solving the linear system of 
			equations

		info : int {0, 1}
			Information level of the solver. Supported are level 0 wich only 
			writes wheter the howard algorithm terminated or not and the norm 
			of the sucessfull or unsecussfull step. Level 1 writtes the norm of
			all the iterations in the Howard algorithm
			
		n_x : int
			Number of points in the equidistant spatial discretization in the 
			first dimension

		n_t : int
			Number of points in the equdistant time discretization

		optim_method : str {``"user_func"``, ``"brute_all"``, ``"brute"``, ``"scalar"``}
			Optimization method wich shall be used

		n_q : int
			Number of points in the equidistant controll discretization

		q_opt_user : Callabel
			User suplied optimization routine

		dx : Callable
			User suplied calculation of the terms infront of the first order 
			derivatives. Has to return a list with entrys of partial derivatives
			in the corresponding direction.

		ddx: Callable
			User suplied calculation of the terms infront of the second order 
			derivatives. Has to return a list with entrys of partial derivatives
			in the corresponding direciton.

		y_min : float
			Lower bound for the second spatial dimension. Default: -1.0

		y_max : float
			Upper bound for the second spatial dimension. Default: 1.0

		n_y : int
			Number of points in the equidistant spatial discretization in the 
			second dimension

		sol : List[ndarray]
			List of the calculated solutions. Each element in the list is a 
			:obj:`numpy.ndarray` with the solution at this time step.

		q_opt : List[ndarray]
			List of the calculated optimal control values. Each element in the 
			list is a :obj:`numpy.ndarray` with the solution at this time step.	

		x : List[ndarray]
			Spatial discretization of the domain. Each entry in the list is 
			a spatial direction 

		dofs : int
			Number of degrees of freedom for the discretized domain without the
			boundary points.

		t_d : ndarray
			Discretized time dimension.

		h_t : float
			Step size in time.

		h_x : float
			Step size in the first spatial dimension.

		h_y : float
			Step size in the second spatial dimension.

	"""

	def __init__(self, dimension: int, order: int, max_min: str, 
		optim_method: str, info: int, theta: Union[float, int]) -> None:
		# problem parameters
		self.x_min: float = -1.0
		self.x_max: float = 1.0
		self.q_min: float = -1.0
		self.q_max: float = 1.0
		self.T: float = 1
		self.f: Callable = []
		self.u_D: Callable = []
		self.u_0: Callable = []
		self.order = order
		self.dimension = dimension
		self.max_min = max_min
		self.theta = theta
		self.tol = 1e-6
		self.max_iter = 100
		self.tol_gmres = 1e-9
		self.info = info
		self.q_d = []
		
		# discretization parameters
		self.n_x: int = 11
		self.n_t: int = 11
		
		# options for optimization procedure
		self.optim_method = optim_method

		# if we do a line search we need the number of points and discretize the
		# control
		if self.optim_method == "brute" or self.optim_method == "brute_all" :
			self.n_q = 11

		if self.optim_method == "user_func":
			self.q_opt_user: Callable = []
		
		# terms in front of first and second order derivativ
		self.dx: Callable = []
		self.ddx: Callable = []
		
		# intervall for second dimension
		if dimension == 2:
			self.y_min: float = -1
			self.y_max: float = 1
			self.n_y: int = 11

		# fields for solution and plotting of the solution
		self.sol: List[np.ndarray] = []
		self.q_opt: List[np.ndarray] = []
		self.x: List[np.ndarray] = []
		self.dofs: int = []

	def discretize(self) -> None:
		"""Function wich discretizes the spatial, time dimensions and if necessary
		the control (discretizes the control only if it isn't specified)
		"""

		# time discretization in reverse order because we solve the HJB 
		# backwards in time
		self.t_d: np.ndarray = np.linspace(self.T, 0, self.n_t, True)
		self.h_t: float = self.T / (self.n_t - 1)

		# spatial discretization
		if self.dimension == 1:
			_x_d: np.ndarray = np.linspace(self.x_min, self.x_max, self.n_x,
				True)
			self.x.append(_x_d)
			self.h_x: float = (self.x_max - self.x_min) / (self.n_x - 1)

		else:
			_x_d: np.ndarray = np.linspace(self.x_min, self.x_max, self.n_x, 
				True)
			self.h_x: float = (self.x_max - self.x_min) / (self.n_x - 1)
			
			# reverse order so the y-axis goes upwards
			_y_d: np.ndarray = np.linspace(self.y_min, self.y_max, self.n_y, 
				True)
			self.h_y: float = (self.y_max - self.y_min) / (self.n_y - 1)
			_x_d, _y_d = np.meshgrid(_x_d, _y_d)
			
			self.x.append(_x_d)
			self.x.append(_y_d)

		if self.optim_method == "brute" or self.optim_method == "brute_all":
			if not isinstance(self.q_d, np.ndarray):
				if not self.q_d:
					self.q_d: np.ndarray = np.linspace(self.q_min, self.q_max, 
						self.n_q, True)
				else:
					self.q_d = np.array(self.q_d)

	def check_input_and_discretize(self) -> None:
		"""
		Function to discretize and checking of obviuos input errors
		"""
		# TODO: write input checks
		self.discretize()
		self.get_non_bcs_ind()
		self.sol.append(self.u_0(self.T, self.x, self))
		if self.dimension == 1:
			self.dofs = self.n_x - 2
		else:
			self.dofs = (self.n_x - 2) * (self.n_y - 2)

	def get_non_bcs_ind(self) -> None:
		"""
		Function to calculate the indices wich are not on the boundary
		"""
		# easier to implement if we make the matrix and then cut off the bound
		# ary points	
		if self.dimension == 1:
			_inds: np.ndarray = np.arange(1, self.n_x - 1)
			self.inds: Tuple[np.ndarray] = np.ix_(_inds, _inds)
		else:
			_inds: np.ndarray = \
				np.arange(0, self.n_x * self.n_y).reshape((self.n_y, 
				self.n_x))
			_inds = _inds[1:-1, 1:-1].ravel()
			self.inds: Tuple[np.ndarray] = np.ix_(_inds, _inds)

	def check_inputs(self) -> None:
		"""
		Function to check for obvious input errors.
		"""
		def check_atribute_type_float_or_int(attributes: List[str]) -> None:
			for attribute in attributes:
				if not (isinstance(getattr(self, attribute), float) or 
				isinstance(getattr(self, attribute), int)):
					raise TypeError(attribute + " must be of type float or int")

		def check_attribute_type_int(attributes: List[str]) -> None:
			for attribute in attributes:
				if not isinstance(getattr(self, attribute), int):
					raise TypeError(attribute + " must be of type int")

		def check_user_funcs_exists(func_names: List[str]) -> None:
			for func in func_names:
				if not callable(getattr(self, func)):
					raise TypeError(func + " musst be a callable function")

		# check type and number of dimensions
		if type(self.dimension) != int:
			raise TypeError("Dimensions must be of type int")

		if self.dimension != 1 and self.dimension != 2:
			raise ValueError("Only one and two dimensions are supported not", 
				self.dimension)

		# check order and type of order
		if type(self.order) != int:
			raise TypeError("Order must be of type int")

		if self.order != 1 and self.order != 2:
			raise ValueError("Only order one and two are supported not", 
				self.order)

		# check input types
		if self.dimension == 1:
			attr_to_type_check = ["T", "x_max", "x_min", "q_max", "q_min", 
			"theta"]

		if self.dimension == 2:
			attr_to_type_check = ["T", "x_max", "x_min", "y_min", "y_max", 
				"q_max", "q_min", "theta"]

		check_atribute_type_float_or_int(attr_to_type_check)

		# check input values
		if self.T < 0:
			raise ValueError("T muss be greater than 0")

		if self.x_max < self.x_min:
			raise ValueError("x_max must be greater than x_min", "x_max",    
			self.x_max, "x_min", self.x_min)  

		if self.q_max < self.q_min:
			raise ValueError("q_max must be greater than q_min", "q_max",    
			self.q_max, "q_min", self.q_min)  

		if self.dimension == 1:
			if hasattr(self, "y_min") or hasattr(self, 'y_max'):
				warnings.warn("y_min or y_max specified for dimension 1")
		else:
			if self.y_max < self.y_min:
				raise ValueError("y_max must be greater than y_min", "y_max", 
				self.y_max, "y_min", self.y_min)

		# check if discretization points are integers
		ints_to_check: List[str] = ["n_t", "n_x", "n_q"]

		if self.dimension == 1:
			if hasattr(self, "n_y"):
				warnings.warn("n_y specified for dimension 1")
		else:
			ints_to_check.append("n_y")

		check_attribute_type_int(ints_to_check)

		# check if the user suplied the necessary functions
		funcs: List[str] = ["f", "u_D", "u_0", "dx"]

		if self.order == 1:
			if hasattr(self, "ddx"):
				warnings.warn("ddx specified for order 1")
		else:
			funcs.append("ddx")

		check_user_funcs_exists(funcs)
		# TODO: Wright check for optimization methods