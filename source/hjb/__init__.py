r""" 
==========
:mod:`hjb`
==========

Introduction
------------

The aim of this module is to solve a Hamilton-Jacobi-Bellman equation (HJB) of
order one or two in one and two dimensions. For this we have to make the assumption
that the HJB is of the following form

.. math::
    -u_t(t,x) + \min_{q \in \mathcal{Q}} \left\{ L^{q} u(t,x) - f(t,x,q) \right\} = 0
    :label: eq:hjb_min

or 

.. math::
    -u_t(t,x) + \max_{q \in \mathcal{Q}} \left\{ L^{q} u(t,x) - f(t,x,q) \right\} = 0
    :label: eq:hjb_max

where :math:`L^{q}` is the operator for the space discretization.
Wich is defined as 

.. math::
    L^q := -\text{Tr } a(t,x,q) D^2 u(t,x) - b (t,x,q) \cdot D u(t,x).

with :math:`a \in \mathbb{R}^{d \times d}` a diagonal matrix and :math:`b \in 
\mathbb{R}^d` whereby :math:`d` is the number of spatial dimensions. This operator
is discretzised with the *Kushner-Dupuis* scheme.
For the time discretization we use the *theta* method such that 

.. math::
    -\Delta_t^+ u_h (t_k, x_i) + \max_{q \in \mathcal{Q}} 
    \left[ \theta (L_h^q u_h (t_k, x_i)) + (1 - \theta) (L_h^q u_h (t_{k+1}, x_i)) 
    -f(t_k, x_i, q)\right] = 0

where :math:`\theta \in [0,1]`. If :math:`\theta > 0` we have to solve a non linear
system of equations in every time step. For the following :math:`\theta` values
we get the well known different methods for solving the coupled ode system from 
the discretization

    * :math:`\theta = 0` explicit Euler
    * :math:`\theta = 0.5` Crank-Nicolson method
    * :math:`\theta = 1` implicit Euler

We also make the assumption that we solve the HJB backwards in time wich is a 
common thing to do.

Since the requirements for a classical solution, i. g. :math:`u \in C^2` are quite
strong we are using the framework of *viscosity solutions* to solve the HJB in a weak
sense. More on viscosity solutions can be found in [1]_.

References
~~~~~~~~~~

.. [1] Michael G. Crandall und Pierre-Louis Lions. „Viscosity Solutions of Hamilton-
        Jacobi Equations“. In: Transactions of the American Mathematical Society
        277.1 (1983), S. 1–42. ISSN: 00029947. URL: http://www.jstor.org/stable/1999343.

        


General Hints
-------------

This module is written such that the user doesn't have to think about what is 
happening behind the scenes. Instead the user only has to provide the terms in
front of the derivatives (i. g. the matrix :math:`a` and the vector :math:`b`)
and the volume force :math:`f`. Since :math:`a` is a diagonal matrix we don't 
consider cross terms in the derivatives and equation :math:numref:`eq:hjb_min` 
simplifies in the two dimensional second order case to

.. math::
    -u_t(t,x) + \min_{q \in \mathcal{Q}} \{ &-a_{11}(t,x,q) u_{xx}(t,x) 
    - a_{22}(t,x,q) u_{yy}(t,x) \\ 
    &- b_1(t,x,q) u_x(t,x) - b_2(t,x,q) u_y(t,x) 
    - f(t,x,q) \} = 0.

The other cases for the order and dimensions as well as the :math:`\max` case in
equation :math:numref:`eq:hjb_max` are analougus.

For this the module is build such that the user has to specify the following functions
``dx``, ``ddx``, ``f``, ``u_0`` and ``u_D``. Where the functions are

    * ``dx`` terms in front of the first derivative (:math:`b` vector)
    * ``ddx`` terms in fron of the second derivative (:math:`a` matrix)
    * ``f`` volume force
    * ``u_0`` initial conditions
    * ``u_D`` dirichlet boundary conditions

The functions ``dx``, ``ddx`` and ``f`` have the signature 
``(t, x, q, problem)`` and the functions ``u_0`` and ``u_D`` the signature
``(t, x, problem)``. Whereas the signature parameters are defined as

    * ``t`` : ``float`` current time 
    * ``x`` : ``List[numpy.ndarray]`` space discretization points
    * ``q`` : ``float`` control value
    * ``problem`` : :class:`HjbProblem` HjbProblem class.

The functions ``f``, ``u_0`` have to return a :obj:`numpy.ndarray` of the appropriate
size of the problem.

For ``dx``, ``ddx`` the return type has to be a ``List[numpy.ndarray]`` of the 
appropriate sice where the first entry in the list is for the first spatial dimension
and the second entry for the second spatial dimension if needed.

The boundary conditions have to have a different return type for the one dimensional
and two dimensional case. In the one dimensional case the return type has to be
``List[float]`` with two entrys. The first entry is the value on the left boundary
point and the second entry is the value on the right boundary point. 

In the two dimensional case the return type has to be ``List[numpy.ndarray]`` with
four entrys. The entrys are in the following order according to their position on
the rectangular domain. First entry **top**, second entry **right**, third entry **bottom** and
fourth entry **left** boundary condition.

.. note::
    The term *appropriate size* is till now vague. If you aren't using the 
    :obj:`scalar` optimization routine then the size is the size of the discretized
    problem. If the Problem is one dimensional than return size is 
    :math:`\mathbb{R}^{n_x}` and in the two dimensional case :math:`\mathbb{R}^{n_x \times n_y}`.

    Since the :obj:`scalar` optimization routine calculates the optimal control on 
    each discretization point the return size has to be of the problem size for the
    discretization and one for the optimizer. This is a little bit ugly but you can
    make sure that you are giving back the appropriate size if your return size is of size
    ``np.shape(x[0])`` or ``np.shape(x[1])`` since they are the same sice.

Examples
--------

To understand the principle of this module we consider three examples step by step
with different complexity. All of the examples can be found in the example folder.

Example 1
~~~~~~~~~

The first example we are looking at is a one dimensional HJB of order one. The HJB
reads as follow

.. math::
    -u_t + |u_x| &= 1 \qquad \text{on } (-1,1) \times (0,1) \\
    u &= 0 \qquad \text{on } \{-1,1\} \times (0,1) \cup (-1,1) \times \{1\},

wich can be equivalent reformulated as

.. math::
    -u_t + \max_{q \in \{-1,1\}} \{q \cdot u_x - 1\} &= 0 \qquad \text{on } (-1,1) \times (0,1)\\
    u &= 0 \qquad \text{on } \{-1,1\} \times (0,1) \cup (-1,1) \times \{1\},

with unique viscosity solution

.. math::
    u(t,x) = \min(1 - |x|, t).

First we have to initialise the :obj:`HjbProblem` class and set the parameters
for the space and time discretization

.. code-block:: python

    from hjb_problem import HjbProblem
    from hjb_solver import solve_hjb
    import numpy as np

    # initialise problem class
    problem = HjbProblem(1, 1, "max", "brute_all", 1, 1)

    # bound on the 1d domain and number of discretization points
    problem.x_min = -1
    problem.x_max = 1
    problem.n_x = 201

    # end time and number of points in the time discretization 
    problem.T = 1
    problem.n_t = 101

Since our control is bounded and we use the :obj:`brute_all` optimization procedure 
we have to specify the bounds on our control and the number of discretization points
for the control

.. code-block:: python

    # bounds on the control and discretization points
    problem.q_min = -1
    problem.q_max = 1
    problem.n_q = 2

Our volume force ``f`` is constant :math:`1` and our initial conditions ``u_0`` are :math:`0` we
can define them e. g. with a *lambda* function

.. code-block:: python

    # volume force
    problem.f = lambda t, x, q, problem: np.ones(np.shape(x[0]))

    # this would also be possible if you aren't using the scalar optimization routine
    # problem.f = lambda t, x, q, problem: np.ones(problem.n_x)

    # same for the initial conditions
    problem.u_0 = lambda t, x, problem: np.zeros(np.shape(x[0]))
    # problem.u_0 = lambda t, x, problem: np.zeros(problem.n_x)

The term infront of the first derivative is constant :math:`q` so one
possible realisation of ``dx`` reads

.. code-block:: python 

    def dx(t: float, x: List[np.ndarray], q: float, problem: HjbProblem) -> List[np.ndarray]:
        dx_val: List[np.ndarray] = []
        dx_val.append(q * np.ones(np.shape(x[0])))
        
        return dx_val
        
    # after defining the function the problem class needs this function as attribute
    problem.dx = dx

and the dirichlet boundary conditions

.. code-block:: python

    problem.u_D = lambda: t, x, problem: [0, 0]

Finally to solve the problem at hand we have to check the inputs, discretize and
call the solver

.. code-block:: python

    # input check and discretisation
    problem.check_input_and_discretize()

    # solve the HJB with the howard algorithm
    solve_hjb(problem)

The solution of the problem and the optimal control values can be found in the 
``problem.sol``, ``problem.q_opt`` attributes. 

Example 2
~~~~~~~~~

For this example we are looking at the two dimensional version of :ref:`Example 1`.
The equation to solve then reads

.. math::
    -u_t + |\nabla \cdot u| &= 1 \qquad \text{on } (-1,1) \times (0,1) \\
    u &= 0 \qquad \text{on } \{-1,1\} \times (0,1) \cup (-1,1) \times \{1\},

Wich again can be reformulated to 

.. math::
    -u_t + \max_{q \in \{-1,1\}} \{q \nabla u - 1\} &= 0 \qquad \text{on } (-1,1) \times (0,1)\\
    u &= 0 \qquad \text{on } \{-1,1\} \times (0,1) \cup (-1,1) \times \{1\},
    
The whole code this time with the :obj:`brute` optimization routine could look
something like this

.. code-block:: python

    from hjb_problem import HjbProblem
    from hjb_solver import solve_hjb
    import numpy as np

    # user defined functions for dx and u_D
    def dx(t, x, q, problem):
        dxv = []
        dxv.append(q * np.ones(np.shape(x[0])))
        dxv.append(q * np.ones(np.shape(x[1])))

        return dxv

    def u_D(t, x, problem):
        u_D = []

        u_D.append(np.zeros(problem.n_x))
        u_D.append(np.zeros(problem.n_y))
        u_D.append(np.zeros(problem.n_x))
        u_D.append(np.zeros(problem.n_y))

        return u_D

    # initialise solver class
    problem = HjbProblem(2, 1, "max", "brute", 1, 1)
    
    # x discretization
    problem.x_min = -1
    problem.x_max = 1
    problem.n_x = 201

    # y discretization
    problem.y_min = -1
    problem.y_max = 1
    problem.n_y = 201

    # end time and time discretization
    problem.T = 1
    problem.n_t = 101

    # control discretization
    problem.q_min = -1
    problem.q_max = 1
    problem.n_q = 2

    # volume force and initial conditions
    problem.u_0 = lambda t, x, problem: np.zeros(np.shape(x[0]))
    problem.f = lambda t, x, q, problem: np.ones(np.shape(x[0]))

    # check inputs, discretize and solve the problem
    problem.check_inputs_and_discretize()
    solve_hjb(problem)

Example 3
~~~~~~~~~

In this example we look at the HJB from [2]_ wich arises from the modeling of the
trading on the intraday electricity market. This example is of dimension two and
order two and more sophisticated than the academic examples till now. Beside a look
at how a somewhat complicated HJB can be solved with this module we are using the
:obj:`optim_user_func` optimization routine to solve this problem.

After the modeling process we have to solve the following HJB. Find 
:math:`V : [0, T] \times \mathcal{U} \rightarrow \mathbb{R}`,
:math:`V = (t, y, z)` such that

.. math::
    &V_t + \mu_Y V_y + \frac{1}{2} \sigma_y V_{yy} + \frac{1}{2} \sigma_D V_{zz} \\
    &+ \sup_{q(t) \in \mathcal{Q}} \{ -(y + \text{sign}(q(t)) h(t) + \varphi(t, q(t))) q(t)
    + q(t) V_z + \psi (q(t)) V_y \} = 0

for :math:`(t,y,z) \in [0,T) \times \mathcal{U}` with terminal conditions (initial
conditions) :math:`V(T, y, z) = g(T, y, z)` for all :math:`(y,z) \in \mathcal{U}`.

As usual first we have to initialise the solver class and make the discretization

.. code-block:: python

    from hjb_problem import HjbProblem
    from hjb_solver import solve_hjb
    import numpy as np

    # intialize solver and discretization parameters
    problem = HjbProblem(2, 2, "", "user_func", 1, 1)

    problem.x_min = -1600
    problem.x_max = 100
    problem.n_x = 301

    problem.y_min = -100
    problem.y_max = 200
    problem.n_y = 101

    problem.T = 17.5
    problem.n_t = 101

This time we use some non default values for the howard algorithm

.. code-block:: python

    problem.tol = 1e-6
    problem.tol_gmres = 1e-9
    problem.max_iter = 1000

Now we define some problem constants. Since we define them here we can make use
of them in the functions we have to specify

.. code-block:: python

    problem.h = lambda t: 2.11e1 - 7.46 * t + 1.36 * t**2 - 1.01e-1 * t**3 \
        - 1.06e-3 * t**4 + 6.3e-4 * t**5 - 3.59e-5 * t**6 + 6.59e-7 * t**7

    problem.k = lambda t: -0.5 * (-8.72e-2 + 1.67e-2 * t - 5.28e-4 * t**2 
        - 4.15e-4 * t**3 + 5.012e-5 * t**4 - 1.95e-6 * t**5 + 2.36e-8 * t**6)

    problem.b = 0.0017

    problem.phi = lambda k, q: k * q
    problem.psi = lambda b, q: b * q

    problem.mu_y = 0.0433
    problem.sig_y = 1.06
    problem.siy_D = 5

With this parameters we can state the terms in front of the derivatives and the
volume force

.. code-block:: python

    def dx(t, x, q, problem):
        dxv = []

        # V_z term
        dxv.append(q * np.ones(np.shape(x[0])))

        # gather b for readability and calculate V_y term
        b = problem.b
        psi = problem.psi(b, q)
        dxv.append(problem.mu_y + psi * np.ones(np.shape(x[1]))) 

        return dxv

    def ddx(t, x, q, problem):
        ddxv = []

        # V_zz term
        ddz = 0.5 * problem.sig_D**2
        ddxv.append(ddz * np.ones(np.shape(x[0])))

        # V_yy term
        ddy = 0.5 * problem.sig_y**2
        ddxv.append(ddy * np.ones(np.shape(x[1])))

        return ddxv

    def f(t, x, q, problem):
        # gather the second spatial dimension for readability
        y = x[1]

        # evaluate the fitted polynomials and the phi function
        h = problem.h(t)
        k = problem.k(t)
        phi = problem.phi(k, q)

        return (-(y + np.sign(q) * h + phi) * q)

Since the terminal values are a optimization problem on its own we load the
precaulated values for performance

.. code-block:: python
    
    def u_0(t, x, problem):
        return np.load("termval.npy")

As a workaround for the boundary conditions we use the initial conditions on the
bound minus a factor

.. code-block:: python

    def u_D(t, x, problem):
        # gather initial conditions
        G = np.load("termval.npy")

        u_D = []
        u_D.append(G[0,:] - G[0,:] * 0.1)
        u_D.append(G[:,-1] - G[:,-1] * 0.1)
        u_D.append(G[-1,:] - G[-1,:] * 0.1)
        u_D.append(G[:,0] - G[:,0] * 0.1)

        return u_D

Finally the last thing to define is the optimization routine. For this we note
that the signature of the user defined optimization function is not the same as
the others. The signature for :obj:`HjbProblem.q_opt_user` is ``(t, x, du, problem)``
where ``du`` are the different derivatives used in the *Kushner-Dupuis* scheme.
``du`` is constructed in the following way. ``du`` is a ``List[List[np.ndarray]]``
with the entrys in the first dimension:

    * ``du[0][x]`` first order forward differences
    * ``du[1][x]`` first order backwards differences  
    * ``du[2][x]`` second order differences

The second dimension is the dimension of the corresponding spatial dimension.

With this at hand the user defined optimization function could look like this

.. code-block:: python

    def q_optim(t, x, du, problem):
        # gather derivatives
        # forward differences
        dufx = du[0][0]
        dufy = du[0][1]

        # backwards differences
        dubx = du[1][0]
        duby = du[1][1]

        # second order differences (not used here only for completeness)
        # ddux = du[2][0]
        # dduy = du[2][1]

        # get some variables for readability and evaluate the fitted polynomials
        y = x[1]
        h = problem.h(t)
        k = problem.k(t)
        b = problem.b

        # split the sign function into a negative and positive part and calculate
        # the corresponding optimal control values
        q_minus = (-y + h + b * duby + dubx) / (2 * k)
        q_plus = (-y - h + b * duby + dubx) / (2 * k)

        # bounds on the controll with small neighbourhood around 0
        q_minus[q_minus < -50] = -50
        q_minus[q_minus > -0.3] = -0.3
        q_plus[q_plus < 0.3] = 0.3
        q_plus[q_plus > 50] = 50

        # get the values off the calculated optimal control
        # initialise a array to store the values
        f_val = np.zeros((problem.n_y, problem.n_x, 3))
        
        # values for the negative part of the control
        f_val[:,:,0] = -(-(y - h + k * q_minus) * q_minus) + (b * q_minus) * duby \
            + q_minus * dubx

        # if the controll is zero the function value is 0 

        # values for the positive part of the control
        f_val[:,:,2] = -(-(y + h + k * q_plus) * q_plus) + (b * q_plus) * dufy \
            + q_plus * dufx

        # with this structure we know that the negative part of the optimal control
        # is the first entry in the third dimension and so on
        ind_q = np.argmin(f_val, 2)

        # initialize a array for the optimal control and fill it with the corresponding
        # values
        q_opt = np.zeros((problem.n_y, problem.n_x))
        q_opt[ind_q == 0] = q_minus[ind_q == 0]
        q_opt[ind_q == 1] = 0
        q_opt[ind_q == 2] = q_plus[ind_q == 2]

        return q_opt

At last we have to give this functions as attribute to the :obj:`HjbProblem` class
check the inputs, discretize and solve the problem

.. code-block:: python

    # user defined functions
    problem.u_0 = u_0
    problem.u_D = u_D
    problem.f = f
    problem.dx = dx
    problem.ddx = ddx
    problem.q_opt_user = q_optim

    # check inputs, disretize and solve the problem
    solve_hjb(problem)


References
~~~~~~~~~~

.. [2] Silke Glas u. a. „Intraday renewable electricity trading: advanced modeling
        and numerical optimal control“. In: Journal of Mathematics in Industry 10.1
        (2020), S. 3. ISSN: 2190-5983. DOI: 10.1186/s13362- 020- 0071- x. URL:
        https://doi.org/10.1186/s13362-020-0071-x.


Solution algorithm
------------------

For solving the HJB equation we are using the well known Howard-Algorithm. The 
algorithm consists of two steps. At the beginning we have to choose a initial
control :math:`q^{(0)} \in \mathcal{Q}`. After that we iterate as long as the 
termination criterion is bigger than a specified tolerance 
:math:`||u^{(l)} - u^{(l - 1)}||_{\infty} > \varepsilon_{tol}`. In the first step
we solve the linear system from the discretization in space and time (since the 
control is fixed) 

.. math::
    \left( [I - h_t \theta A^q] u(t_k, x_i) = \left[ I + h_t (1 - \theta) A^q \right]
    u(t_{k+1}, x_i) + h_t f(t_k, x_i, q)  \right)^{(l)}
    :label: eq:howard_lin

After this we update the control acording to 

.. math::
    q^{(l+1)} = \text{arg}\,\max_{q \in \mathcal{Q}} \left\{-\theta A^q u(t_k, x_i)
    - (1 - \theta) A^q u(t_{k+1}, x_i) - f(t_k, x_i, q) \right\}
    :label: eq:arg_max

in the :math:`\max` case and analougus in the :math:`\min` case. Equation 
:math:numref:`eq:arg_max` implys that we have to solved a optimization problem at
every discretization point in space and time.

We are solving the linear system of equations at each iteration in equation 
:math:numref:`eq:howard_lin` with the :obj:`scipy.sparse.linalg.gmres` solver. 

For the Howard-Algorithm we can set these parameters

    * :obj:`HjbProblem.tol` tolerance for the convergence of the howard algorithm (default = ``1e-6``)
    * :obj:`HjbProblem.tol_gmres` tolerance for the :obj:`scipy.sparse.linalg.gmres` solver for the solution of the linear system of equation (default = ``1e-9``)
    * :obj:`HjbProblem.max_iter` maximal amount of iterations in the howard algorithm (default = ``100``)

.. note::

    The default value for :obj:`HjbProblem.max_iter` can be to low if the term 
    :math:`h_t/h_x >> 1` for the convergence of the howard algorithm.
    
.. autosummary::
    :toctree: generated

    hjb_solver

Optimization methods
--------------------

To solve the optimization problem in equation :math:numref:`eq:arg_max` four
different options are implemented. Each method comes with it own advantages and
disadvantages. 

At first we are looking at the :obj:`brute` method. This method discretizes the
control and performs a line search for the best function value in the sup/inf. 
This version performs the comparison at each discretized control value to save 
some memory. This can be much slower than the :obj:`brute_all` method if the 
control discretization is fine since python is quite slow with loops and function
calls to C routines. But you shouldn't run out of memory this can only happen if
your space discretization is pretty fine.

Next is the :obj:`brute_all` method. This is conceptual the same as the 
:obj:`brute` method with one difference. To gain some performance the line search
is performed at all discretization points for the control at once. This means that
it uses much more memory. The memory consumption is in the order 
:math:`\mathcal{O}(n_x \cdot n_y \cdot n_q)`. This means you can easily run out of
memory for a fine control and space discretization. But you gain some performance
since we are saving some calls to the underlying :obj:`numpy.argmax` funtion.

The third method for the optimization problem is via user defined function 
:obj:`optim_user_func`. If you have a optimality criteria wich can be derived e. g.
analyticaly the normal way is to insert this optimality condition into the HJB and solve 
the nonlinear partial differential equation. This doesn't have to be done with this method.
This has the advantage that one does not have to perform the error prone algebraic manipulations
if the optimality criteria is lengthy. Anothter benefit is that you don't have to write 
a new discretization scheme for the resulting non linear PDE as long as the HJB is in the form
of equation :math:numref:`eq:hjb_min` and equation :math:numref:`eq:hjb_max`. This is by 
far the fastest method.

.. autosummary::
    :toctree: generated

    brute
    brute_all
    scalar
    optim_user_func

Classes
-------

.. autosummary::
    :toctree: generated

    HjbProblem
    Solver
    HjbOperator
"""

from .hjb_problem import HjbProblem
from .hjb_solver import solve_hjb
from .solver_helper import Solver
from .hjb_operator import HjbOperator
from .optimization import brute, brute_all, scalar, optim_user_func
from . import hjb_problem
from numpy import ndarray
from . import hjb_solver