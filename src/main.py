# -*- coding utf-8 -*-

from typing import Final, Callable, Tuple, List

from numpy.typing import NDArray
from numpy.linalg import norm

from numpy import (hstack, ones, outer, clip,
				   float64, int64, array, inf,
				   sum, exp, zeros, eye, diag,
				   concatenate, dot, log1p, abs)

from pandas import DataFrame, read_excel

df	: Final[DataFrame]
y	: Final[NDArray[int64]]
X	: Final[NDArray[float64]]

df	= read_excel(
		'../data/titanic.xlsx',
		engine = 'openpyxl')

y	= df['survived'].to_numpy()
X	= df.drop(
		columns=['survived', 'name']
	).to_numpy(dtype=float64)

def func_loglikelihood(wbar: NDArray[float64], X: NDArray[float64], y: NDArray[int64]) -> float64:

	"""
	
	Description
	===========

	Compute the log-likelihood value for logistic regression.
	
	Parameters
	----------

	wbar : NDArray[float64] | Vector-like
		Model parameters (With bias term) ; shape (p + 1,)
	
	X : NDArray[float64] | Matrix-like
		Feature matrix ; shape (N, p)

	y : NDArray[int64] | Vector-like
		Target vector ; shape (N)
	
	Returns
	-------

	float64 : The log-likelyhood value

	"""

	z	: Final[NDArray[float64]]
	z	= clip(wbar[0] + X @ wbar[1:], -500, 500)
	return sum(y * z - log1p(exp(z)))


def grad_loglikelihood(wbar: NDArray[float64], X: NDArray[float64], y: NDArray[int64]) -> NDArray[float64]:

	"""

	Description
	===========

	Compute the gradient of the log-likelihood for
	logistic regression (Sigmoid activation function).

	Parameters
	----------

	wbar : NDArray[float64] | Vector-like
		Model parameters (With bias term) ; shape (p + 1,)

	X : NDArray[float64] | Matrix-like
		Feature matrix ; shape (N, p)

	y : NDArray[int64] | Vector-like
		Target vector ; shape (N)

	Returns
	-------

	NDArray[float64] : Gradient vector ; shape (p + 1,)

	"""

	offsets	: Final[NDArray[float64]]
	offsets	= y - (1 / (1 + exp(-clip(wbar[0] + X @ wbar[1:], -500, 500))))

	return concatenate(([sum(offsets)], X.T @ offsets))


def hes_loglikelihood(wbar: NDArray[float64], X: NDArray[float64]) -> NDArray[float64]:

	"""

	Description
	===========

	Compute the Hessian of the log-likelihood for logistic regression.

	Parameters
	----------

	wbar : NDArray[float64] | Vector-like
		Model parameters (With bias term) ; shape (p + 1,)

	X : NDArray[float64] | Matrix-like
		Feature matrix ; shape (N, p)

	Returns
	-------

	NDArray[float64] : Hessian matrix ; shape (p + 1, p + 1)

	"""

	pred	: Final[NDArray[float64]]
	biasX	: Final[NDArray[float64]]

	pred	= 1 / (1 + exp(-clip(wbar[0] + X @ wbar[1:], -500, 500)))
	biasX	= hstack((ones((X.shape[0], 1)), X))

	return -(biasX.T @ diag(pred * (1 - pred)) @ biasX)


def step_length_Wolfe(
		func		: Callable[[NDArray], float64],
		grad		: Callable[[NDArray], NDArray],
		start		: NDArray,
		direction	: NDArray,
		param		: Tuple[float64, float64, float64],
	) -> float64:

	"""

	Description
	===========

	Compute a step length that satisfies the Wolfe conditions for line search.

	Parameters
	----------

	func : Callable[[NDArray], float64]
		Function g: R^n -> R to minimize.

	grad : Callable[[NDArray], NDArray]
		Gradient of g.

	start : NDArray
		Starting point (s ∈ R^n) for line search.

	direction : NDArray
		Search direction (d ∈ R^n).

	param : Tuple[float64, float64, float64]
		Tuple containing values (c1, c2, α1).

	Returns
	-------
	float64 : Step length α satisfying Wolfe conditions.

	"""

	assert 0 < param[0] < param[1] < 1, \
	"c1 and c2 must satisfy 0 < c1 < c2 < 1."

	a	: float64
	ap	: float64
	an	: float64
	L	: float64
	U	: float64
	
	a	= param[2]
	ap	= 0.0
	an	= a
	L	= ap
	U	= inf

	phi_0	: Final[float64]			= func(start, X, y)
	phip_0	: Final[NDArray[float64]]	= dot(grad(start, X, y), direction)

	def phi(a: float64) -> float64:
		return func(start + a * direction, X, y)

	def phip(a: float64) -> NDArray[float64]:
		return dot(grad(start + a * direction, X, y), direction)

	while True:

		phi_a	= phi(an)
		phip_a	= phip(an)

		if (phi_a > phi_0 + param[0] * an * phip_0) \
			or (phi_a >= phi(ap) and L > 0):
			U = an ; an = 0.5 * (L + U)

		elif abs(phip_a) > param[1] * abs(phip_0):
			L = an; an = 2 * an if U == inf else 0.5 * (L + U)

		else:
			return an

		ap = an

		if abs(U - L) < 1e-8:
			return an


def line_search_BFGS(
		func			: Callable[[NDArray], float64],
		grad			: Callable[[NDArray], NDArray],
		x0				: NDArray,
		H0				: NDArray,
		iteration_limit	: int,
		epsilon			: float64,
		param			: Tuple[float64, float64, float64]
	) -> NDArray:

	"""

	Description
	===========

	Perform line search using the BFGS direction and Wolfe step lengths.

	Parameters
	----------

	func : Callable[[NDArray], float64]
		Function to minimize, defined on R^n.

	grad : Callable[[NDArray], NDArray]
		Gradient of the function to minimize.

	x0 : NDArray
		Starting point of the line search (in R^n).

	H0 : NDArray
		Initial approximation of the inverse Hessian (R^(n x n)).

	iteration_limit : int
		Maximum number of line search steps.

	epsilon : float64
		Gradient norm tolerance for stopping criterion.

	param : Tuple[float64, float64, float64]
		Tuple containing values (c1, c2, α1) for Wolfe step lengths.

	Returns
	-------

	NDArray : A matrix whose columns are the visited iterates, excluding the starting point.

	"""

	x		: NDArray[float64]
	H		: NDArray[float64]
	g		: NDArray[float64]
	s		: NDArray[float64]
	y2		: NDArray[float64]
	x2		: NDArray[float64]
	visited	: List[NDArray[float64]]

	x		= x0
	H		= H0
	visited	= [x]

	for _ in range(iteration_limit):

		g	= grad(x, X, y)

		if norm(g) < epsilon:
			break

		x2	= x + step_length_Wolfe(
				func, grad, x, -H @ g, param) * (-H @ g)

		visited.append(x2)

		s	= x2 - x
		y2	= grad(x2, X, y) - g

		if dot(y2, s) > 1e-10:
			rho	= 1.0 / dot(y2, s)
			V	= eye(x0.shape[0]) - rho * outer(s, y2)
			H	= V @ H @ V.T + rho * outer(s, s)

		x = x2

	return array(visited).T


def logistic_regression(X: NDArray[float64], y: NDArray[float64]) -> NDArray[float64]:

	"""

	Description
	===========

	Perform logistic regression using BFGS to maximize the log-likelihood.

	Parameters
	----------

	X : NDArray[float64]
		Feature matrix (N x p), where N is the number of samples and p is the number of features.

	y : NDArray[float64]
		Target vector (N,), where N is the number of samples.

	Returns
	-------

	NDArray[float64] : Vector w that approximately maximizes the log-likelihood.

	"""

	p		: Final[int64]
	wbar	: Final[NDArray[float64]]

	_, p	= X.shape
	wbar	= zeros(1 + p)

	def maximize_log(
			wbar	: NDArray[float64],
			X		: NDArray[float64],
			y		: NDArray[int64]
			)		-> NDArray[float64]:
		return -func_loglikelihood(wbar, X, y)

	def maximize_grad(
			wbar	: NDArray[float64],
			X		: NDArray[float64],
			y		: NDArray[int64]
		)			-> NDArray[float64]:
		return -grad_loglikelihood(wbar, X, y)

	return line_search_BFGS(
				maximize_log, maximize_grad,
				wbar, eye(p + 1), 1000, 1e-4,
				(1e-3, 0.9, 1e-2))[:, -1]


def logistic_prediction(X: NDArray[float64], wbar: NDArray[float64]) -> NDArray[float64]:

	"""

	Description
	===========

	Compute the prediction vector for a logistic regression model.

	Parameters
	----------

	X : NDArray[np.float64]
		Feature matrix (N x p), where N is the number of samples and p is the number of features.

	wbar : NDArray[np.float64]
		Weight vector (p + 1,), where the first element is the bias (w0), and the rest are weights.

	Returns
	-------

	NDArray[np.float64] : Prediction vector (N,), containing binary predictions (0 or 1).

	"""

	return ((1 / (1 + exp(-(X @ wbar[1:] + wbar[0])))) >= 0.5).astype(float64)


def accuracy(y: NDArray[float64], yhat: NDArray[float64]) -> float:

	"""

	Description
	===========

	Compute the accuracy of a model.

	Parameters
	----------

	y : NDArray[np.float64]
		True labels (N,), where N is the number of samples.

	yhat : NDArray[np.float64]
		Predicted labels (N,), where N is the number of samples.

	Returns
	-------

	float : Accuracy of the model, ranging from 0 to 1.

	"""

	assert y.shape == yhat.shape, \
	'Shapes of y and yhat must match.'

	return sum(y == yhat) / y.size

def main() -> None:

	''' Script entrypoint '''

	print("Training model...")

	weights: Final[NDArray[float64]] = logistic_regression(X, y)
	predictions: Final[NDArray[float64]] = logistic_prediction(X, weights)
	
	print("Model weights:", weights)
	print("Predictions shape:", predictions.shape)
	print("Target shape:", y.shape)
	print("Model accuracy:", accuracy(y, predictions))

if __name__ == '__main__':
	main()
