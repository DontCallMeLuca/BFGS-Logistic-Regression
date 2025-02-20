<h1 align="center">✨ BFGS Logistic Regression ✨</h1>

<h6 align="center"><em>Simple logistic regression machine learning model which uses the BFGS algorithm for optimizations</em></h6>

## 📝 Overview

This project implements logistic regression using the BFGS (Broyden–Fletcher–Goldfarb–Shanno) optimization algorithm for the Titanic survival prediction dataset. The implementation includes custom gradient computation, log-likelihood optimization, and Wolfe line search conditions.

The goal of this project is to try and predict the value of 'survived' by the other variables in the data set, except 'name', because it's not numerical.

## 📄 Dataset Overview
The dataset used for this particular project is the Titanic survival prediction dataset.
This gives us a clear binary goal for our logistic regression, since it's best applied on $0,1$ outputs 

| Feature       | Description      |
|---------------------|-------------------|
| **name** | Passenger name |
| **survived** | 1 if the passenger survived, 0 otherwise |
| **pclass** | Class the passenger was in (1 for first, 2 for second, 3 for third) |
| **female** | 1 if the passenger was female, 0 otherwise |
| **age** | Passenger age in years |
| **sibsp** | Number of siblings and spouses on board |
| **parch** | Number of parents and children on board |
| **fare** | Passenger fare |
| **embarkS** | 1 if the passenger embarked in Southampton, 0 otherwise |
| **embarkC** | 1 if the passenger embarked in Cherbourg, 0 otherwise |
| **embardQ** | 1 if the passenger embarked in Queenstown, 0 otherwise |

## 🔢 Mathematical Foundation

### Data Foundation

Because we are trying to predict if a passanger would have survived,
we can start by defining $N$ to be the number of passengers, and let

```math
y_i = \begin{cases} 1 & \text{if passenger }i\text{ survived}\\
0 & \text{otherwise}
\end{cases}
```

**For all**

```math
i = 1,...,N
```

The explanatory variables for this passenger are collected in the vector:

```math
x_i\in\mathbb{R}^{p}
```

The information for all passengers is gathered in the objects:

```math
\mathbf{y} =
\begin{bmatrix}
y_1 \\
\vdots \\
y_N
\end{bmatrix}
\quad \text{and} \quad
\mathbf{X} =
\begin{bmatrix}
\mathbf{x}_1^T \\
\vdots \\
\mathbf{x}_N^T
\end{bmatrix}
```

**Where:**
- y is the target vector
- X is the feature matrix

### Logistic Regression
Because we are trying to predict the binary variable $y_i$, we introduce the sigmoid function:

```math
\phi(t) = \frac{1}{1+e^{-t}}
```

**Where:**

```math
\phi:\mathbb{R}\rightarrow(0,1)
```
<br>

When $t \leq 0$, then $\phi(t)$ is nearly $0$, and if $t \geq 0$, then $\phi(t)$ is close to $1$.
This function is therefore suitable to convert some underlying variable to a survival probability.
Besides, this function has the useful properties:

```math
1 - \phi(t) = \frac{\phi(t)}{e^t} = \phi(-t)
```

Which will be used later.

We will model the survival probability of passenger $i = 1,...,N$ by:

```math
p_i = \phi(w_0 + w^Tx_i)
```

**Where:**
- $w_0\in\mathbb{R}$
- $w\in\mathbb{R}^p$

Are model parameters which will be determined later.

Given the survival probability $p_i$, the likelihood of observation $i$ is:

```math
\ell_i := \begin{cases}p_i & \text{if }y_i = 1\\
1 - p_i & \text{if }y_i = 0
\end{cases}
```

Because of the simple structure of this likelihood formula,
<br>
we can write the log-likelihood function as:

```math
\begin{align}
  f(w_0,w) &= \sum_{i=1}^{N}\log{\ell_i} \\
           &= \sum_{i=1}^{N}[y_i\log(p_i)+(1-y_i)\log(1-p_i)] \\
           &= \sum_{i=1}^{N}[y_i\log\left(\frac{p_i}{1-p_i}\right)+\log(1-p_9)]
\end{align}
```

Using both equalities, this can be further simplified to:

```math
\begin{align}
  f(w_0,w) &= \sum_{i=1}^{N}[y_i\log(\frac{p_i}{p_i}e^{w_0+w^Tx_i})+\log(\phi(-w_0-w^Tx_i))] \\
           &= \sum_{i=1}^{N}[y_i(w_0+w^Tx_i)-\log(1+e^{w_0+w^Tx_i})]
\end{align}
```

This log-likelihood function can be maximized over $w_0$ and $w$.
<br>
To this end, not that the gradient of the log-likelihood is:

```math
\begin{align}
  \nabla f(w_0,w) &= \sum_{i=1}^{N}\left(y_i
  \begin{bmatrix} 
    1 \\
    x_i 
  \end{bmatrix}
  -\frac{e^{w_0+w^Tx_i}}{1+e^{w_0+w^Tx_i}}
  \begin{bmatrix} 
    1 \\
    x_i 
  \end{bmatrix}\right) \\
  &= \sum_{i=1}^{N}\left(y_i-\phi(w_0+w^Tx_i)\right)
  \begin{bmatrix}
    1 \\
    x_i 
  \end{bmatrix}
\end{align}
```

It can be shown that $\phi'(t)=\phi(t)[1-\phi(t)]$, which implies taht the Hessian of the log-likelihood is:

```math
\nabla^2f(w_0,w) = -\sum_{i=1}^{N}\phi(w_0 + w^Tx_i)[1-\phi(w_0+w^Tx_i)]
\begin{bmatrix} 
  1 \\
  x_i 
\end{bmatrix}
\begin{bmatrix} 
  1 \\
  x_i 
\end{bmatrix}^T
```

For convenience of notation, the arguments of the log-likelihood can be combined in one vector:

```math
\bar{w}\equiv\begin{bmatrix} 
  w_0 \\
  w 
\end{bmatrix}
```

### Wolfe Conditions
The model uses line search to approximate stationary points of the log-likelihood function.
<br>
The step lengths will be chosen to satisfy the Wolfe conditions.

#### Problem Definition

**Find $\alpha > 0$ satisfying:**

```math
\phi(\alpha) \leq \phi(0) + c_1\alpha\phi'(0)
\phi'(\alpha) \geq c_2\phi'(0)
```

**Where:**

```math
\begin{alignat*}{2}
&\phi: \mathbb{R} \to \mathbb{R}, \quad \phi \in C^1(\mathbb{R})
&\phi'(0) < 0\\
&0 < c_1 < c_2 < 1\\
&\alpha_1 > 0
\end{alignat*}
```

#### Algorithm Formulation

```math
\begin{alignat*}{2}
& \alpha_0 = 0\\
& k = 1\\
& L = 0\\
& U = \infty\\
& \text{while }\phi(\alpha_k) > \phi(0) + c_1\alpha_k\phi'(0)\text{ or }\phi'(\alpha_k) < c_2\phi'(0)\text{ do}\\
& \text{if }\phi(\alpha_k) > \phi(0) + c_1\alpha_k\phi'(0)\text{ or }\phi'(\alpha_k) \geq 0\text{ or }\phi(\alpha_k) \geq \phi(\alpha_{k-1})\text{ then}\\
& \quad U \leftarrow \alpha_k\text{ if }\phi'(\alpha_k) \geq 0;\text{ otherwise }L \leftarrow \alpha_k\\
& \quad \alpha_{k+1} \leftarrow \begin{cases}
2\alpha_k & \text{if }U = \infty\\
\frac{1}{2}(L + U) & \text{otherwise}
\end{cases}\\
& \text{else}\\
& \quad \alpha_{k+1} \leftarrow 2\alpha_k\\
& k \leftarrow k + 1\\
& \text{return }\alpha_k
\end{alignat*}
```

### Predictions
Using the `logitic_regression(X, y)` function, we can estimate the values of $w_0$ and $w$.
We can define these estimates as $\hat{w}_0$ and $hat{w}$ respectively.
<br>
We can now use the estimated paramters to predict who was likely to survive the Titanic's journey.
<br>
THe survial prediction for passenger $i=1,...,N$ is:

```math
\hat{y}_i = \begin{cases}
1 & \text{if } \phi(\hat{w}_0 + \hat{w}^T x_i) \geq \frac{1}{2} \\
0 & \text{otherwise}
\end{cases}
```

These predictions can be combined in a vector:

```math
\hat{y}\in\mathbb{R}^N
```

To determine how good the predictions are, we can calculate the in-sample accuracy of the model.
This is the fraction of observations that were predicted correctly; in other words, the fraction of observations
$i=1,...,N$ for which $y_i=\hat{y}_i$. 

## 🔧 Implementation Details
- Uses numerical stability techniques including gradient clipping
- Implements type hints for better code maintainability
- Includes comprehensive docstrings for all functions
- Custom implementation of BFGS optimization algorithm
- Input validation for matrix dimensions
- Assertion checks for algorithm parameters

## 📝 Requirements
- Python 3.7+
- NumPy
- Pandas
- openpyxl

```sh
pip install numpy pandas openpyxl
```

## 💻 Usage

```python
# Train the model
weights = logistic_regression(X, y)

# Make predictions
predictions = logistic_prediction(X, weights)

# Calculate accuracy
acc = accuracy(y, predictions)
```

However, this is already defined in the script, therefore you can just run it as:

```sh
cd src
python main.py
```

## ✅ Results

The trained weights should look something like this:

```python
[ 1.42230346e+00 -1.00931975e+00  2.60894379e+00 -3.76857249e-02
 -3.48025217e-01  4.98522524e-02  4.63081938e-04  5.03541323e-01
  1.18261077e+00 -2.63848633e-01]
```

With an accuracy around 80%, which is all we can ask for with such a small dataset.

## 📃 License
This project uses the `GNU GENERAL PUBLIC LICENSE v3.0` license
<br>
For more info, please find the `LICENSE` file here: [License](LICENSE)
