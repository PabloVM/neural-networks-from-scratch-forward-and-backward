"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    # TODO: Estimate the gradient of scalar f w.r.t. array x via central finite differences
    grad = np.zeros_like(x,dtype=float)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        x[idx] = orig
    return grad

# Step 2 - gradient_check
def gradient_check(analytic_grad, numeric_grad, tol=1e-5):
    # TODO: Return max relative error between analytic and numeric gradients.
    analytic_grad = np.asarray(analytic_grad)
    numeric_grad  = np.asarray(numeric_grad)
    abs_dif = np.abs(analytic_grad - numeric_grad)
    denominator = np.maximum(analytic_grad,numeric_grad)
    denominator = np.maximum(denominator,tol)
    error = abs_dif / denominator
    return float(np.max(error))

# Step 3 - make_dense
import numpy as np
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer.

    Inputs:
      in_dim: int, input feature size
      out_dim: int, output feature size
      weight_init_fn: callable(in_dim, out_dim) -> (W, b)

    Returns layer dict with keys:
      params: {'W': (in_dim, out_dim), 'b': (out_dim,)}
      forward(x) -> (y, cache) with y shape (batch, out_dim)
      backward(dout, cache) -> (dx, grads) with grads {'W', 'b'}
        Analytic dx/dW/db must match numerical_gradient via gradient_check.
    """
    W,b = weight_init_fn(in_dim,out_dim)
    def forward(x):
      y = x @ W + b 
      cache = (x,W)
      return y, cache
    def backward(dout,cache):
      x = cache[0]
      W = cache[1]
      dW = x.T @ dout
      db = np.sum(dout,axis=0)
      dx = dout @ W.T
      return dx, {"W":dW,"b":db}
    return{
      "params" : {"W":W,"b":b},
      "forward":forward,
      "backward":backward
    }

# Step 4 - make_activation (not yet solved)
# TODO: implement

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

