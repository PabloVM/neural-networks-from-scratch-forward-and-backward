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

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    match kind:
      case "relu":
        def forward(x):
          y = np.maximum(x,0)
          cache = x>0
          return y,cache
        def backward(dout,cache):
          dx = dout * cache
          return dx,{}
    
    return {
      "params" : {},
      "forward" : forward,
      "backward": backward
    }

# Step 5 - initialize_weights
def initialize_weights(in_dim, out_dim, scheme='he'):
    """Return (W, b) for a dense layer.

    Inputs:
      in_dim: int fan-in
      out_dim: int fan-out
      scheme: str initialization family (default 'he')

    Returns:
      W: np.ndarray shape (in_dim, out_dim), finite, symmetry-breaking,
         scale stable with depth (fan-in dependent)
      b: np.ndarray shape (out_dim,), near zero
    """
    match scheme:
      case "he":
        param = np.sqrt(2/in_dim)
    
    W = np.random.randn(in_dim,out_dim) * param
    b = np.zeros(out_dim)
    return W,b

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

