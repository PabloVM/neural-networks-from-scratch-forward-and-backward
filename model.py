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
      case _:
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

# Step 6 - make_loss
def make_loss(kind='cross_entropy'):
    """Return a classification loss_fn(logits, labels) -> (loss, d_logits).

    Inputs to loss_fn:
      logits: (batch, C) float array of raw class scores
      labels: (batch,) int array of class indices in [0, C)
    Outputs:
      loss: Python float, mean scalar loss over the batch (finite)
      d_logits: (batch, C) gradient of loss w.r.t. logits (finite)
    Must pass gradient_check, be minimized by confident correct predictions,
    and stay finite under saturated logits.
    """
    match kind:
      case "cross_entropy":
        def loss_fn(logits,labels):
          b,C = logits.shape
          exp_logits = np.exp(logits - np.max(logits,axis=1,keepdims=True))
          soft_max = exp_logits / np.sum(exp_logits,axis=1,keepdims=True)

          loss = float(np.mean(- np.log(soft_max[np.arange(b),labels])))
          one_hot = np.zeros_like(soft_max)
          one_hot[np.arange(b),labels] = 1
          d_logits = (soft_max - one_hot) / b
          return loss,d_logits

    return loss_fn

# Step 7 - make_sequential
def make_sequential(layers):
    """Compose protocol-honoring layers into one sequential model.

    Inputs:
      layers: list of layer dicts, each with
        forward(x) -> (y, cache),
        backward(dout, cache) -> (dx, grads_dict),
        params: dict of ndarrays (possibly empty).

    Returns a dict with:
      forward(x) -> (y, caches)
        y: final activation after applying every layer in order
        caches: opaque structure needed by backward
      backward(dout, caches) -> (dx, grads_list)
        dx: gradient w.r.t. the original input x
        grads_list: list of length len(layers); grads_list[i] is the
          grads_dict from layers[i] ({} for param-free layers)
      params: aggregated live view of all layer params, length len(layers),
        same order as layers (so in-place updates affect the model)
    """
    def forward(x):
      caches = []
      y=x
      for layer in layers:
        y,cache = layer["forward"](x)
        caches.append(cache)
        x = y 
      return y,caches
    def backward(dout,caches):
      grads_list = []
      dx=dout
      for layer,cache in reversed(list(zip(layers,caches))):
        dx,grads = layer["backward"](dout,cache)
        grads_list.append(grads)
        dout = dx
      grads_list.reverse()
      return dx,grads_list
    return{
      "forward":forward,
      "backward":backward,
      "params":[layer["params"] for layer in layers ]
    }

# Step 8 - forward_backward
def forward_backward(model, loss_fn, x, y):
    """Run one full forward-backward sweep on a batch.

    Inputs:
      model: sequential dict with 'forward', 'backward', 'params'
             model['forward'](x) -> (logits, caches)
             model['backward'](d_logits, caches) -> (dx, param_grads)
      loss_fn: callable (logits, y) -> (loss, d_logits)
      x: np.ndarray (batch, in_dim)
      y: np.ndarray (batch,) integer labels

    Returns:
      loss: float, scalar batch loss
      param_grads: nested np.ndarrays matching model['params'] layout
                   (gradients of loss w.r.t. every parameter)
    """
    #forward
    logits,caches = model["forward"](x)
    #loss
    loss,d_logits = loss_fn(logits,y)
    #backward
    dx,param_grads = model["backward"](d_logits,caches)

    return loss, param_grads

# Step 9 - make_optimizer
def make_optimizer(params, lr=1e-2, kind='sgd'):
    """Build an optimizer that updates params in place.

    Inputs:
      params: arrays, possibly nested in lists/dicts (or dict of arrays) to optimize
      lr: float learning rate
      kind: str algorithm name (e.g. 'sgd')

    Returns:
      dict with key 'step'. step(grads) applies one in-place update
      using grads structured like params. Parameter shapes must stay
      unchanged. Repeated steps must reduce a simple convex objective
      within a modest fixed budget and keep values finite.
    """
    def step(grads):
      for param,grad in zip(params,grads):
        if isinstance(param,np.ndarray):
            param -= lr * grad
        elif isinstance(param,dict):
          for k in param.keys():
            param[k] -= lr * grad[k] 
    return{
      "step":step
    }

# Step 10 - train_step
def train_step(model, loss_fn, optimizer, x_batch, y_batch):
    """Perform one complete optimization step over a minibatch.

    Inputs:
      model: sequential model dict with 'forward', 'backward', and 'params'
      loss_fn: callable (logits, y) -> (loss, d_logits)
      optimizer: dict with 'step'(grads) applying in-place parameter updates
      x_batch: np.ndarray of shape (B, D)
      y_batch: np.ndarray of shape (B,) integer class labels

    Returns:
      loss: float, scalar batch loss evaluated BEFORE the parameter update.
      Model parameters are updated in place; shapes unchanged and values finite.
    """
    loss,param_grads = forward_backward(model,loss_fn,x_batch,y_batch)

    optimizer["step"](param_grads)

    return loss

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

