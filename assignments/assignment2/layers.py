import numpy as np


def _softmax(z):
    # z -= np.max(z)
    return np.exp(z - np.max(z)) / (np.sum(np.exp(z - np.max(z))))

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        return _softmax(predictions)
    else:
        return np.apply_along_axis(_softmax, 1, predictions)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    y_true = np.zeros(probs.shape)
    if isinstance(target_index, int):
        y_true[target_index] = 1
    else:
        count = 0
        for sample_y_true in target_index:
            y_true[count][sample_y_true] = 1
            count += 1

    return -np.sum(y_true * np.log(probs))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    y_true = np.zeros(predictions.shape)
    if isinstance(target_index, int):
        y_true[target_index] = 1
    else:
        count = 0
        for sample_y_true in target_index:
            y_true[count][sample_y_true] = 1
            count += 1

    softmax_prob = softmax(predictions)
    loss = cross_entropy_loss(softmax_prob, target_index)

    dprediction = softmax_prob - y_true

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    
    f = lambda x: reg_strength * np.sum(np.square(x))

    loss = reg_strength * np.sum(W ** 2)

    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])

    grad = W.copy()
    while not it.finished:
        ix = it.multi_index

        bigger_x = W.copy()
        bigger_x[ix] += reg_strength
        lesser_x = W.copy()
        lesser_x[ix] -= reg_strength

        numeric_grad_at_ix = (f(bigger_x) - f(lesser_x)) / (2 * reg_strength)
        grad[ix] = numeric_grad_at_ix
        it.iternext()

    return loss, grad
    


def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W

    loss, dW = softmax_with_cross_entropy(predictions, target_index)

    dW = X.transpose().dot(dW)

    return loss, dW


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        out = np.maximum(0,X)
        self.cashe = X
        return out

    def backward(self, d_out):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops       
        d_result = d_out.copy()
        d_result[self.cashe < 0] = 0     
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        output = np.dot(self.X, self.W.value) + self.B.value
        return output

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment        
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out)
        d_input = np.dot(d_out, self.W.value.T)        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
