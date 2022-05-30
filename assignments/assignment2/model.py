import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.layer2 = ReLULayer()
        self.layer3 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        params = self.params()
        
        for param_key in params:
            param = params[param_key]
            param.grad = np.zeros_like(param.grad)
            
        step1 = self.layer1.forward(X)
        step2 = self.layer2.forward(step1)
        step3 = self.layer3.forward(step2)
        
        loss, dL = softmax_with_cross_entropy(step3, y)
        
        dstep3 = self.layer3.backward(dL)
        dstep2 = self.layer2.backward(dstep3)
        dstep1 = self.layer1.backward(dstep2)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param_key in params:
            param = params[param_key]
            loss_p, grad_p = l2_regularization(param.value, self.reg)
            param.grad += grad_p
            loss += loss_p

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set
        Arguments:
          X, np array (test_samples, num_features)
        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        #pred = np.zeros(X.shape[0], np.int)
        step1 = self.layer1.forward(X)
        step2 = self.layer2.forward(step1)
        step3 = self.layer3.forward(step2)
        probs = softmax(step3)
        pred = np.array(list(map(lambda x: x.argsort()[-1], probs)))

        return pred

    def params(self):
        return {'W1': self.layer1.W, 'B1': self.layer1.B, 'W2': self.layer3.W, 'B2': self.layer3.B}
