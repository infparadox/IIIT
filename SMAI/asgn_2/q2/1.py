import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import pandas as pd

def activation_func(x,param):
	if(param==1):
		return np.tanh(x)
	return 1 / (1 + np.exp(-x))

def CrossVal(X, y, k, theta,param):
    [a, b] = X.shape
    crossValAcc = []
    foldSize = len(X) / int(k)
    for foldNum in range(int(k)):
        y_train = np.array([])
        X_test = np.array([])
        y_test = np.array([])
        X_train = np.array([])
        for j in range(len(X)):
            if (j / foldSize) == foldNum:
                y_test = np.append(y_test, y[j])
                X_test = np.append(X_test, X[j])
            else:
                y_train = np.append(y_train, y[j])
                X_train = np.append(X_train, X[j])
        y_train = y_train.astype(int)
        X_test = X_test.reshape(len(y_test), b)
        y_test = y_test.astype(int)
        X_train = X_train.reshape(len(y_train), b)
        model = build_model(theta, param, len(X_train), X_train, y_train, 5, print_loss=True)
        prediction = []
        for x in X_test:
            prediction.append(predict(model, x))
        misclassifications = 0
        for i in range(len(X_test)):
            if prediction[i] != y_test[i]:
                misclassifications += 1
        crossValAcc.append(float(len(X_test) - misclassifications) / len(X_test))
    return crossValAcc

def readfile(filename):
    df = pd.read_csv(filename)
    some_values = [1, 2, 3]
    df = df.loc[(df['A35'].isin(some_values))]
    df = df[df.A34 != '?']
    df.A34 = df.A34.astype(float)
    df.to_csv('dataInput.csv', sep = ' ')
    X = np.array(df.drop(['A35'], axis = 1))
    Y = np.array(df['A35'])
    Y = Y - 1
    return [X, Y]

# Helper function to evaluate the total loss on the dataset
# def calculate_loss(X, y, num_examples, model):
#     W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
#     # Forward propagation to calculate our predictions
#     z1 = X.dot(W1) + b1
#     # a1 = np.tanh(z1)
#     a1 = activation_func(z1,param)  ### 1 for tanh , 0 for sigmoid
#     z2 = a1.dot(W2) + b2
#     exp_scores = np.exp(z2)
#     probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
#     # Calculating the loss
#     corect_logprobs = -np.log(probs[range(num_examples), y])
#     data_loss = np.sum(corect_logprobs)
#     # Add regulatization term to loss (optional)
#     # data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
#     return 1./num_examples * data_loss

# Helper function to predict an output (0 or 1)
# def predict(model, x):
#     W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
#     # Forward propagation
#     z1 = x.dot(W1) + b1
#     # a1 = np.tanh(z1)
#     a1 = activation_func(z1,param)  ### 1 for tanh , 0 for sigmoid
#     z2 = a1.dot(W2) + b2
#     exp_scores = np.exp(z2)
#     probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
#     return np.argmax(probs, axis=1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(theta, param, num_examples, X, y, nn_hdim, num_passes=20000, print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(inputdim, nn_hdim) / np.sqrt(inputdim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, outputdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, outputdim))
    # This is what we return at the end
    model = {}
    # Gradient descent. For each batch...
    one = 0
    loss = 1
    i = 0
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    #for i in xrange(0, num_passes):
        # Forward propagation
    while loss > theta and i < 20000:
        i += 1
        z1 = X.dot(W1) + b1
        # a1 = np.tanh(z1)
        a1 = activation_func(z1,param)  ### 1 for tanh , 0 for sigmoid
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        if one == 0:
            one = 1
            # print probs
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        # delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        if(param==1):
        	delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
       	else :
       		delta2 = delta3.dot(W2.T) * (a1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)


        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        loss = calculate_loss(X, y, num_examples, model)
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(X, y, num_examples, model))
    return model

[X, y] = readfile('dermatology.data')
[_, b] = X.shape

outputdim = 3 # output layer dimensionality
inputdim = b # input layer dimensionality


theta = 0.001 # cutoff if loss less than theta
epsilon = 0.001
param = 0  ## Sigmoid for 0

Accuracies = CrossVal(X, y, 5.0, theta,param)
accu=0
for accuracy in Accuracies:
    accu+=accuracy
accu=accu/len(Accuracies)
print(Accuracies)
print accu
