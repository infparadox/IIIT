import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, cross_validation , neighbors

def read(filename):
    df = pd.read_csv(filename)
    df=df.loc[(df['class'] == 0) | (df['class'] == 1) | (df['class'] == 2) | (df['class'] == 3)] # Take rows containing class : 1,2,3
    #df.to_csv('data2Input.csv', sep = ' ')
    X = np.array(df.drop(['class'], axis = 1)) # Drop last col
    Y = np.array(df['class'])
    #Y = Y - 1
    return [X,Y]

### If param=1(tanh) , else sigmoid
def activation_func(x,param):
	if(param==1):
		return np.tanh(x)
	return 1 / (1 + np.exp(-x))

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model,in_data,out_data,param):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
	X = in_data
	y = out_data
	z1=X.dot(W1)+b1
	#a1 = np.tanh(z1)
	a1 = activation_func(z1,param)  ### 1 for tanh , 0 for sigmoid
	z2 = a1.dot(W2) + b2
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	# Calculating the loss
	corect_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(corect_logprobs)
	# Add regulatization term to loss (optional)
	data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1./num_examples * data_loss

#Helper function to predict an output (0 or 1)

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    # a1 = np.tanh(z1)
    a1 = activation_func(z1,param)  ### 1 for tanh , 0 for sigmoid
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def calc_accuracy(predicted,actual):
	n = len(predicted)
	correct = 0.0
	for i in range(0,n):
		if(predicted[i]==actual[i]):
			correct+=1
	return(correct/n)
'''
This function learns parameters for the neural network and returns the model.
- nn_hdim: Number of nodes in the hidden layer
- num_passes: Number of passes through the training data for gradient descent
- print_loss: If True, print the loss every 1000 iterations
'''

def build_model(theta, in_data,out_data,nn_hdim, num_passes, param,print_loss=False):

    # Initialize the parameters to random values. We need to learn these.
    X = in_data
    y = out_data
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    one = 0
    loss = 1
    i = 0
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    # for i in xrange(0, num_passes):
        # Forward propagation
    while loss > theta and i < 20000:
        i += 1
        z1 = X.dot(W1) + b1
        #a1 = np.tanh(z1)
        a1 = activation_func(z1,param)  ### 1 for tanh , 0 for sigmoid
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  ## Softmax activation
        '''if one == 0:
            one = 1
            print probs
		'''
        # Backpropagation
        delta3 = probs    # Output Layer
        print delta3.shape
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        if(param==1):
        	delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
       	else :
       		delta2 = delta3.dot(W2.T) * (a1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        #dW2 += reg_lambda * W2
        #dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        loss = calculate_loss(model,X,y,param)
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model,X,y,param))

    return model


#### Begin ####
[X_train,y_train] = read('pendigits.tra')
[X_test,y_test] = read('pendigit.tes')
print(X_test.shape)
[_, b] = X_train.shape

nn_input_dim = b # input layer dimensionality
nn_output_dim = 4 # output layer dimensionality = No. of Classes

num_examples = len(X_train) # training set size

# Gradient descent parameters (I picked these by hand)
epsilon = 0.0001 # learning rate for gradient descent
reg_lambda = 0.001 # regularization strength
epochs = 20000 # Epochs
num_neurons = 10 # Neurons in hidden layer
param = 1# param = 1 (tanh) , else sigmoid
theta = 0.07

# Build a model with a 3-dimensional hidden layer
model = build_model(theta, X_train,y_train,num_neurons,epochs, param, print_loss=True)
out = predict(model,X_test)
print("Predicted Class:")
print(out)
print("Actual Class:")
print(y_test)
accuracy = calc_accuracy(out,y_test)
print("Accuracy:{}".format(accuracy))
