from random import randrange
from csv import reader
from numpy import dot, array
import numpy as np
from matplotlib import pyplot as plt

norm = lambda x: -1 if x == 2 else 1

def readdata(filename):
	training_data = []
	with open(filename) as file:
		csv_reader = reader(file)
		
		cr = reader(file)
		for x in cr:
			e = x[-1]
			break
		for row in csv_reader:
			try:
				row.insert(0,1)
				if row[-1] == e:
					row[-1] = '-1'
				else:
					row[-1] = '1'
				x = [row[0]]+row[2:]
				training_data.append(array(x,dtype=float))
			except:
				continue
	return training_data


def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for i in range(len(folds)):
		fold = folds[i]
		train_set = folds[:i] + folds[i+1:]
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Make a prediction with weights
def predict_voted(row, weights):
	
	activation = np.zeros(len(weights[0][0]))
	count = 0
	for weight in weights:
		count += weight[1]
		activation = np.add(activation,weight[1]*weight[0])
	activation = activation/count
	# print activation
	value = dot(activation,row[:-1])
	return 1.0 if value >= 0.0 else 0.0

def weighted_mean(weights):
	activation = np.zeros(len(weights[0][0]))
	count = 0
	for weight in weights:
		count += 1
		activation = np.add(activation,weight[1]*weight[0])
	activation = activation/count
	return activation

def train_weights_voted(train, n_epoch):
	Voted_weights = []
	weights = np.zeros(len(train[0])-1)
	for epoch in range(n_epoch):
		c = 1
		for row in train:
			prediction = dot(weights,row[:-1]*row[-1])
			if prediction <= 0:
				Voted_weights.append((weights,c))
				weights = np.add(weights,row[-1]*row[:-1])
				# print weights
				c = 1
			else:
				c += 1
	return Voted_weights

def train_weights_online(train, n_epoch):
	# print len(train)
	weights = np.zeros(len(train[0])-1)
	for epoch in range(n_epoch):
		for row in train:
			prediction = dot(weights,row[:-1]*row[-1])
			if prediction <= 0:		
				weights = np.add(weights,row[-1]*row[:-1])
	return weights

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, n_epoch):
	check_class = lambda x: -1 if x < 0.0 else 1
	predictions = list()
	weights = train_weights_online(train, n_epoch)
	for row in test:
		prediction = dot(weights,row[:-1])
		predictions.append(check_class(prediction))
	return(predictions)

def voted_perceptron(train, test, n_epoch):
	check_class = lambda x: -1 if x < 0.0 else 1
	predictions = list()
	weights = train_weights_voted(train, n_epoch)
	weight_mean = weighted_mean(weights)
	for row in test:
		prediction = dot(row[:-1], weight_mean)
		predictions.append(check_class(prediction))
	return(predictions)

# load and prepare data
files = ["breastcancer.csv","ionosphere.csv"]
perceptrons = [perceptron,voted_perceptron]
n_folds = 10
means = []
for file in files:
	print("Printing results for %s" %(file))
	for p in perceptrons:
		dataset = readdata(file)
		#print(dataset)
		epochs = [10,15, 20, 25, 30, 35, 40, 45, 50]
		# print('Results for %s perceptron for %s' %(p,file))
		mean = []
		for epoch in epochs:
			scores = evaluate_algorithm(dataset, p, n_folds, epoch)
			m = sum(scores)/float(len(scores))
			mean.append((m,epoch))
			print('for %s Epochs accuracy is %.3f%%' % (epoch,m))
		means.append(mean)
		print "***********************************"
		print "Now results for Voted perceptron"
# voted_cancer --> black
# vanilla_cancer --> blue
# voted_ionospere --> red
# vanilla_ionosphere --> green
voted_cancer = array(means[1])
vanilla_cancer = array(means[0])
voted_ionospere = array(means[3])
vanilla_ionosphere = array(means[2])
plt.figure(1)
plt.title('Plot of Accuracy vs Epochs',fontsize=20)
plt.xlabel('Accuracy',fontsize=20)
plt.ylabel('Epochs',fontsize=20)
a=plt.scatter(voted_cancer[:, 0], voted_cancer[:, 1], color='b')
b=plt.scatter(vanilla_cancer[:, 0], vanilla_cancer[:, 1], color='r')
c=plt.scatter(voted_ionospere[:, 0], voted_ionospere[:, 1], color='g')
d=plt.scatter(vanilla_ionosphere[:, 0], vanilla_ionosphere[:, 1], color='y')
plt.legend((a,b, c, d),
           ('Voted_Cancer', 'Vanilla Cancer', 'Voted_Ionosphere', 'Vanilla_Ionosphere'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=20)
plt.show()