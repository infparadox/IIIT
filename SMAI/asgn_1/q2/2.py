import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

Total_pts = 4

def plot(class_1,class_2):  # To plot 2d dataset Points
    y1 = []
    x1 = []
    for i in range(len(class_1)):
        x1.append((class_1[i][0]))
        y1.append((class_1[i][1]))
    y2 = []
    x2 = []
    for i in range(len(class_2)):
        x2.append((class_2[i][0]))
        y2.append((class_2[i][1]))
    plt.axis([-5, 5, -5, 5])
    plt_1, =plt.plot(x1, y1, 'ro',label="Class_1")
    plt_2, =plt.plot(x2, y2, 'go',label="Class_2")
    plt.legend([plt_1,plt_2], ["Class_1", "Class_2"])
    plt.legend()
    plt.xlabel('X1',fontsize=20)
    plt.ylabel('X2',fontsize=20)
    plt.legend(loc='best', shadow=True, fancybox=True, numpoints=1,fontsize=20)
    plt.title('Data Set Points',fontsize=20)
    plt.show()

def LeastSquare(dataset):  # Classifier using Least Square Approach
	X=dataset
	TX=X.transpose()
	res=np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
   	temp=np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
   		[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
   		[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
   	fin=np.array([[1.0],[1.0],[1.0]])
   	Y = np.array([[1.0],[1.0],[1.0],[1.0],[-1.0],[-1.0],[-1.0],[-1.0]])
   	res = np.dot(TX, X)
   	res = inv(res)
   	temp = np.dot(TX, Y)
   	W = np.dot(res, temp)
   	print "for Least Square W:"
   	print W
   	return W

def plot1(out, class_1, class_2, out1, dataset): # To plot both Classifiers in single plot
    x1 = []
    y1 = []
    for i in range(len(class_1)):
        x1.append(float(class_1[i][0]))
        y1.append(float(class_1[i][1]))
    x2 = []
    y2 = []
    for i in range(len(class_2)):
        x2.append(float(class_2[i][0]))
        y2.append(float(class_2[i][1]))
    xmin =-15 
    xmax = 15
    for i in dataset:
        xmin = min(xmin, i[0])
        xmax = max(xmax, i[0])
    #print xmin, xmax
    x = np.linspace(xmin, xmax)
    y = -1*(((out1[0] * x) + out1[2])/out1[1])   # First Classifier
    plt_1,= plt.plot(x,y,c="green",label="Least Square")
    plt.axis([-15, 15, -15, 15])
    plt.plot(x1, y1, 'ro')
    plt.plot(x2, y2, 'go')
    plt.axis([-15, 15, -15, 15])
    plt_2,= plt.plot([-15,15],[-(float(out[0][1])*(-15.0) - out[1])/float(out[0][0]), 
            -(out[0][1]*15 - out[1])/out[0][0]],linestyle = '-',linewidth=2,label="FLD")  # Second Classifier
    plt.legend([plt_1,plt_2], ["Least Square", "FLD"],fontsize=20)
    plt.legend(loc='best', shadow=True, fancybox=True, numpoints=1,fontsize=20)
    plt.xlabel('X1',fontsize=20)
    plt.ylabel('X2',fontsize=20)
    plt.title('Classifiers',fontsize=20)
    plt.show()

def FLD(class_1, class_2):   # Fisher's Linear Discriminant
    avg1 = np.array([[0.0], [0.0]])
    avg2 = np.array([[0.0], [0.0]])
    for i in range(len(class_1)):
        avg1[0][0] += float(class_1[i][0])
        avg1[1][0] += float(class_1[i][1])    
    for i in range(len(class_2)):
        avg2[0][0] += float(class_2[i][0])
        avg2[1][0] += float(class_2[i][1])
    avg1[0][0] = float(avg1[0][0])/float(len(class_1))
    avg1[1][0] = float(avg1[1][0])/float(len(class_1))    
    avg2[0][0] = float(avg2[0][0])/float(len(class_2))
    avg2[1][0] = float(avg2[1][0])/float(len(class_2))    
    S1 = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in range(len(class_1)):
        t1 = np.array([[float(class_1[i][0])],[float(class_1[i][1])]])
        t1 = np.subtract(t1, avg1)
        r1 = np.dot(t1, t1.transpose())
        S1 = np.add(S1, r1)
    S2 = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in range(len(class_2)):
        t2 = np.array([[float(class_2[i][0])],[float(class_2[i][1])]])
        t2 = np.subtract(t2, avg2)
        r2 = np.dot(t2, t2.transpose())
        S2 = np.add(S2, r2)
    SW = np.add(S1, S2)
    SW = inv(SW)
    W = np.dot(SW, np.subtract(avg1, avg2))
    print "For Fisher W:"
    print W
    # finding the intercept
    intercept = 0
    for i in class_1:
        intercept = intercept + np.dot(W.transpose(), i)

    for i in class_2:
        intercept = intercept + np.dot(W.transpose(), i)
    print intercept/8
    return (W, intercept/8)
    #eturn (W, 0)

######## Begin ########
#### Dataset1
dataset2 = np.array([[3,3,1],[3,0,1],[2,1,1],[0,1.5,1],[-1,1,1],[0,0,1],[-1,-1,1],[1,0,1]])  # Append with 1
class2_1 = np.array([[3, 3],[3, 0],[2, 1],[0, 1.5]])  # Class 1 pts
class2_2 = np.array([[-1, 1],[0, 0],[-1, -1],[1, 0]]) # Class 2 pts
#### Dataset2
dataset1 = np.array([[3,3,1],[3,0,1],[2,1,1],[0,2,1],[-1,1,1],[0,0,1],[-1,-1,1],[1,0,1]])  # Append with 1
class1_1 = np.array([[3, 3],[3, 0],[2, 1],[0, 2]])   # Class 1 pts
class1_2 = np.array([[-1, 1],[0, 0],[-1, -1],[1, 0]]) # Class 2 pts
#### For dataset1
# plot(class1_1,class1_2)
# out_1 = LeastSquare(dataset1)
# out_2 = FLD(class1_1, class1_2)
# plot1(out_2, class1_1, class1_2, out_1, dataset1)
#### For dataset2
plot(class2_1,class2_2)
out_1 = LeastSquare(dataset2)
out_2 = FLD(class2_1, class2_2)
plot1(out_2, class2_1, class2_2, out_1, dataset2)