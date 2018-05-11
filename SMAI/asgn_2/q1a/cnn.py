
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import matplotlib as mpl
from tensorflow.examples.tutorials.mnist import input_data
import cv2

def xrange(x):
	return iter(range(x))

def show(image,str):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	plt.suptitle(str)
	imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
	imgplot.set_interpolation('nearest')
	#ax.xaxis.set_ticks_position('top')
	#ax.yaxis.set_ticks_position('left')
	plt.show()


# RELU ACTIVATION FN
def activation(x):
	if(x>=0):
		return x
	return 0
    
def do_convolution(image, cfilter, base):
    (cfilter_x, cfilter_y, cfilter_z) = cfilter.shape
    im_x, im_y, im_z = image.shape
    r1 = []
    dot_sum = 0
    tem = im_x - cfilter_x +1 
    for i in range(tem):
        r2 = []
        for j in range(im_y - cfilter_y + 1):
            im = image[i:i+cfilter_x,j:j+cfilter_y,:]
            dot_sum = np.dot(im.flatten(), cfilter.flatten()) + base
            r2.append(dot_sum)
        r1.append(r2)
    w = np.array(r1)
    ans = w.reshape(w.shape[0],w.shape[1],1)
    return ans

def convolution_layer(image, filters):
    output = []
    im_x,im_y,im_z = image.shape
    for cfilter in filters:
        cfilter_x, cfilter_y, cfilter_z = cfilter.shape
        conv = do_convolution(image, cfilter, 0)
        p1 = im_x - cfilter_x + 1
        p2 = im_y - cfilter_y + 1
        p3 = im_z - cfilter_z + 1
        output.append(f(conv[:p1, :p2, :p3]))
    return output        

def get_filters(x,y,z,n):
    output = []
    for i in range(n):
        r=np.random.rand(x,y,z)*2
        output.append(r.astype('int'))
    return output

def calculate(array):
    return np.amax(array)

# def pooling_layer(imset, layer_dim, stride):
#     output = []
#     l, h = layer_dim
#     for r in imset:
#         x = r.reshape(r.shape[0],r.shape[0])
#         out = []
#         for i in range(0,x.shape[0]-l+1,stride):
#             row = []
#             for j in range(0,x.shape[1]-h+1,stride):
#                 row.append(calculate(x[i:i+l, j:j+h]))
#             out.append(row)
#         output.append(np.asarray(out))
#     return np.asarray(output), np.dstack(output)

f = np.vectorize(activation)

#### Begin
#### Read Image
im = cv2.imread("im1.jpeg")
r = 32
dim = (32, 32)

# perform the actual resizing of the image and show it
resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
#print(resized.shape)
im = resized  # Input : (32x32x3)

filters = get_filters(5,5,im.shape[2],6)

t = convolution_layer(im,filters)
##### Output 1 : 6@(28x28)

print("Output Size :")
x,y,_=t[0].shape
print("{}@{}*{}".format(len(t),x,y))

for r in t:
	str = "Convolutional Layer 1"
	show(r.reshape(r.shape[0],r.shape[0]),str)

#pooling_layer 1
list_out2, out2 = pooling_layer(t,(2,2),2)

#pooling_layer layer 1 output

print("Output Size :")
x,y,z=list_out2.shape
print("{}@{}*{}".format(x,y,z))

for r in list_out2:
	str = "Pooling Layer 1"
	show(r,str)
### ##### Output 2 : 6@(14x14)

filters = get_filters(5,5,out2.shape[2],16)
t2 = convolution_layer(out2,filters)

#convolutional layer 2 output
print("Output Size :")
x,y,_=t2[0].shape
print("{}@{}*{}".format(len(t2),x,y))

for r in t2:
	str = "Convolutional Layer 2"
	show(r.reshape(r.shape[0],r.shape[0]),str)
##### Output 3 : 16@(10x10)

# pooling_layer 2
list_out3, out3 = pooling_layer(t2,(2,2),2)

print("Output Size :")
x,y,z=list_out3.shape
print("{}@{}*{}".format(x,y,z))

for r in list_out3:
	str = "Pooling Layer 2"
	show(r,str)
##### Output 4 : 16@(5x5)

#Fully connected 1
flat_mat = out3.flatten()  ### 400 neurons
w = np.asmatrix(np.random.rand(120,flat_mat.shape[0]+1))
x = np.asmatrix(np.append(flat_mat,1)) # Bias Term
w=w.T
x=x.T
valf = np.asarray(np.matmul(w.T,x))

#Fully connected 2
flat_mat = valf.flatten()
w = np.asmatrix(np.random.rand(84,flat_mat.shape[0]+1))
x = np.asmatrix(np.append(flat_mat,1))
w=w.T
x=x.T
valf2 = np.asarray(np.matmul(w.T,x))

#Fully connected 3
flat_mat = valf2.flatten()
w = np.asmatrix(np.random.rand(10,flat_mat.shape[0]+1))
x = np.asmatrix(np.append(flat_mat,1))
w=w.T
x=x.T
valf3 = np.asarray(np.matmul(w.T,x))

print("Final Output:")
print(valf3)

