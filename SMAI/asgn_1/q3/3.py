import numpy as np
from os import listdir
import os.path
import math
from scipy.linalg import svd
from scipy.spatial.distance import cosine
import random

def cosine_similarity(train_data, test_data):
    misclassified = 0
    for test_vec in test_data:
        similarity = []
        cnt_labels = [0 for i in range(len(listdir('./q2data/train/')))]
        for train_vec in train_data:
            similarity.append((1 - cosine(test_vec[0], train_vec[0]), train_vec[1]))

        similarity = sorted(similarity)[-10:]

        for entry in similarity:
            cnt_labels[int(entry[1])] += 1

        maxval = max(cnt_labels)
        idx = 0
        for i in range(len(cnt_labels)):
            if cnt_labels[i] == maxval:
                idx = i
                if i == int(test_vec[1]):
                    break

        if idx != int(test_vec[1]):
            misclassified += 1

    return 1 - (misclassified / float(len(test_data)))


def one_vs_all_perceptron(train_data, test_data):
    #import ipdb; ipdb.set_trace()
    weights = []
    for curr_class in range(len(listdir('./q2data/train/'))):
        W = np.zeros(train_data[0][0].shape)
        B = 0
        epochs = 10
        output = []

        C = 1
        for i in range(epochs):
            for td in train_data:
                Y_cap = np.dot(W, td[0]) + B
                curr_label = 1
                if int(td[1]) == int(curr_class):
                    curr_label = 1
                else:
                    curr_label = -1

                if (curr_label * Y_cap) <= 0:
                    output.append((W, B, C))
                    W += curr_label * td[0]
                    B += curr_label
                    C = 1
                else:
                    C = C + 1

        weights.append((output, curr_class))

    misclassified = 0
    for td in test_data:
        maxclass = 0
        maxval = -100000000000
        for weight in weights:
            Y_cap = 0
            for k in range(len(weight[0])):
                Y_cap += weight[0][k][2]*np.sign(np.dot(weight[0][k][0],td[0])+weight[0][k][1])
            if maxval < Y_cap:
                maxval = Y_cap
                maxclass = int(weight[1])

        if int(td[1]) != int(maxclass):
            misclassified += 1

    return 1 - (misclassified/float(len(test_data)))


fp_stopwords = open("stopwords.txt",'r')
stopwords = (fp_stopwords.read()).split('\n')

word_dict = {}
doc_to_address = {}
doc_count = 0
ignorechars = '''."/?\[]{}(),:'!'''
doc_label = []

for direc in listdir('./q2data/train'):
    locs = './q2data/train/'+direc+'/'
    files = listdir(locs)
    for filename in files:
        doc_label.append(direc)
        fp_data = open(locs+filename,'r')
        tokens = ((fp_data.read())).split()
        for w in tokens:
            w = w.lower().translate(None, ignorechars)
            if w in stopwords:
                continue
            if w in word_dict:
                word_dict[w].append(doc_count)
            else:
                word_dict[w] = [doc_count]
        doc_to_address[doc_count] = locs+filename
        doc_count += 1

dictkeys = word_dict.keys()
#dictkeys.sort()

A = np.zeros([len(dictkeys), doc_count])

for i, k in enumerate(dictkeys):
    for d in word_dict[k]:
            A[i,d] += 1

WordsPerDoc = np.sum(A, axis=0)
DocsPerWord = np.sum(np.asarray(A > 0), axis=1)

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
            A[i,j] = (A[i,j] / WordsPerDoc[j]) * math.log(1 + (float(A.shape[1]) / DocsPerWord[i]))

A = A.transpose()
#w, v = np.linalg.eig(np.matmul(A, A.transpose()))
s = np.arange(len(doc_label))
random.shuffle(s)
A = A[s]
temparr = []
for si in s:
    temparr.append(doc_label[si])
doc_label = temparr

breakup_point = (len(doc_label)*80)/100

training = A[:breakup_point]
train_doc_label = doc_label[:breakup_point]
test = A[breakup_point:]
test_doc_label = doc_label[breakup_point:]

umat, smat, vh = np.linalg.svd(training, full_matrices=False)

discard = 200

smat = smat[:-discard]
umat = umat[:,:-discard]
vh = vh[:-discard,:]
#print smat

train_reduced_doc = []
for doc, label in zip(training, train_doc_label):
    train_reduced_doc.append((np.matmul(doc,vh.transpose()), label))

test_reduced_doc = []
for doc, label in zip(test, test_doc_label):
    test_reduced_doc.append((np.matmul(doc,vh.transpose()), label))

print "cosine similarity accuracy: "
print cosine_similarity(train_reduced_doc, test_reduced_doc) * 100

print "one vs all perceptron accuracy: "
print one_vs_all_perceptron(train_reduced_doc, test_reduced_doc) * 100
