import random as rnd
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

class rand2DGaussian:
	def __init__(self):
		self.data = []
		self.label = []
		
	def add_data(self, classifier):
		self.label.append(classifier)
		if classifier == 1: #mean (-1, 1)
			self.data.append([np.random.normal(-1,1),np.random.normal(1,1)])
		elif classifier == -1: #mean (1,-1)
			self.data.append([np.random.normal(1,1),np.random.normal(-1,1)])
	
	#generate n random data points using add_data with labels -1 or 1
	def generate_data(self, n):
		for i in range(n):
			self.add_data(rnd.choice([-1,1]))

def isBetweenHyperplanes(x, w, b, margins):
	eq = x.dot(w) + b
	return margins[0] <= eq <= margins[1] or margins[1] <= eq <= margins[0]

#split data to training and testing
def split_data(data, labels, percent):
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	for i in range(len(data)):
		if rnd.random() < percent:
			test_data.append(data[i])
			test_labels.append(labels[i])
		else:
			train_data.append(data[i])
			train_labels.append(labels[i])
	return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
	# return train_data, train_labels, test_data, test_labels

def shuffle_data(data, labels):
	combined = list(zip(data, labels))
	rnd.shuffle(combined)
	data, labels = zip(*combined)
	return np.array(data), np.array(labels)

#score function for SVM
def get_score(test_data, test_labels, w, intercept):
	correct = 0
	for i in range(len(test_data)):
		if test_labels[i] == 1:
			if test_data[i].dot(w) + intercept > 0:
				correct += 1
		else:
			if test_data[i].dot(w) + intercept <= 0:
				correct += 1
	return correct/len(test_data)

class Classifier:
	def __init__(self):
		self.w = [0]*len(train_data[0])
		self.intercept = 0
		self.classes = 0
		self.selected_train_data = []
		self.selected_train_labels = []
		self.score = 0
		self.new_sample_added = False
		self.toggle_var = 1
		self.b = []
		self.b_pos = []
		self.b_neg = []
		self.regulizer = 0.05

	def sample_selection(self, training_sample):
		train_sample_data = training_sample[1:]
		train_sample_label = np.array([training_sample[0]])
		if len(self.classes) < 2:
			self.selected_train_data.append(train_sample_data)
			self.selected_train_labels.append(train_sample_label)
			self.new_sample_added = True
			self.classes = np.unique(self.selected_train_labels)
		else:
			if(self.toggle_var==train_sample_label):
				if(isBetweenHyperplanes(train_sample_data, self.w, self.intercept, self.b_neg) and train_sample_label == -1):
					self.selected_train_data.append(train_sample_data)
					self.selected_train_labels.append(train_sample_label)
					self.new_sample_added = True
					self.toggle_var = -train_sample_label
				if(isBetweenHyperplanes(train_sample_data, self.w, self.intercept, self.b_pos) and train_sample_label == 1):
					self.selected_train_data.append(train_sample_data)
					self.selected_train_labels.append(train_sample_label)
					self.new_sample_added = True
					self.toggle_var = -train_sample_label

	def train(self, train_data, train_label):
		train_data, train_label = shuffle_data(train_data, train_label)
		for i, (train_sample_data, train_sample_label) in enumerate(zip(train_data, train_label)):
			self.classes = np.unique(self.selected_train_labels)
			training_sample = np.concatenate((train_sample_label,train_sample_data))
			self.sample_selection(training_sample)
			
			if len(self.classes) == 2 and self.new_sample_added:
				#print('new sample', i, len(self.selected_train_data), self.new_sample_added, self.selected_train_labels[-1])
				self.new_sample_added = False
				self.selected_train_data = np.array(self.selected_train_data)
				self.selected_train_labels = np.array(self.selected_train_labels)
				Weights = cp.Variable((len(train_sample_data),1))
				gamma = cp.Variable()
				# hinge loss
				loss = cp.sum(cp.pos(1 - cp.multiply(self.selected_train_labels, self.selected_train_data @ Weights + gamma)))
				c = cp.norm(Weights, 1)
				slack = cp.Parameter(nonneg=True, value = self.regulizer)
                
				prob = cp.Problem(cp.Minimize(loss/len(self.selected_train_data) + c*slack))
				prob.solve()
				#get params
				self.w = Weights.value.flatten()
				# print(w)
				margin = 1/np.linalg.norm(self.w)
				#get the gab between the hyperplane and the origin
				self.intercept = gamma.value
				#get the margin of the hyperplane
				
				# print(margin)
				# print('w:', w, 'intercept:', intercept, 'margin:', margin)
				self.b = [float(self.intercept - 0*margin), float(self.intercept + 1*margin)]
				self.b_pos = [float(self.intercept + 0*margin), float(self.intercept + 1*margin)]
				self.b_neg = [float(self.intercept - 1*margin), float(self.intercept - 0*margin)]
				# print('b_pos:', b_pos, 'b_neg:', b_neg)
				#score the testing data
				
				self.selected_train_data = list(self.selected_train_data)
				self.selected_train_labels = list(self.selected_train_labels)

	def f(self, input):
		if input.dot(self.w) + self.intercept > 0:
			return 1
		else:
			return -1

	def get_score(self, test_data, test_labels):
			correct = 0
			for i in range(len(test_data)):
				if self.f(test_data[i]) == test_labels[i]:
					correct += 1
			return correct/len(test_data)

	def test(self, test_data, test_labels): 
		self.score = self.get_score(test_data, test_labels)
		self.classifications = [1 if x > 0 else -1 for x in test_data @ self.w + self.intercept]
		return self.classifications

synthetic_data = rand2DGaussian()
synthetic_data.generate_data(12000)

train_data, train_labels, test_data, test_labels = split_data(synthetic_data.data, synthetic_data.label, 0.2)
#reshape train_labels
train_labels = np.reshape(train_labels, (len(train_labels),1))
test_labels = np.reshape(test_labels, (len(test_labels),1))

classifier = Classifier()
classifier.train(train_data, train_labels)
classifier.test(test_data, test_labels)
print('Size of Train Sets: ', train_data.shape[0])
print('Size of Selected Data: ', len(classifier.selected_train_data))
print("Score using", len(classifier.selected_train_data), "data points: ", classifier.score)
