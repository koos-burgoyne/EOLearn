import numpy as np

def var_mean(X,classes,class_counts):
	num_classes = len(classes)
	num_features = len(X[0])-1
	variance = np.zeros((num_classes,num_features))
	mean = np.zeros((num_classes,num_features))
	# Iterate over all the data instances to calculate mean for each feature given a class
	for i in range(len(X)):
		for j in range(num_features):
			# Sum up values in prep for calculating mean
			mean[np.where(classes==X[i][len(X[0])-1])[0][0]][j] += X[i][j]
	for i in range(num_classes):
		for j in range(num_features):
			mean[i][j] /= class_counts[i]
	# Iterate over all the data instances to calculate variance for each feature given a class
	for i in range(len(X)):
		for j in range(num_features):
			# Sum up values in prep for calculating variance
			class_idx = np.where(classes==X[i][len(X[0])-1])[0][0]
			variance[class_idx][j] += (X[i][j] - mean[class_idx][j])**2
	# Divide all variance sums (over classes and features) by n-1
	for i in range(num_classes):
		for j in range(num_features):
			if class_counts[i] > 1:
				variance[i][j] /= (class_counts[i]-1)
			else:
				variance[i][j] /= 1
	return variance,mean

def calc_class_likelihoods(instance,v,m,classes):
	num_classes = len(classes)
	num_features = len(instance)-1
	feature_probs = np.zeros((num_classes,num_features))
	for i in range(num_features):
		for j in range(num_classes):
			# Calculate Gaussian likelihood of class membership
			if v[j][i] == 0:
				feature_probs[j][i] = (1/(np.sqrt(2*np.pi*1)))*np.exp((-(instance[i]-m[j][i])**2)/(2*1))
			else:
				feature_probs[j][i] = (1/(np.sqrt(2*np.pi*v[j][i])))*np.exp((-(instance[i]-m[j][i])**2)/(2*v[j][i]))
	probs_by_class = np.zeros(num_classes)
	for i in range(num_classes):
		probs_by_class[i] = np.prod(feature_probs[i])
	class_probs = np.zeros(num_classes)
	for i in range(num_classes):
		class_probs[i] = probs_by_class[i]/np.sum(probs_by_class)
	return class_probs

def prior_probs(X,classes,class_counts):
	n = len(X)
	probs = {}
	for i in range(len(classes)):
		probs[classes[i]] = class_counts[i]/n
	return probs

def classify(train,test):
	# Get number of classes and frequency of each
	classes,class_counts = np.unique(np.array(train)[:,len(train[0])-1], return_counts=True)
	# Calculate prior probability for each class
	priors = prior_probs(train,classes,class_counts)
	# Calculate class distribution parameters
	variance,mean = var_mean(train,classes,class_counts)
	predictions = []
	# Assign a class to each test instance
	for instance in test:
		# Calculate likelihoods, and multiply by prior
		class_likelihoods = calc_class_likelihoods(instance,variance,mean,classes)
		for i in range(len(classes)):
			class_likelihoods[i] *= priors[classes[i]]
		# Calculate class probabilities by normalizing with all class likelihoods
		class_probs = np.zeros(len(classes))
		for i in range(len(classes)):
			class_probs[i] = class_likelihoods[i]/sum(class_likelihoods)
		# Predict the most probable class
		predictions.append(classes[np.where(class_probs==max(class_probs))[0][0]])
	return predictions

"""a = np.array([[1,1,0],[2,2,0],[3,4,0],[12,12,1],[3,3,0],[6,7,3],[7,8,3]])
print(classify(a,a))"""