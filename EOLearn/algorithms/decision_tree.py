import numpy as np

# Calculate the value of a leaf
def calculate_leaf(X):
	values = [instance[-1] for instance in X]
	return max(set(values), key=values.count)

# Split a dataset based on an attribute and an attribute value
def attribute_based_split(attribute, value_to_split_by, X):
	L, R = [], []
	for instance in X:
		if instance [attribute] < value_to_split_by:
			L.append(instance )
		else:
			R.append(instance )
	return L, R

# Calculate the Gini index for a split dataset
def calculate_gini(split_data, classes):
	# count all samples in split data
	n_instances = float(sum([len(attr_based_group) for attr_based_group in split_data]))
	# Calculate weighted Gini index for each split
	gini = 0.0
	for attr_based_group in split_data:
		size = float(len(attr_based_group))
		# don't divide by 0
		if size == 0:
			continue
		G_ceof = 0.0
		# score based on each class coefficient
		for _class in classes:
			val = [instance[-1] for instance in attr_based_group].count(_class) / size
			G_ceof += val * val
		# weight the coefficient by its relative size to the size of all combined instances in the split data
		gini += (1.0 - G_ceof) * (size / n_instances)
	return gini
 
# Select the best split point for a dataset
def choose_best_split(X):
	class_values = list(set(np.array(X)[:,-1]))
	b_index, b_value, _gini_coeff, b_groups = 999, 999, 999, None
	for index in range(len(X[0])-1):
		for instance in X:
			groups = attribute_based_split(index, instance[index], X)
			gini = calculate_gini(groups, class_values)
			if gini < _gini_coeff:
				b_index, b_value, _gini_coeff, b_groups = index, instance[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create child splits for a node or make leaf
def split_node(node, max_depth, min_size, depth):
	L, R = node['groups']
	del(node['groups'])
	# calculate a leaf
	if not L or not R:
		node['L'] = node['R'] = calculate_leaf(L + R)
		return
	# if maximum depth has been reached
	if depth >= max_depth:
		node['L'], node['R'] = calculate_leaf(L), calculate_leaf(R)
		return
	# if L child is too small, calculate a leaf
	if len(L) <= min_size:
		node['L'] = calculate_leaf(L)
	# else choose the best split to continue
	else:
		node['L'] = choose_best_split(L)
		split_node(node['L'], max_depth, min_size, depth+1)
	# if L child is too small, calculate a leaf
	if len(R) <= min_size:
		node['R'] = calculate_leaf(R)
	# else choose the best split to continue
	else:
		node['R'] = choose_best_split(R)
		split_node(node['R'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_DT(X, max_depth, min_size):
	root = choose_best_split(X)
	split_node(root, max_depth, min_size, 1)
	return root
 
# Predict using a DT and an instance of data
def predict_DT(tree, instance):
	if instance[tree['index']] < tree['value']:
		if isinstance(tree['L'], dict):
			return predict_DT(tree['L'], instance)
		else:
			return tree['L']
	else:
		if isinstance(tree['R'], dict):
			return predict_DT(tree['R'], instance)
		else:
			return tree['R']
 
# Classification and Regression Tree Algorithm
def classify(train, test, max_depth, min_size):
	#print("Performing Decision Tree classification")
	#print("  Params:")
	#print("    Max Depth: " + str(max_depth))
	#print("    Min Size : " + str(min_size))
	#print("  Building tree...")
	tree = build_DT(train, max_depth, min_size)
	predictions = []
	#print("  Predicting using tree...")
	for i in range(len(test)):
		#if (i % (len(test)/10)) == 0:
		#	print("\t",int(i/len(test)*100),"% complete")
		prediction = predict_DT(tree, test[i])
		predictions.append(prediction)
	#print("Decision Tree classification complete\n")
	return predictions