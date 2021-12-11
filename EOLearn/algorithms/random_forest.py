# Use the decision tree algorithm
import EOLearn.algorithms.decision_tree as DT
import random

# Return a randomly resampled subsample
def resample(dataset, proportion):
	sample = []
	sample_size = round(len(dataset) * proportion)
	while len(sample) < sample_size:
		idx = random.randrange(len(dataset))
		sample.append(dataset[idx])
	return sample

# Make predictions on an instance from esemble of trees
def bag_pred(trees, instance):
	predictions = [DT.predict_DT(tree, instance) for tree in trees]
	return max(set(predictions), key=predictions.count)
 
# Random Forest Algorithm
def classify(train, test, max_depth, min_nodes, resample_proportion, num_trees, num_features, seed_val):
	#print("Performing Random Forest classification...")
	#print("  Params:")
	#print("    Num Trees: " + str(num_trees))
	#print("    Max Depth: " + str(max_depth))
	#print("    Num Feat : " + str(num_features))
	#print("    Min Size : " + str(min_size))
	#print("    Seed Val : " + str(seed_val))
	random.seed(seed_val)
	trees = [None for i in range(num_trees)]
	for i in range(num_trees):
		#print("  Building tree "+str(i+1)+" of "+str(num_trees))
		sample = resample(train, resample_proportion)
		tree = DT.build_DT(sample, max_depth, min_nodes, num_features)
		trees[i] = tree
	#print("  Making Predictions...")
	predictions = [bag_pred(trees, instance) for instance in test]
	#print("Random Forest classification complete\n")
	return predictions