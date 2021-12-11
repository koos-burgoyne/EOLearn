import numpy as np
import time

# Calculate error metrics for predicted data. Returns a dictionary containing error metrics.
def calculate_metrics(conf_matrix):
    # Total number of instances represented in the matrix
    n = conf_matrix.sum()
    # Dictionary storage for error metrics
    metrics = {}
    # False Positives
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    # False Negatives
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    # True Positives
    TP = np.diag(conf_matrix)
    # True Negatives
    TN = list()
    for i in range(len(TP)):
        TN.append(TP.sum() - conf_matrix[i][i])
    TN = np.array(TN)
    # Convert to float
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Store values
    metrics["FP"] = FP
    metrics["FN"] = FN
    metrics["TP"] = TP
    metrics["TN"] = TN
    # Calculate and store metrics
    # This next piece avoid division by zero
    TP_FN = TP+FN
    TPR = []
    FNR = []
    precision = []
    for i in range(len(TP)):
        if TP_FN[i] != 0:
            TPR.append(TP[i]/TP_FN[i])
            FNR.append(FN[i]/TP_FN[i])
        else:
            TPR.append(0.)
            FNR.append(0.)
    # Sensitivity, recall, or true positive rate
    metrics["TPR"] = np.array(TPR)
    # Specificity or true negative rate
    metrics["TNR"] = TN/(TN+FP)
    # False positive rate
    metrics["FPR"] = FP/(FP+TN)
    # False negative rate
    metrics["FNR"] = np.array(FNR)
    # Cohen's Kappa
    P_ch = 0
    for i in range(len(conf_matrix)):
      next_val = (conf_matrix[i].sum()/n)*(conf_matrix[:,i].sum()/n)
      if next_val != 0:
          P_ch += next_val
    metrics["Kappa"] = ((TP.sum()/n) - P_ch)/(1-P_ch)
    # Matthews Correlation Coefficient
    TP = TP.sum()
    TN = TN.sum()
    FP = FP.sum()
    FN = FN.sum()
    metrics["MCC"] = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    # Overall accuracy
    metrics["acc"] = TP/n
    return metrics

# Print error metrics fromm a prediction to file
def print_metrics_to_file(metrics,fname,classifier):
    outfile = open(fname,"w")
    # Write the name of the classifier
    outfile.write(classifier+"\n")
    # Start with the classes as the first row in the csv
    line = "classes,"
    for class_val in range(len(metrics["classes"])):
        line += str(metrics["classes"][class_val])+","
    outfile.write(line+"\n")
    # Then class labels, if there
    if "labels" in metrics:
        line = "labels,"
        for lbl in metrics["labels"]:
            line += str(lbl)+","
        outfile.write(line+"\n")
    # Now write the rest of the metrics
    for key,val in metrics.items():
        # Classes have already been written, so pass 
        if key=="classes" or key=="labels":
            continue
        # Special case: writing the confusion matrix
        if key=="confusion":
            outfile.write("confusion matrix\n")
            for i in range(len(val)):
                line = ","
                for j in range(len(val[i])):
                    line += str(val[i][j])+","
                outfile.write(line+"\n")
            continue
        # Regular case
        values = str(key)+","
        # If the val is an array, iterate over the array values and build a string
        if "array" in str(type(val)):
            for i in range(len(val)):
                values += str(round(val[i],2))+","
        # If not an array, just include the value
        else:
            values += str(round(val,3))
        # Write the string to file
        outfile.write(str(values)+"\n")
    # Close the file
    outfile.close()
    return
 
# Calculate accuracy percentage based on actual and predicted values
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / len(actual)

# Calculate a confusion matrix based on the observed values, predicted values, and the classes required
def confusion_matrix(obs,pred,classes):
    n = len(classes)
    conf_mat = np.zeros((n,n))
    for i in range(len(obs)):
        obs_idx = np.where(classes==obs[i])[0][0]
        pred_idx = np.where(classes==pred[i])[0][0]
        conf_mat[pred_idx][obs_idx] += 1
    return conf_mat

# Return the classes of a dataset
def get_classes(X):
    return np.array(list(set(X[:,-1])))

# Split a dataset into k folds
def data_to_folds(dataset, n_folds):
    dataset_split = list()
    fold_size = int(len(dataset) / n_folds)
    start = 0
    end = fold_size
    for i in range(n_folds):
        dataset_split.append(dataset[start:end])
        start += fold_size
        end += fold_size
        if i==0:
            start += 1
    return np.array(dataset_split, dtype=object)

# Evaluate an algorithm using a cross validation split
def cross_fold_validation(algorithm, dataset, num_folds, classes, *args):
    timer = 0
    cm = None
    count = 0
    # Iterate over folds
    for fold in dataset:
        print("  Fold",count+1,"of",num_folds, end="\r")
        # Assemble train set
        train = list()
        for _fold in range(len(dataset)):
            if _fold != count:
                for row in dataset[_fold]:
                    train.append(row)
        # Assemble test set
        test = list()
        for row in fold:
            row_copy = list(row)
            test.append(row_copy)
        # Convert to Numpy array
        train = np.array(train)
        test = np.array(test)
        # Classify using the selected classsifier
        start = time.time()
        predicted = algorithm(train, test, *args)
        end = time.time()
        timer += end-start
        # Get observed class values
        observed = fold[:,-1]
        # Calculate confusion matrix, summed over all folds
        if count == 0:
            cm = confusion_matrix(observed,predicted,classes)
        else:
            cm += confusion_matrix(observed,predicted,classes)
        count += 1
    # Print time and average accuracy
    error_metrics = calculate_metrics(cm)
    error_metrics["classes"] = classes
    error_metrics["confusion"] = cm
    error_metrics["runtime"] = timer/num_folds
    print("    "+str(round(timer/num_folds,2))+" sec average runtime")
    print("    "+str(round(error_metrics["acc"]*100,3))+"%\n")
    return error_metrics

# TODO
def cross_fold_sklearn(_classifier, dataset, num_folds, classes):
    timer = 0
    cm = None
    count = 0
    # Iterate over folds
    for fold in dataset:
        print("  Fold",count+1,"of",num_folds, end="\r")
        # Assemble train set
        train = list()
        for _fold in range(len(dataset)):
            if _fold != count:
                for row in dataset[_fold]:
                    train.append(row)
        # Assemble test set
        test = list()
        for row in fold:
            row_copy = list(row)
            test.append(row_copy)
        # Convert to Numpy array
        train = np.array(train)
        test = np.array(test)
        # Classify using the selected classsifier
        start = time.time()
        classifier = _classifier
        classifier.fit(train[:,:-1],train[:,-1])
        predicted = classifier.predict(test[:,:-1])
        end = time.time()
        timer += end-start
        # Get observed class values
        observed = fold[:,-1]
        # Calculate confusion matrix, summed over all folds
        if count == 0:
            cm = confusion_matrix(observed,predicted,classes)
        else:
            cm += confusion_matrix(observed,predicted,classes)
        count += 1
    # Print time and average accuracy
    error_metrics = calculate_metrics(cm)
    error_metrics["classes"] = classes
    error_metrics["confusion"] = cm
    error_metrics["runtime"] = timer/num_folds
    print("    "+str(round(timer/num_folds,2))+" sec average runtime")
    print("    "+str(round( error_metrics["acc"] *100,3))+"%")
    return error_metrics