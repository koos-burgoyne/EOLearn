import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import mode

# Function : Calculate the KNN
# Arguments: Training data (x and y), testing data, and value k for how many neighbours to include
# Returns  : List of predicted class values
def classify(train, test, k):
    print("KNN prediction on",k,"neighbours")
    print("  Training set size:", np.shape(train))
    print("  Testing set size:", np.shape(test))
    # Build a KDTree from sklearn
    print("  Building Tree...", end="\r")
    tree = KDTree(train, leaf_size=2)   
    
    # Compute the KNN
    print("  Querying Tree...     ", end="\r")
    dist, ind = tree.query(test, k=k)
    
    # Storage for predicted labels
    y_pred = []
    # Get the mode class of the KNN
    print("  Retrieving predictions...   ")
    for row in range(len(ind)):
        if (row % int(len(ind)/100)) == 0:
            print(" ",row / int(len(ind)/100),"% of ",len(ind)," predictions",end="\r")
        y_pred.append(mode(train[ind[row]][:,-1])[0][0])
    
    print("  KNN Complete                                   \n")
    return np.array(y_pred)