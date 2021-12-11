import matplotlib.pyplot as plt
import numpy as np
import os

def import_data(data_dir):
    # Check for directory name
    if data_dir=="":
        print("Error: No data directory entered")
        raise SystemExit()
    bands = []
    labels = []
    # Loop through files
    print("Importing GeoTiff data from " + data_dir[:-1])
    for filename in os.listdir(data_dir):
        # Process if .tif format
        if filename.endswith(".tif"):
            # Retrieve the labels for the data
            if "labels" in filename:
                # Exception handling for image reading
                try:
                    lbls = plt.imread(data_dir[:-1] + filename)
                except:
                    print("Error: Could not open file")
                    print("     : "+data_dir[:-1]+filename)
                    raise SystemExit()
                # Append labels to the list
                labels.append(lbls)
            print("\t"+str(data_dir[:-1] + filename))
            # Retrieve the band information from the data
            # Exception handling for image reading
            try:
                band_data = plt.imread(data_dir[:-1] + filename)
            except:
                    print("Error: Could not open file")
                    print("     : "+data_dir[:-1]+filename)
                    raise SystemExit()
            # Append band data from file to the bands list
            bands.append(band_data)
    # Get image dimensions
    rows = len(bands[0])
    cols = len(bands[0][0])
    print(" Image Dimensions:",str(rows)+", "+str(cols))
    print(" Data Imported\n")
    return {"bands":np.array(bands), "labels":np.array(np.reshape(labels,-1)), "rows":rows, "cols":cols, "n":rows*cols}

# Convert a set of images to an array of flattened images so each column is an image
def retrieve_features(data):
    print("Retrieving features from input data")
    # Number of bands is number of images
    num_bands = len(data["bands"])-1
    # Storage to return
    X = [[] for i in range(len(data["bands"]))]
    # Iterate over bands, flatten and store them
    for i in range(num_bands):
        print("  Retrieving Band",i,end="\r")
        X[i] = data["bands"][i].flatten()
    # Add the labels to the image as the last column
    X[-1] = data["labels"].flatten()
    print("  Complete                 \n")
    return np.transpose(np.array(X))

# Same as the function retrieve_features above, but this one returns the labels separately for SKLearn classifiers
def retrieve_features_labels(data):
    print("Retrieving features from input data\n")
    num_bands = len(data["bands"])-1
    X = [[] for i in range(len(data["bands"])-1)]
    for i in range(num_bands):
        X[i] = data["bands"][i].flatten()
    y = data["labels"].flatten()
    return np.transpose(np.array(X)), y

# Split data into train and test sets based on the number of instances provided
def train_test_split(X, num_train, num_test):
    print("\nSplit training and testing data")
    print("\t"+str(num_train)+" training instances")
    print("\t"+str(num_test)+" testing instances\n")
    # Assign training Data
    train_idx = np.random.randint(0, len(X), num_train)
    # Assign testing Data
    test_idx = np.random.randint(0, len(X), num_test)
    # Return new data
    return X[train_idx], X[test_idx]

# Split data into train and test sets based on numbers provided, but this time the data labels are in separate storage
def train_test_split_separate(X, y, num_train, num_test):
    print("\nSplit training and testing data")
    print("\t"+str(num_train)+" training instances")
    print("\t"+str(num_test)+" testing instances")
    print()
    # Assign training Data
    train_idx = np.random.randint(0, len(X), num_train)
    # Assign testing Data
    test_idx = np.random.randint(0, len(X), num_test)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# Randomly resample a dataset to the percentage input
# Can specify if result should have balanced class counts
def resample(X, percent, balanced=False, seed=None):
    # Assign seed value for randomization
    if seed!=None:
        np.random.seed(seed)
    print("Resampling:")
    print("  Original size:",len(X))
    # Calculate new sample size
    n = int(len(X)*percent)
    print("  Sample size:",n)
    if not balanced:
        # Get random indexes
        test_idx = np.random.randint(0, len(X), n)
        sample = X[test_idx]
    else:
        # Get classes and class counts from the class/label column
        print("  Balancing: Splitting Data by Class...",end="\r")
        classes, class_counts = np.unique(X[:,-1], return_counts=True)
        num_classes= len(classes)
        class_size = int(n/num_classes)
        # Storage for the indexes of dataset columns of each class
        class_idxs = [[] for i in range(num_classes)]
        # Get the indexes of pixels in each class
        for col in range(len(X)):
            class_idx = np.where(classes==X[col][-1])[0][0]
            class_idxs[class_idx].append(col)
        # Convert to numpy arrays
        for i in range(num_classes):
            class_idxs[i] = np.array(class_idxs[i], dtype=int)
        # Split the data by class
        data_by_class = [X[class_idxs[i]] for i in range(1,num_classes)]
        print("  Balancing: Creating new sample...   ",end="\r")
        sample = list()
        for i in range(len(data_by_class)):
            for j in range(class_size):
                random_idx = np.random.randint(0,len(data_by_class[i]))
                sample.append(data_by_class[i][random_idx])
        # Convert to numpy array
        sample = np.array(sample)
    print("  Resample Complete                   \n")
    # Return the new sample
    return sample

# Remove data instances where the class value is zero. This is helpful where RS images are clipped and the non-value pixels are zero
def strip_zeros(X):
    print("Removing Instances with Class Zero from Data\n")
    n = len(X)
    m = len(X[0])
    # Remove zeros from the data
    zero_idx = list()
    for i in range(n):
        if X[i,-1]!=0:
            zero_idx.append(i)
    zero_idx = np.array(zero_idx)
    X = X[zero_idx]
    return X

# Resample a dataset so that all classes are increased to the size of the largest class
def upsample(X, seed=None):
    if seed!=None:
        np.random.seed(seed)
    # Convert to numpy array
    X = np.array(X)
    # Number of instance
    n = len(X)
    # Number of features including labels
    m = len(X[0])
    # Get classes and class counts from the class/label column
    classes, class_counts = np.unique(X[:,-1], return_counts=True)
    num_classes= len(classes)
    # Storage for the indexes of dataset columns of each class
    class_idxs = [[] for i in range(num_classes)]
    # Get the indexes of pixels in each class
    for col in range(n):
        class_idx = np.where(classes==X[col][-1])[0][0]
        class_idxs[class_idx].append(col)
    # Convert to numpy arrays
    for i in range(num_classes):
        class_idxs[i] = np.array(class_idxs[i])
    # Split the data by class
    data_by_class = [X[class_idxs[i]] for i in range(num_classes)]
    # Get the index of the largest class from the data_by_class array, and then 
    # use that to index to the class and get it's length
    size_largest_class = np.max(class_counts)
    print("Resampling all classes to size ", size_largest_class)
    #size_largest_class = len(data_by_class[np.argmax(data_by_class)])
    data = []
    for i in range(num_classes):
        print("  Upsampling class",i,"of",num_classes,end="\r")
        for j in range(size_largest_class):
            random_idx = np.random.randint(0,class_counts[i])
            data.append( data_by_class[i][random_idx] )
    print("  Upsampling Complete\n")
    return np.array(data)

# Resample a dataset so that all classes are increased to the size of the smallest class
def downsample(X, seed=None):
    if seed!=None:
        np.random.seed(seed)
    # Convert to numpy array
    X = np.array(X)
    # Number of instance
    n = len(X)
    # Number of features including labels
    m = len(X[0])
    # Get classes and class counts from the class/label column
    classes, class_counts = np.unique(X[:,-1], return_counts=True)
    num_classes= len(classes)
    # Storage for the indexes of dataset columns of each class
    class_idxs = [[] for i in range(num_classes)]
    # Get the indexes of pixels in each class
    for col in range(n):
        class_idx = np.where(classes==X[col][-1])[0][0]
        class_idxs[class_idx].append(col)
    # Convert to numpy arrays
    for i in range(num_classes):
        class_idxs[i] = np.array(class_idxs[i])
    # Split the data by class
    data_by_class = [X[class_idxs[i]] for i in range(num_classes)]
    # Get the index of the largest class from the data_by_class array, and then 
    # use that to index to the class and get it's length
    size_smallest_class = np.min(class_counts)
    print("Resampling all classes to size ", size_smallest_class)
    #size_smallest_class = len(data_by_class[np.argmax(data_by_class)])
    data = []
    for i in range(num_classes):
        print("  Downsampling class",i,"of",num_classes,end="\r")
        for j in range(size_smallest_class):
            random_idx = np.random.randint(0,class_counts[i])
            data.append( data_by_class[i][random_idx] )
    print("  Downsampling Complete\n")
    return np.array(data)

# Resample a dataset so that all classes smaller than the mean class size are combined
# Alos, if wanting to display the predictions as an image it is necesary to store indexes 
# for plotting so that the plotting knows which pixel index to put the predicted values 
# in after having flattened the images for processing.
def combine_classes(X, seed=None, store_indexes=False):
    print("Combining Classes:")
    if seed!=None:
        np.random.seed(seed)
    # Store Indexes for plotting
    if store_indexes:
        print("  Storing Indexes...",end="\r")
        new_X = np.zeros((len(X),len(X[0])+1))
        for i in range(len(X)):
            new_X[i][0] = i
            for j in range(1,len(X[0])+1):
                new_X[i][j] = X[i][j-1]
        X = new_X
        #print(np.shape(X), np.shape(new_X))
    print("  Separating by class...",end="\r")
    # Number of instance
    n = len(X)
    # Number of features including labels
    m = len(X[0])
    # Get classes and class counts from the class/label column
    classes, class_counts = np.unique(X[:,-1], return_counts=True)
    num_classes= len(classes)
    # Storage for the indexes of dataset columns of each class
    class_idxs = [[] for i in range(num_classes)]
    # Get the indexes of pixels in each class
    for col in range(n):
        class_idx = np.where(classes==X[col][-1])[0][0]
        class_idxs[class_idx].append(col)
    # Convert to numpy arrays
    for i in range(num_classes):
        class_idxs[i] = np.array(class_idxs[i])
    # Split the data by class
    data_by_class = [X[class_idxs[i]] for i in range(1,num_classes)]
    # Combine classes
    print("  Combining by class...   ",end="\r")
    mean_class_size = 0
    for i in range(len(data_by_class)):
        mean_class_size += len(data_by_class[i])
    mean_class_size /= len(data_by_class)
    too_small = list()
    big_enough = list()
    for i in range(len(data_by_class)):
        if len(data_by_class[i]) > mean_class_size:
            big_enough.append(i)
        else:
            too_small.append(i)
    new_X = []
    # Add classes that were big enough and don't need to be combined
    for idx in big_enough:
        for j in range(len(data_by_class[idx])):
            new_X.append(data_by_class[idx][j])
    counter = 0
    for idx in too_small:
        for j in range(len(data_by_class[idx])):
            if counter < mean_class_size:
                data_by_class[idx][j][-1] = counter
                new_X.append(data_by_class[idx][j])
            else:
                counter += 1
                data_by_class[idx][j][-1] = counter
                new_X.append(data_by_class[idx][j])
    print("  Old Data                  ")
    print("   ",num_classes,classes)
    print("   ",class_counts)
    print("   Size:",np.shape(X))
    classes, class_counts = np.unique(np.array(new_X)[:,-1], return_counts=True)
    num_classes= len(classes)
    print("  New Data")
    print("   ",num_classes,classes)
    print("   ",class_counts)
    print("   Size:",np.shape(np.array(new_X)))
    print()
    return np.array(new_X)

# The land cover class labels by value
def land_cover_classes():
    classes = {
        0:"Combination",
        11:"Open Water",
        12:"Snow/Ice",
        21:"Developed, Open",
        22:"Developed, Lo.Int.",
        23:"Developed, Med.Int.",
        24:"Developed, Hi.Int.",
        31:"Barren Land",
        41:"Deciduous Forest",
        42:"Evergreen Forest",
        43:"Mixed Forest",
        51:"Dwarf Scrub",
        52:"Shrub/Scrub",
        71:"Grassland",
        72:"Sedge",
        73:"Lichen",
        74:"Moss",
        81:"Pasture/Hay",
        82:"Cultivate Crops",
        90:"Woody Wetlands",
        95:"Herbaceous Wetlands",
    }
    return classes