# This is the machine learning library used below and it contains functions for data formatting, classifying, and evaluation
# To read more about this library, simply use the EOLearn.help() function
import EOLearn as EOL

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def main():
    
    # Import data from input directory
    data_dir = r"training_data//"
    data = EOL.format_data.import_data(data_dir)

    # Retrieve the features and corresponding labels
    X = EOL.format_data.retrieve_features(data)
    # Combine classes smaller than the mean of class sizes
    X = EOL.format_data.combine_classes(X, seed=0)
    # Random resample to 10% of original dataset size
    X = EOL.format_data.resample(X, 0.1)

    # Plot data stats (save to file)
    EOL.visualise.plot_band_counts(X)
    EOL.visualise.plot_class_counts(X)
    EOL.visualise.plot_class_distr(X)

    # Get the classes in the data
    classes = EOL.evaluate.get_classes(X)
    # Number of folds to split data into when performing cross-fold validation
    num_folds = 5
    X_folds = EOL.evaluate.data_to_folds(X, num_folds)

    # Parameters
    # k: The number of neighbours to tally classes from when performing a prediction
    k = 7
    # Tree parameters - the same for Decision Tree and Random Forest
    # Maximum number of nodes in any path from the root
    max_depth     = 10
    # Minimum number of data instances per leaf
    min_leaf_size = 5
    # Random Forest Parameters
    # Size of bootstrapped sample as a percentage of training data
    sample_size = 1
    # Number of trees in the forest
    n_trees     = 50
    # Number of features to select from at each decision point in tree building
    n_features  = int((len(X[0])-1)/2)
    # Seed value for randomization
    seed_val    = 4

    print("SK-Learn")
    print(" KNN:")
    metrics = EOL.evaluate.cross_fold_sklearn(KNeighborsClassifier(n_neighbors=k), X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/SKL_KNN_metrics.csv","SKL K Nearest Neighbors")

    print(" Naive Bayes:")
    metrics = EOL.evaluate.cross_fold_sklearn(GaussianNB(), X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/SKL_NB_metrics.csv","SKL Naive Bayes")

    print(" Decision Tree:")
    metrics = EOL.evaluate.cross_fold_sklearn(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf_size), X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/SKL_DT_metrics.csv","SKL Decision Tree")
    
    print(" Random Forest:")
    metrics = EOL.evaluate.cross_fold_sklearn(RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_leaf_size, max_features=n_features, n_estimators=n_trees, random_state=seed_val), X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/SKL_RF_metrics.csv","SKL Random Forest")
    
    print(" Multi-Layer Perceptron:")
    metrics = EOL.evaluate.cross_fold_sklearn(MLPClassifier(), X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/SKL_MLP_metrics.csv","SKL Multi-Layer Perceptron")
    
    print(" Support Vector Machine:")
    metrics = EOL.evaluate.cross_fold_sklearn(SVC(class_weight='balanced'), X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/SKL_SVC_metrics.csv","SKL Multi-Layer Perceptron")
    
    print("EO-Learn")
    print(" KNN:")
    metrics = EOL.evaluate.cross_fold_validation(EOL.knn.classify, X_folds, num_folds, classes, k)
    EOL.evaluate.print_metrics_to_file(metrics,"results/EOL_KNN_metrics.csv","EOLearn KNN")
    
    print(" Naive Bayes:")
    metrics = EOL.evaluate.cross_fold_validation(EOL.naive_bayes.classify, X_folds, num_folds, classes)
    EOL.evaluate.print_metrics_to_file(metrics,"results/EOL_NB_metrics.csv","EOLearn Naive Bayes")
    
    print(" Decision Tree:")
    metrics = EOL.evaluate.cross_fold_validation(EOL.decision_tree.classify, X_folds, num_folds, classes, max_depth, min_leaf_size)
    EOL.evaluate.print_metrics_to_file(metrics,"results/EOL_DT_metrics.csv","EOLearn Decision Tree")

    print(" Random Forest:")
    metrics = EOL.evaluate.cross_fold_validation(EOL.random_forest.classify, X_folds, num_folds, classes, max_depth, min_leaf_size, sample_size, n_trees, n_features, seed_val)
    EOL.evaluate.print_metrics_to_file(metrics,"results/EOL_RF_metrics.csv","EOLearn Random Forest")
    
    
    return

if __name__ == "__main__":
    main()
