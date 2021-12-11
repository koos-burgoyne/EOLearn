# Data import and formatting
import EOLearn.format_data as format_data

# Data visualisation
import EOLearn.visualise as visualise

# Algorithms
import EOLearn.algorithms.knn as knn
import EOLearn.algorithms.random_forest as random_forest
import EOLearn.algorithms.decision_tree as decision_tree
import EOLearn.algorithms.naive_bayes as naive_bayes

# Evaluation of algorithms and their results
import EOLearn.evaluate as evaluate

def help():
    print("\n*** Welcome to this simple library of machine learning functions ***\n")
    print("This library support three broad types of operation:")
    print("    Data Import and Formatting")
    print("    Data Classification")
    print("    Evaluation of Classifications")
    print()
    print("To use data import and formatting tools:")
    print("    EOLearn.format_data.<function>")
    print("    Available functions:")
    print("         import_data(data_dir)")
    print("         retrieve_features(data)")
    print("         retrieve_features_labels(data)")
    print("         train_test_split(X,num_train,num_test)")
    print("         train_test_split_separate(X,num_train,num_test)")
    print()
    print("To use classification tools:")
    print("    There are two methods for classifying using the tools in this library:")
    print("         Using the classification functions to call an algorithm on train and test data directly")
    print("             EOLearn.<algorithm>.classify(train,test,<args>)")
    print("                For choice of algorithms and their arguments, see below")
    print("         Using the evaluate library to perform cross-fold validation with an algorithm on a dataset")
    print("             EOLearn.evaluate.cross_fold_validation(<algorithm>,<data>,num_folds,<args>)")
    print("    Available functions:")
    print("         EOLearn.knn.classify(train,test,k)")
    print("         EOLearn.decision_tree.classify(train,test,max_depth,min_leaf_size)")
    print("         EOLearn.random_forest.classify(train,test,max_depth,min_leaf_size,sample_size,n_trees,n_features,seed_val)")
    print("         EOLearn.naive_bayes.classify(train,test)")
    print()
    print("***                      End of Help                             ***\n")
    return