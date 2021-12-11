# This is the EO-Learn machine learning library used below and it contains functions for 
# data formatting, classifying, and evaluation.
# To read more about this library, simply use the EOLearn.help() function.
import EOLearn as EOL

def main():
    
    # Import NLCD 2019 Land Cover Classes
    LCC = EOL.format_data.land_cover_classes()

    # Seed Parameter for Randomization Repeatability
    seed = 0

    ## Import and Process Train/Test Data ## 
    data_dir = r"training_data//"
    train_data = EOL.format_data.import_data(data_dir)
    train_dimensions = (train_data["rows"],train_data["cols"])
    
    # Show labels
    EOL.visualise.show_NLCD(train_data["bands"][-1].flatten(), LCC, train_dimensions)

    # Retrieve the features and corresponding labels
    train_X = EOL.format_data.retrieve_features(train_data)
    train_X = EOL.format_data.combine_classes(train_X, seed, store_indexes=True)
    
    EOL.visualise.show_as_img(train_X[:,-1], train_dimensions, train_X[:,0], LCC, show=True)
    
    # Plot validation data stats (save to file)
    EOL.visualise.plot_band_counts(train_X,"train")
    EOL.visualise.barplot_classes(train_X,"train")
    EOL.visualise.plot_class_counts(train_X,"train")
    EOL.visualise.plot_class_distr(train_X,"train")


    ## Import and Process Validation Data ##
    data_dir = r"validation_data//"
    validation_data = EOL.format_data.import_data(data_dir)
    
    # Save dimensions of validation data for plotting
    validation_dimensions = (validation_data["rows"],validation_data["cols"])
    
    # Show NLCD 2019 labeling for validation data
    EOL.visualise.show_NLCD(validation_data["bands"][-1].flatten(), LCC, validation_dimensions)
    
    # Retrieve the features and corresponding labels
    validate_X = EOL.format_data.retrieve_features(validation_data)
    validate_X = EOL.format_data.combine_classes(validate_X, seed, store_indexes=True)
    
    # Retrieve classes in 
    val_X_classes = EOL.evaluate.get_classes(validate_X)
    
    # Plot validation data stats (save to file)
    EOL.visualise.plot_band_counts(validate_X,"valid")
    EOL.visualise.barplot_classes(validate_X,"valid")
    EOL.visualise.plot_class_counts(validate_X,"valid")
    EOL.visualise.plot_class_distr(validate_X,"valid")
    

    ## Implement EOLearn KNN ##
    print("EOL KNN:")
    # KNN Parameter K: The number of neighbours to tally classes from when performing a prediction
    k = 3
    
    # Perform K-Nearest Neighbours classification
    predictions = EOL.algorithms.KNN.classify(train_X, validate_X[:,1:], k)
    
    # Output the predictions
    print("Number of Predicated Values:",len(predictions))
    
    # Image
    EOL.visualise.show_as_img(predictions,validation_dimensions,validate_X[:,0],LCC,show=True,fname="assets/pred_knn3_corrected.png")
    
    # Calculate Error Metrics and Print to File
    conf_mat = EOL.evaluate.confusion_matrix(predictions, validate_X[:,-1], val_X_classes)
    error_met = EOL.evaluate.calculate_metrics(conf_mat)
    error_met["classes"] = val_X_classes
    error_met["labels"] = [[LCC[_class] for _class in val_X_classes]]
    error_met["confusion"] = conf_mat
    EOL.evaluate.print_metrics_to_file(error_met,"validation_knn.csv","EOL_KNN")
    
    return

if __name__ == "__main__":
    main()
