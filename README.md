### ML_lib - A simple machine learning library

This simple machine learning library performs classification of land-cover of remotely sensed images. The data should be pre-formatted in GeoTiff format, with each band being separately stored as a separate image that is geo-referenced to all others. Additionally, labels for land-cover classes (for example from the National Land-Cover Dataset) should be treated the same, a separate geo-referenced image with the text 'label' in the file name.

The library has four modules:
* Data import and formatting
* Algorithms for classsification
* Evaluation of classifications
* Visualization of data (both labeled input and classified output)
Examples of the use of the library can be found in the python scripts in this repository:
* library_evaluation.py which performs model training and evaluation for all the available models in the library, as well as a demonstration of training and evaluating SciKit Learn classifiers. 
* KNN_evaluation.py which performs a training of a KNN model on some training data and then validates the model on separate data

The testing report pdf contains more detailed information about how the library performed in testing.

An example of validation data (a separate image from the training image) classified using EOLearn KNN, achieving ~63% accuracy, with the NLCD data on the left and the EOLearn KNN classification on the right:
![Comparison of NLCD (left) classification to EOLearn KNN](/assets/result_to_label_comparison.png "Comparison of NLCD (left) classification to EOLearn KNN")

&copy; Chris Burgoyne 2021