# LRDT
Logistic Regressión classifier with binary variables from random trees

This module adds the capability to include non-linear interactions between variables to a logistic regression classifier.

By calling fit_tree(X,y) the algorithm runs many random initialized decission trees and extract rules from each of them.
The it select the n-best rules and add binary features to the training datatset.

The by the transform method a new train dataset is obtained with all the new binary features to improve the performance.

This usually improves significatively the performace of the simple Logistic Regresion, and gives more accurate prediction than both single Decission Trees and single Logistic Regeressión, while also gives more interpretavility to the obtained model.


