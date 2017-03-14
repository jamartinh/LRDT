# LRDT

Logistic Regression classifier with binary variables from random trees.

This module adds the capability to include *non-linear* interactions between variables to a Logistic Regression classifier.

By calling the fit_tree(X,y) method the algorithm runs many random initialized decision trees and extract rules from each of them.
Then it select the n-best rules and adds binary features to the training datatset.

Next, by calling the transform(X) method, a new train dataset is obtained with all the new binary features added in order to improve the predictive power.

This usually improves significantly the performance of the simple Logistic Regression, and gives more accurate prediction than both single Decision Trees and single Logistic Regression, while also gives more interpretavility to the obtained model.

Some experiments show comparable performace with the XGBoost algorithm.


