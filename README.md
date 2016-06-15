# LRDT
Logistic Regrssión classifier with binary variables from random trees

This module add the capability to include non-linear interactions between variables to a logistic regression classifier.

By calling fit_tree(X,y) the algorithm runs many random initialized decission trees and extract rules from each of them.
The it select the n best rules and ads binary features to the training datatset.

This usually improves a significantly the performace of the simple Logistic regresion, and gives more accurate prediction than both, single Decission Trees and single Logistic Regeressión.
