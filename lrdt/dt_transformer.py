__author__ = "Jose Antonio Martin H."

import random
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
from sklearn.utils import check_random_state, compute_sample_weight

__all__ = ['DTRTransformer', 'get_rules_of_decision_tree']


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


# a rule is a dict whose keys are feature names and values  min and max
#


def simplify_rule(rule):
    if len(rule) == 1:
        feature_name, sign, value = rule[0].split()
        return feature_name + " " + sign + " " + value

    features = OrderedDict()
    feature_rules = OrderedDict()
    none_num = 9999999999
    for part in rule:
        feature_name, sign, value = part.split()
        value = float(value)
        if feature_name not in features:
            features[feature_name] = [value, none_num] if sign == '>' else [-none_num, value]
        elif sign == '>':
            features[feature_name][0] = max(value, features[feature_name][0])
        else:
            features[feature_name][1] = min(value, features[feature_name][1])

        feature_rules[feature_name] = ''
        if features[feature_name][0] != -none_num:
            feature_rules[feature_name] += str(features[feature_name][0]) + ' <= '

        feature_rules[feature_name] += feature_name

        if features[feature_name][1] != none_num:
            feature_rules[feature_name] += ' <= ' + str(features[feature_name][1])

    return " and ".join(feature_rules.values())


def get_rules_of_decision_tree(dt, feature_names=None, percent_threshold=0.1, proportion_threshold=0.8, min_depth=1):
    """

    :param dt: fitted decission tree
    :param feature_names: a list containing  optional features names
    :param percent_threshold: the minimum number of support percent for a rule to be included
    :param proportion_threshold: the minimum accuracy on a class to include this rule
    :param min_depth: the minimum length of a rule.
    :return: a list of rules dict and a set of rules
    """

    rules = list()
    rule_set = set()
    left = dt.tree_.children_left
    right = dt.tree_.children_right
    threshold = dt.tree_.threshold
    value = dt.tree_.value

    if feature_names is None:
        features = ['f%d' % i for i in dt.tree_.feature]
    else:
        features = [feature_names[i] for i in dt.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0, parent='', sign='', path=[]):

        node_val = value[node]
        percent = (dt.tree_.n_node_samples[node] /
                   float(dt.tree_.n_node_samples[0]))
        propotions = node_val / dt.tree_.weighted_n_node_samples[node]
        power = max(propotions[0]) * percent

        new_rule = {
            'rule': path,
            'support': value[node][0],
            'percent': percent,
            'proportions': propotions[0],
            'class': np.argmax(propotions[0]),
            'probability': propotions[0][np.argmax(propotions[0])],
            'power': power,
        }

        rule = ''
        if node > 0:
            rule = str(features[parent]) + sign + str(round(threshold[parent], 1))

        if percent >= percent_threshold and max(propotions[0]) > proportion_threshold and node > 0 and len(path) >= min_depth and rule != '':
            new_rule['rule'].append(rule)
            new_rule['features'] = [x.split()[0] for x in new_rule['rule']]
            if len(set(new_rule['features'])) > 1:
                new_rule['rules'] = list(new_rule['rule'])
                new_rule['rule'] = " and ".join(new_rule['rule'])
            else:
                new_rule['rules'] = [rule]
                new_rule['rule'] = rule

            rules.append(new_rule)
            rule_set.add(new_rule['rule'])
            return

        if threshold[node] != -2:
            if left[node] != -1:
                recurse(left,
                        right,
                        threshold,
                        features,
                        left[node],
                        depth + 1, node,
                        ' <= ',
                        list(path + [rule]) if rule else list(path))
            if right[node] != -1:
                recurse(left,
                        right,
                        threshold,
                        features,
                        right[node],
                        depth + 1,
                        node,
                        ' > ',
                        list(path + [rule]) if rule else list(path))

    recurse(left, right, threshold, features, 0, 0, None, '', [])

    return rules, rule_set


class DTRTransformer(BaseEstimator, TransformerMixin):
    """Transforms an input features vector X into a X_new vector with semi-random decision tree rules added as new features to use them for classification or regression.

    Parameters
    ----------
    percent_threshold: the minimum percentage of the population covered for a rule to be included, default: 0.1 (10%)
    proportion_threshold = the minimum number of correctly predicted instances of any class to be included, default: 0.8
    max_rules: the maximum total number of rules (new features) to generate, default: 100
    features_fraction: the fraction of features to use at each iteration for fitting a decission tree, default 0.3
    min_depth: the minimum depth of each decission tree fitted, default: 1
    max_depth = the maximum depth of each decision tree fitted, default: 5
    feature_names: a list of feature names, default:  None, if None and X is a pandas DataFrame, columns of X will be used.
    rule_prefix: a prefix to include as part of each new feature generated, default: 'dtr_rule'
    n_iter: number of iterations,i.e., number of decision tree rounds to look for good rules, default: 1 (uses a single desicion tree and retuns its best rules)
    """

    def __init__(self, estimator, percent_threshold=0.1, proportion_threshold=0.8, max_rules=100, features_fraction=0.3, min_depth=1, max_depth=5,
                 feature_names=None, rule_prefix='dtr_rule', n_iter=1, n_jobs=1, random_state=None, verbose=0):

        self.estimator = estimator
        self.percent_threshold = percent_threshold
        self.proportion_threshold = proportion_threshold
        self.max_rules = max_rules
        self.features_fraction = features_fraction
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.rule_prefix = rule_prefix
        self.n_iter = n_iter
        self.rule_dict = None
        self.rule_set = None
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.posterior = pd.DataFrame()
        self.rule_stats = pd.DataFrame()

        self.classes = None

    def __len__(self):
        """Overloads `len` output to be the number of generated rules"""
        if not self.rule_set:
            return 0
        return len(self.rule_set)

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self.rule_dict[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not self.rule_set:
            return "empty or unfitted"
        return list(self.rule_set)

    def _transform_with_rules(self, X):
        _X_copy = X.copy()
        _X_copy['rule_class'] = 0.0
        self.posterior = pd.DataFrame(index=_X_copy.index.copy())
        self.probas = pd.DataFrame(index=_X_copy.index.copy())
        for i in range(len(self.classes)):
            self.probas['c_%s' % i] = -1.0
        self.posterior['class'] = -1
        self.posterior['p'] = -1.0
        self.posterior['name'] = None
        self.posterior['rule'] = None
        total_sample = len(self.posterior.index)
        for k, rule in reversed(self.rule_dict.items()):
            # init dataframe column to 0.0
            _X_copy[k] = 0.0
            # get indices of samples fullining the rule
            ind = _X_copy.query(rule['rule']).index
            _X_copy.loc[ind, k] = 10 * (rule['class'] - 0.5) * rule['probability']
            # _X_copy.loc[ind, 'rule_class'] = 2 * (rule['class'] - 0.5)
            self.posterior.loc[ind, 'class'] = rule['class']
            self.posterior.loc[ind, 'p'] = rule['probability']
            self.posterior.loc[ind, 'name'] = k
            self.posterior.loc[ind, 'rule'] = rule['rule']
            for i in range(len(self.classes)):
                self.probas.loc[ind, 'c_%s' % i] = rule['proportions'][i]

        # get statistics about the rule system
        self.rule_stats = pd.DataFrame(self.posterior[['name', 'rule', 'class']].groupby(['name', 'rule']).agg(['count']))
        self.rule_stats = self.rule_stats.reset_index().set_index('name')
        self.rule_stats['pct_support'] = (self.rule_stats[('class', 'count')] / total_sample).round(2)
        self.rule_stats.sort_values(by=('class', 'count'), ascending=True, inplace=True)

        return _X_copy

    def transform(self, X):
        """Transform X by adding the rules as binary variables.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_components]
            Transformed array.
        """
        if not self.rule_set:
            raise NotFittedError("DTRTransformer not fitted.")

        return self._transform_with_rules(X)

    def fit_tree(self, X, y,
                 percent_threshold=None,
                 proportion_threshold=None,
                 features_fraction=None,
                 min_depth=None,
                 max_depth=None,
                 max_rules=None,
                 n_iter=None,
                 n=None,
                 frac=None,
                 n_jobs=None):
        """Extract rules from a series of random tress a stores them in the rules dict.

                Parameters
                ----------
                X : array-like, shape = [n_samples, n_features]
                    Training vectors, where n_samples is the number of samples and
                    n_features is the number of features.

                y : array-like, shape = [n_samples]
                    Target values.


                Returns
                -------
                self
                """

        self.percent_threshold = percent_threshold if percent_threshold is not None else self.percent_threshold
        self.proportion_threshold = proportion_threshold if proportion_threshold is not None else self.proportion_threshold
        self.max_rules = max_rules if max_rules is not None else self.max_rules
        self.features_fraction = features_fraction if features_fraction is not None else self.features_fraction
        self.min_depth = min_depth if min_depth is not None else self.min_depth
        self.max_depth = max_depth if max_depth is not None else self.max_depth
        self.n_iter = n_iter if n_iter is not None else self.n_iter

        n_jobs = self.n_jobs if n_jobs is None else n_jobs

        self.classes = pd.Series(y).unique()
        MAX_INT = np.iinfo(np.int32).max
        rule_list = []
        rule_set = set()
        dt = tree.DecisionTreeClassifier(criterion='gini',
                                         max_depth=self.max_depth,
                                         splitter='best'
                                         )
        dt_fitted = dt.fit(X, y)
        rules_tuple, rules_set = get_rules_of_decision_tree(dt_fitted,
                                                            feature_names=list(X.columns),
                                                            percent_threshold=self.percent_threshold,
                                                            proportion_threshold=self.proportion_threshold,
                                                            min_depth=self.min_depth
                                                            )
        rule_list += rules_tuple
        rule_set |= rules_set

        if self.n_iter >= 1:

            counter = 0
            return_values_list = Parallel(n_jobs=n_jobs, verbose=1,require='sharedmem',prefer='threads'
                                          )(delayed(self.fit_one_tree)(X, y) for _ in range(self.n_iter))

            for rules_tuple, rules_set in return_values_list:
                # rule_list, rule_set = self.fit_one_tree(X, counter, rule_list, rule_set, y)
                rule_list += rules_tuple
                rule_set |= rules_set

        final_rule_dict = OrderedDict()
        final_rule_set = set()

        rules_iterator = sorted(rule_list, key=lambda x: x['percent'], reverse=True)
        self.max_rules = min(self.max_rules, len(rule_set))

        for r in filter(lambda rule: rule['class'] == 0, rules_iterator):
            if len(final_rule_set) >= self.max_rules // 2:
                break
            r['rule'] = simplify_rule(r['rules'])
            if r['rule'] not in final_rule_set and len(r['features']) >= self.min_depth:
                final_rule_set.add(r['rule'])
                final_rule_dict['%s_%d' % (self.rule_prefix, len(final_rule_dict) + 1)] = r

        rules_iterator = sorted(rule_list, key=lambda x: x['percent'], reverse=True)
        for r in filter(lambda rule: rule['class'] == 1, rules_iterator):
            if len(final_rule_set) >= self.max_rules:
                break
            r['rule'] = simplify_rule(r['rules'])
            if r['rule'] not in final_rule_set and len(r['features']) >= self.min_depth:
                final_rule_set.add(r['rule'])
                final_rule_dict['%s_%d' % (self.rule_prefix, len(final_rule_dict) + 1)] = r

        self.rule_dict = final_rule_dict
        self.rule_set = final_rule_set

        return self

    def fit_one_tree(self, X, y):
        dt = tree.DecisionTreeClassifier(criterion=random.choice(['gini', 'entropy']),
                                         max_depth=self.max_depth,
                                         class_weight=random.choice(['balanced', None]),
                                         splitter=random.choice(['best', 'random'])
                                         )
        # counter += 1
        # if self.verbose >= 2 and counter % 10 == 0:
        #    print('Fitting tree %d of %d ' % (counter, self.n_iter))
        _X_train = X.sample(frac=self.features_fraction, axis=1)
        _y_train = y[_X_train.index]
        dt_fitted = dt.fit(_X_train, _y_train, sample_weight=compute_sample_weight(class_weight='balanced', y=_y_train))
        rules_tuple, rules_set = get_rules_of_decision_tree(dt_fitted, list(_X_train.columns), percent_threshold=self.percent_threshold,
                                                            proportion_threshold=self.proportion_threshold
                                                            )
        rule_list = rules_tuple
        rule_set = rules_set
        return rule_list, rule_set

    def fit(self, X, y):
        return self.estimator.fit(X, y)

    def predict(self, X):
        y_pred_class = self.estimator.predict(X)

        # if not self.rule_set:
        return y_pred_class

        # out_predictions = self.posterior['class'].where(self.posterior['class'] != -1, y_pred_class)
        # return out_predictions.values

    def predict_proba(self, X):
        probas = self.estimator.predict_proba(X)
        # if not self.rule_set:
        return probas

        # out_probas = self.probas.where(self.probas.max(axis = 1) > -1.0, probas)
        # return out_probas.values

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.


        Returns
        -------
        X_new : array-like, shape = [n_samples, n_components]
            Transformed array.
        """
        return self.fit(X, y).transform(X)
