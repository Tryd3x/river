from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from . import base

__all__ = ["CategoricalNB"]

class CategoricalNB(base.BaseNB):
    """ Naive Bayes classifier for categorical models.
    """

    def __init__(self, alpha = 1.0):
        self.alpha = alpha
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def learn_one(self, x: dict, y):
        y = str(y).lower()
        # Update class count
        self.class_counts[y] +=1

        # Update feature count given class in model
        for i,t in x.items():
            self.feature_counts[i][t][y]+=1

        return self

    def p_feature_given_class(self, feature, category, label):
        num = self.feature_counts.get(feature, {}).get(category, {0.0}).get(label,0) + self.alpha
        den = self.class_counts[label] + self.alpha * len(self.feature_counts.get(feature,{}).keys())

        return num/den
    
    def p_class(self,label):
        return self.class_counts[label]/sum(self.class_counts.values())

    def joint_log_likelihood(self,x: dict):
        return {y : np.log(self.p_class(y))
                + sum([
                    np.log(self.p_feature_given_class(
                        feature=i,
                        category=t,
                        label=y)) for i,t in x.items()])
                for y in self.class_counts.keys()}

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        y = base.one_hot_encode(y)
        classes = str(y.columns).lower()
        y = sparse.csc_matrix(y).T

        # update class counts
        for label, count in zip(classes, y.sum(axis=1)):
            self.class_counts[label] += count.item()

        # update feature count

        

        return self

    def joint_log_likelihood_many(self): 
        pass