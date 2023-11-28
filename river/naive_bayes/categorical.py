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
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.category_counts = defaultdict(set)

    def learn_one(self, x: dict, y):
        y = str(y).lower()
        # Update class count
        self.class_counts[y] +=1

        # Update feature count given class in model
        for i,t in x.items():
            self.category_counts[i].add(t)
            self.feature_counts[f"{i}_{t}"][y]+=1

        return self

    def p_feature_given_class(self, feature, category, label):

        num = self.feature_counts.get(f"{feature}_{category}", {}).get(label,0) + self.alpha
        print(num)
        den = self.class_counts[label] + self.alpha * len(self.category_counts[feature])

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

        X = self._encodeX(X)
        y = base.one_hot_encode(y)
        features, classes =  X.columns, [str(c).lower() for c in y.columns]

        y = sparse.csc_matrix(y).T
        # update class counts
        for label, count in zip(classes, y.sum(axis=1)):
            self.class_counts[label] += count.item()
        
        # update feature counts
        fc = y @ X

        for r,c in zip(*fc.nonzero()):
            self.feature_counts[features[c]][classes[r]] += fc[r,c]


    def joint_log_likelihood_many(self): 
        """ !!! Under construction !!!
        -----
        TODO
        - Handle missing features
        - Use sparse matrix approach for efficiency
        """
        
        pass

    def _encodeX(self, df: pd.DataFrame):
        feature_subsets = []
        for col in df.columns:
            temp = base.one_hot_encode(df[col])
            temp.columns = [f"{col}_{temp_col}" for temp_col in temp.columns]
            feature_subsets.append(temp)
        
        return pd.concat(feature_subsets,axis=1)