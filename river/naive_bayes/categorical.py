from __future__ import annotations

import collections
import math

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
        self.class_counts = collections.Counter()
        self.feature_counts = collections.defaultdict(collections.Counter)

    def learn_one(self, x, y):
        pass
        # self.class_counts.update((y,))
        # for i, t in x.items():
            # feature_counts[i][t].update(y)

    def p_feature_given_class(self, feature, label):
        num = self.feature_counts.get(feature, {}).get(label, 0.0) + self.alpha
        den = self.class_counts[label] + self.alpha * len(self.feature_counts)

        return num/den
    
    def p_class(self,):
        pass

    def joint_log_likelihood(self):
        pass

    def learn_many(self):
        pass

    def _feature_log_prob(self):
        pass

    def joint_log_likelihood_many(self):
        pass