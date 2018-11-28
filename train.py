import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

mode = sys.argv[1]
n_estimators = int(sys.argv[2])
max_depth = int(sys.argv[3])
cell_line = sys.argv[4]
print("Training cell_line={} in mode {} with n_estimators={}, max_depth={}".format(cell_line, mode, n_estimators, max_depth))

nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window', 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-eep/augmented_training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
assert np.sum(training_df['enhancer_chrom']==training_df['promoter_chrom']) == len(training_df)
predictors_df = training_df.drop(nonpredictors, axis = 1)
if mode == 'piq-only':
    predictors_df = predictors_df.iloc[:,272:]
elif mode == 'genomic-only':
    predictors_df = predictors_df.iloc[:,:272]
elif mode == 'genomic-piq':
    pass
else:
    raise Exception("Unsupported mode")
labels = training_df['label']

estimator = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate = 0.1, max_depth = max_depth, max_features = 'log2', random_state = 0)
np.random.seed(0)
idxs_per_chrom = {}
for chrom in pd.unique(training_df['enhancer_chrom']):
    training_df['enhancer_chrom'] == chrom
    idxs_per_chrom[chrom] = np.where(training_df['enhancer_chrom'] == chrom)[0]
cv_idxs = []
for chrom in np.random.choice(list(idxs_per_chrom.keys()),10,replace=False) :
    train_idxs = list(set(range(len(training_df)))-set(idxs_per_chrom[chrom]))
    test_idxs = idxs_per_chrom[chrom]
    cv_idxs.append((train_idxs, test_idxs))
cv = cv_idxs

scores = cross_val_score(estimator, predictors_df, labels, scoring = 'f1', cv = cv, n_jobs = -1)
print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

estimator.fit(predictors_df, labels)
importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
print(importances.head(16))
