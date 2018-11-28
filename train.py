import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

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

def train(args):
    predictors_df, labels, estimator, idxs_per_chrom, chrom = args
    train_idxs = np.array(list(set(range(len(predictors_df)))-set(idxs_per_chrom[chrom])),dtype=int)
    test_idxs = idxs_per_chrom[chrom].astype(int)
    x_train, y_train = predictors_df.iloc[train_idxs], labels.iloc[train_idxs]
    x_test, y_test = predictors_df.iloc[test_idxs], labels.iloc[test_idxs]
    estimator = estimator.fit(x_train,y_train)
    y_test_pred = estimator.predict(x_test)
    y_test_probs = estimator.predict_proba(x_test)
    return chrom, f1_score(y_test, y_test_pred), roc_auc_score(y_test, y_test_probs[:,1])

k = 10
pool = Pool(k)
chroms = np.random.choice(list(idxs_per_chrom.keys()),k,replace=False)
inputs_ = [(predictors_df, labels, estimator, idxs_per_chrom, chrom) for chrom in chroms]
f1s, aucs = [], []
for res in pool.imap_unordered(train, inputs_):
    chrom, f1, auc = res
    print(chrom, f1, auc)
print("Avg perf")
print(np.mean(f1s), np.mean(aucs))

# estimator.fit(predictors_df, labels)
# importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
# print(importances.head(16))
