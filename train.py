import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode')
parser.add_argument('--cell_line')
parser.add_argument('--fixed_params')
args = parser.parse_args()
mode = args.mode
cell_line = args.cell_line
fixed_params = args.fixed_params
print("Training cell_line={} in mode {}".format(cell_line, mode))

# Get training data
nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window', 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

orig_training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-eep/training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-eep/augmented_training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
assert np.sum(training_df['enhancer_chrom']==training_df['promoter_chrom']) == len(training_df)
predictors_df = training_df.drop(nonpredictors, axis = 1)
orig_num_cols = len(orig_training_df.columns)
if mode == 'piq-only':
    predictors_df = training_df.iloc[:,orig_num_cols:].drop(nonpredictors, axis = 1, errors='ignore')
elif mode == 'genomic-only':
    predictors_df = training_df.iloc[:,:orig_num_cols].drop(nonpredictors, axis = 1, errors='ignore')
elif mode == 'genomic-piq':
    predictors_df = training_df.drop(nonpredictors, axis = 1, errors='ignore')
else:
    raise Exception("Unsupported mode")
labels = training_df['label']

# Get idxs per chrom
idxs_per_chrom = {}
chroms = pd.unique(training_df['enhancer_chrom'])
for chrom in chroms:
    training_df['enhancer_chrom'] == chrom
    idxs_per_chrom[chrom] = np.where(training_df['enhancer_chrom'] == chrom)[0]

# Nested cross validation
if fixed_params == 'True':
    parameters = {
        'n_estimators': [4000],
        'max_depth': [5]
    }
else:
    parameters = {
        'n_estimators': [100, 200, 500, 1000, 4000],
        'max_depth': [2, 5, 10]
    }
print("Proper Cross Validation")
f1s, roc_aucs, importances = [], [], []
for test_chrom in chroms:
    test_idxs = idxs_per_chrom[test_chrom].astype(int)
    X_test, y_test = predictors_df.iloc[test_idxs], labels.iloc[test_idxs]

    def cv(test_chrom):
        for chrom in set(chroms)-set([test_chrom]):
            train_idxs = np.array(list(set(range(len(predictors_df)))-set(idxs_per_chrom[chrom])-set(idxs_per_chrom[test_chrom])),dtype=int)
            val_idxs = idxs_per_chrom[chrom].astype(int)
            yield train_idxs, val_idxs

    clf = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1,max_features='log2',random_state=0), 
                        parameters, cv=cv(test_chrom), scoring='roc_auc', iid=True, n_jobs=-1)
    clf.fit(predictors_df, labels)

    y_test_pred = clf.predict(X_test)
    y_test_probs = clf.predict_proba(X_test)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_probs[:,1])
    f1s.append(f1)
    roc_aucs.append(roc_auc)

    this_importances = pd.Series(clf.best_estimator_.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
    importances.append(this_importances)
print("F1: {:.5f}, ROC_AUC: {:.5f}".format(np.mean(f1s), np.mean(roc_aucs)))
for idx, chrom in enumerate(chroms):
    print("{}\t{:.5f}\t{:.5f}".format(chrom, f1s[idx], roc_aucs[idx]))
for idx, chrom in enumerate(chroms):
    print(chrom)
    print(importances[idx].head(16))

if fixed_params:
    print("Improper cross validation")
    estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.1, max_depth = 5, max_features = 'log2', random_state = 0)
    cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

    scores = cross_val_score(estimator, predictors_df, labels, scoring = 'roc_auc', cv = cv, n_jobs = -1)
    print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

    estimator.fit(predictors_df, labels)
    importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
    print(importances.head(16))
