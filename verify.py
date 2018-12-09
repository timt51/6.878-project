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
if fixed_params == 'False':
    parameters = {
        'n_estimators': [100, 200, 500, 1000, 4000],
        'max_depth': [2, 5, 10]
    }
    print("Proper Cross Validation")
    f1s, roc_aucs, importances, best_params = [], [], [], []
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

        best_params.append(clf.best_params_)
    print("F1: {:.5f}, ROC_AUC: {:.5f}".format(np.mean(f1s), np.mean(roc_aucs)))
    for idx, chrom in enumerate(chroms):
        print("{}\t{:.5f}\t{:.5f}".format(chrom, f1s[idx], roc_aucs[idx]))
    for idx, chrom in enumerate(chroms):
        print(chrom)
        print(importances[idx].head(16))
    for idx, chrom in enumerate(chroms):
        print(chrom)
        print(best_params[idx])
else:
    parameters = {
        ('HUVEC', 'genomic-only', 'chr1'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr10'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr11'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr12'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr13'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr14'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr15'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr16'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr17'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr18'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr19'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr2'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr20'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr21'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr22'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr3'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr4'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr5'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr6'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr7'): (100, 2),
        ('HUVEC', 'genomic-only', 'chr8'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr9'): (100, 2),
        ('HUVEC', 'genomic-only', 'chrX'): (100, 2),

        ('HUVEC', 'piq-only', 'chr1'): (500, 2),
        ('HUVEC', 'piq-only', 'chr10'): (500, 2),
        ('HUVEC', 'piq-only', 'chr11'): (500, 2),
        ('HUVEC', 'piq-only', 'chr12'): (100, 5),
        ('HUVEC', 'piq-only', 'chr13'): (500, 2),
        ('HUVEC', 'piq-only', 'chr14'): (500, 2),
        ('HUVEC', 'piq-only', 'chr15'): (500, 2),
        ('HUVEC', 'piq-only', 'chr16'): (500, 2),
        ('HUVEC', 'piq-only', 'chr17'): (500, 2),
        ('HUVEC', 'piq-only', 'chr18'): (500, 2),
        ('HUVEC', 'piq-only', 'chr19'): (500, 2),
        ('HUVEC', 'piq-only', 'chr2'): (500, 2),
        ('HUVEC', 'piq-only', 'chr20'): (500, 2),
        ('HUVEC', 'piq-only', 'chr21'): (500, 2),
        ('HUVEC', 'piq-only', 'chr22'): (500, 2),
        ('HUVEC', 'piq-only', 'chr3'): (500, 2),
        ('HUVEC', 'piq-only', 'chr4'): (500, 2),
        ('HUVEC', 'piq-only', 'chr5'): (500, 2),
        ('HUVEC', 'piq-only', 'chr6'): (500, 2),
        ('HUVEC', 'piq-only', 'chr7'): (500, 2),
        ('HUVEC', 'piq-only', 'chr8'): (500, 2),
        ('HUVEC', 'piq-only', 'chr9'): (500, 2),
        ('HUVEC', 'piq-only', 'chrX'): (500, 2),

        ('HUVEC', 'genomic-piq', 'chr1'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr10'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr11'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr12'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr13'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr14'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr15'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr16'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr17'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr18'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr19'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr2'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr20'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr21'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr22'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr3'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr4'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr5'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr6'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr7'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr8'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chr9'): (500, 2),
        ('HUVEC', 'genomic-piq', 'chrX'): (500, 2),

        ('K562', 'genomic-only', 'chr1'): (100, 2),
        ('K562', 'genomic-only', 'chr10'): (200, 2),
        ('K562', 'genomic-only', 'chr11'): (100, 2),
        ('K562', 'genomic-only', 'chr12'): (200, 2),
        ('K562', 'genomic-only', 'chr13'): (200, 2),
        ('K562', 'genomic-only', 'chr14'): (200, 2),
        ('K562', 'genomic-only', 'chr15'): (100, 2),
        ('K562', 'genomic-only', 'chr16'): (200, 2),
        ('K562', 'genomic-only', 'chr17'): (500, 2),
        ('K562', 'genomic-only', 'chr18'): (200, 2),
        ('K562', 'genomic-only', 'chr19'): (100, 2),
        ('K562', 'genomic-only', 'chr2'): (200, 2),
        ('K562', 'genomic-only', 'chr20'): (200, 2),
        ('K562', 'genomic-only', 'chr21'): (200, 2),
        ('K562', 'genomic-only', 'chr22'): (200, 2),
        ('K562', 'genomic-only', 'chr3'): (200, 2),
        ('K562', 'genomic-only', 'chr4'): (200, 2),
        ('K562', 'genomic-only', 'chr5'): (200, 2),
        ('K562', 'genomic-only', 'chr6'): (200, 2),
        ('K562', 'genomic-only', 'chr7'): (200, 2),
        ('K562', 'genomic-only', 'chr8'): (200, 2),
        ('K562', 'genomic-only', 'chr9'): (200, 2),
        ('K562', 'genomic-only', 'chrX'): (200, 2),

        ('K562', 'piq-only', 'chr1'): (200, 2),
        ('K562', 'piq-only', 'chr10'): (500, 2),
        ('K562', 'piq-only', 'chr11'): (500, 2),
        ('K562', 'piq-only', 'chr12'): (500, 2),
        ('K562', 'piq-only', 'chr13'): (500, 2),
        ('K562', 'piq-only', 'chr14'): (500, 2),
        ('K562', 'piq-only', 'chr15'): (500, 2),
        ('K562', 'piq-only', 'chr16'): (200, 2),
        ('K562', 'piq-only', 'chr17'): (200, 2),
        ('K562', 'piq-only', 'chr18'): (500, 2),
        ('K562', 'piq-only', 'chr19'): (200, 2),
        ('K562', 'piq-only', 'chr2'): (500, 2),
        ('K562', 'piq-only', 'chr20'): (200, 2),
        ('K562', 'piq-only', 'chr21'): (500, 2),
        ('K562', 'piq-only', 'chr22'): (500, 2),
        ('K562', 'piq-only', 'chr3'): (200, 2),
        ('K562', 'piq-only', 'chr4'): (500, 2),
        ('K562', 'piq-only', 'chr5'): (500, 2),
        ('K562', 'piq-only', 'chr6'): (200, 2),
        ('K562', 'piq-only', 'chr7'): (500, 2),
        ('K562', 'piq-only', 'chr8'): (200, 2),
        ('K562', 'piq-only', 'chr9'): (200, 2),
        ('K562', 'piq-only', 'chrX'): (500, 2),

        ('K562', 'genomic-piq', 'chr1'): (500, 2),
        ('K562', 'genomic-piq', 'chr10'): (500, 2),
        ('K562', 'genomic-piq', 'chr11'): (500, 2),
        ('K562', 'genomic-piq', 'chr12'): (500, 2),
        ('K562', 'genomic-piq', 'chr13'): (500, 2),
        ('K562', 'genomic-piq', 'chr14'): (500, 2),
        ('K562', 'genomic-piq', 'chr15'): (200, 5),
        ('K562', 'genomic-piq', 'chr16'): (500, 2),
        ('K562', 'genomic-piq', 'chr17'): (100, 5),
        ('K562', 'genomic-piq', 'chr18'): (500, 2),
        ('K562', 'genomic-piq', 'chr19'): (500, 2),
        ('K562', 'genomic-piq', 'chr2'): (500, 2),
        ('K562', 'genomic-piq', 'chr20'): (500, 2),
        ('K562', 'genomic-piq', 'chr21'): (500, 2),
        ('K562', 'genomic-piq', 'chr22'): (500, 2),
        ('K562', 'genomic-piq', 'chr3'): (200, 5),
        ('K562', 'genomic-piq', 'chr4'): (500, 2),
        ('K562', 'genomic-piq', 'chr5'): (500, 2),
        ('K562', 'genomic-piq', 'chr6'): (500, 2),
        ('K562', 'genomic-piq', 'chr7'): (100, 5),
        ('K562', 'genomic-piq', 'chr8'): (500, 2),
        ('K562', 'genomic-piq', 'chr9'): (100, 5),
        ('K562', 'genomic-piq', 'chrX'): (500, 2),
    }

    def cv(test_chrom):
        train_idxs = np.array(list(set(range(len(predictors_df)))-set(idxs_per_chrom[test_chrom])),dtype=int)
        val_idxs = idxs_per_chrom[test_chrom].astype(int)
        yield train_idxs, val_idxs

    print("Proper cross validation")
    f1s = []
    roc_aucs = []
    for chrom in chroms:
        n_estimators, max_depth = parameters[(cell_line, mode, chrom)]
        estimator = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate = 0.1, max_depth = max_depth, max_features = 'log2', random_state = 0)
        idxs = list(cv(chrom))
        train_idxs, test_idxs = idxs[0]
        estimator = estimator.fit(predictors_df.iloc[train_idxs], labels.iloc[train_idxs])
        pred_test_probs = estimator.predict_log_proba(predictors_df.iloc[test_idxs])
        pred_test_labels = estimator.predict(predictors_df.iloc[test_idxs])
        f1 = f1_score(labels.iloc[test_idxs], pred_test_labels)
        roc_auc = roc_auc_score(labels.iloc[test_idxs], pred_test_probs[:,1])
        f1s.append(f1)
        roc_aucs.append(roc_auc)
    print("F1: {:.5f}, ROC_AUC: {:.5f}".format(np.mean(f1s), np.mean(roc_aucs)))
    for idx, chrom in enumerate(chroms):
        print("{}\t{:.5f}\t{:.5f}".format(chrom, f1s[idx], roc_aucs[idx]))
    
    # estimator.fit(predictors_df, labels)
    # importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
    # print(importances.head(16))
