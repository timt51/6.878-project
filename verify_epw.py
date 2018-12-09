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

orig_training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-epw/training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
if mode == 'piq-window-only' or mode == 'genomic-piq-window':
    training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-epw/augmented_window_training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
elif mode == 'piq-no-window-only' or mode == 'genomic-piq-no-window':
    training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-epw/augmented_no_window_training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
elif mode == 'genomic-only': # doesn't matter
    training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-epw/augmented_no_window_training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
assert np.sum(training_df['enhancer_chrom']==training_df['promoter_chrom']) == len(training_df)
predictors_df = training_df.drop(nonpredictors, axis = 1)
orig_num_cols = len(orig_training_df.columns)
if mode == 'piq-window-only' or mode == 'piq-no-window-only':
    predictors_df = training_df.iloc[:,orig_num_cols:].drop(nonpredictors, axis = 1, errors='ignore')
elif mode == 'genomic-only':
    predictors_df = training_df.iloc[:,:orig_num_cols].drop(nonpredictors, axis = 1, errors='ignore')
elif mode == 'genomic-piq-window' or mode == 'genomic-piq-no-window':
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
        ('HUVEC', 'genomic-only', 'chr1'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr10'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr11'): (100, 5),
        ('HUVEC', 'genomic-only', 'chr12'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr13'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr14'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr15'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr16'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr17'): (100, 5),
        ('HUVEC', 'genomic-only', 'chr18'): (100, 5),
        ('HUVEC', 'genomic-only', 'chr19'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr2'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr20'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr21'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr22'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr3'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr4'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr5'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr6'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr7'): (200, 2),
        ('HUVEC', 'genomic-only', 'chr8'): (500, 2),
        ('HUVEC', 'genomic-only', 'chr9'): (500, 2),
        ('HUVEC', 'genomic-only', 'chrX'): (500, 2),

        ('HUVEC', 'piq-no-window-only', 'chr1'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr10'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr11'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr12'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr13'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr14'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr15'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr16'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr17'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr18'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr19'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr2'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr20'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr21'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr22'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr3'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr4'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr5'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr6'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr7'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr8'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chr9'): (500, 2),
        ('HUVEC', 'piq-no-window-only', 'chrX'): (500, 2),

        ('HUVEC', 'piq-window-only', 'chr1'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr10'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr11'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr12'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr13'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr14'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr15'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr16'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr17'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr18'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr19'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr2'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr20'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr21'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr22'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr3'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr4'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr5'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr6'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr7'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr8'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chr9'): (750, 2),
        ('HUVEC', 'piq-window-only', 'chrX'): (750, 2),

        ('HUVEC', 'genomic-piq-no-window', 'chr1'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr10'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr11'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr12'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr13'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr14'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr15'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr16'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr17'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr18'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr19'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr2'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr20'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr21'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr22'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr3'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr4'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr5'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr6'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr7'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr8'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chr9'): (500, 2),
        ('HUVEC', 'genomic-piq-no-window', 'chrX'): (500, 2),

        ('HUVEC', 'genomic-piq-window', 'chr1'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr10'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr11'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr12'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr13'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr14'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr15'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr16'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr17'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr18'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr19'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr2'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr20'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr21'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr22'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr3'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr4'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr5'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr6'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr7'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr8'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chr9'): (750, 2),
        ('HUVEC', 'genomic-piq-window', 'chrX'): (750, 2),

        ('K562', 'genomic-only', 'chr1'): (200, 2),
        ('K562', 'genomic-only', 'chr10'): (200, 2),
        ('K562', 'genomic-only', 'chr11'): (100, 5),
        ('K562', 'genomic-only', 'chr12'): (200, 2),
        ('K562', 'genomic-only', 'chr13'): (200, 2),
        ('K562', 'genomic-only', 'chr14'): (200, 2),
        ('K562', 'genomic-only', 'chr15'): (200, 2),
        ('K562', 'genomic-only', 'chr16'): (200, 2),
        ('K562', 'genomic-only', 'chr17'): (200, 2),
        ('K562', 'genomic-only', 'chr18'): (200, 2),
        ('K562', 'genomic-only', 'chr19'): (200, 2),
        ('K562', 'genomic-only', 'chr2'): (200, 2),
        ('K562', 'genomic-only', 'chr20'): (100, 5),
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

        ('K562', 'piq-no-window-only', 'chr1'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr10'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr11'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr12'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr13'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr14'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr15'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr16'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr17'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr18'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr19'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr2'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr20'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr21'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr22'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr3'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr4'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr5'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr6'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr7'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr8'): (500, 2),
        ('K562', 'piq-no-window-only', 'chr9'): (500, 2),
        ('K562', 'piq-no-window-only', 'chrX'): (500, 2),

        ('K562', 'piq-window-only', 'chr1'): (750, 2),
        ('K562', 'piq-window-only', 'chr10'): (750, 2),
        ('K562', 'piq-window-only', 'chr11'): (750, 2),
        ('K562', 'piq-window-only', 'chr12'): (750, 2),
        ('K562', 'piq-window-only', 'chr13'): (750, 2),
        ('K562', 'piq-window-only', 'chr14'): (750, 2),
        ('K562', 'piq-window-only', 'chr15'): (750, 2),
        ('K562', 'piq-window-only', 'chr16'): (750, 2),
        ('K562', 'piq-window-only', 'chr17'): (750, 2),
        ('K562', 'piq-window-only', 'chr18'): (750, 2),
        ('K562', 'piq-window-only', 'chr19'): (750, 2),
        ('K562', 'piq-window-only', 'chr2'): (750, 2),
        ('K562', 'piq-window-only', 'chr20'): (750, 2),
        ('K562', 'piq-window-only', 'chr21'): (750, 2),
        ('K562', 'piq-window-only', 'chr22'): (750, 2),
        ('K562', 'piq-window-only', 'chr3'): (750, 2),
        ('K562', 'piq-window-only', 'chr4'): (750, 2),
        ('K562', 'piq-window-only', 'chr5'): (750, 2),
        ('K562', 'piq-window-only', 'chr6'): (750, 2),
        ('K562', 'piq-window-only', 'chr7'): (750, 2),
        ('K562', 'piq-window-only', 'chr8'): (750, 2),
        ('K562', 'piq-window-only', 'chr9'): (750, 2),
        ('K562', 'piq-window-only', 'chrX'): (750, 2),

        ('K562', 'genomic-piq-no-window', 'chr1'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr10'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr11'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr12'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr13'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr14'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr15'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr16'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr17'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr18'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr19'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr2'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr20'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr21'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr22'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr3'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr4'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr5'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr6'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr7'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr8'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chr9'): (500, 2),
        ('K562', 'genomic-piq-no-window', 'chrX'): (500, 2),

        ('K562', 'genomic-piq-window', 'chr1'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr10'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr11'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr12'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr13'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr14'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr15'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr16'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr17'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr18'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr19'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr2'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr20'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr21'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr22'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr3'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr4'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr5'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr6'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr7'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr8'): (750, 2),
        ('K562', 'genomic-piq-window', 'chr9'): (750, 2),
        ('K562', 'genomic-piq-window', 'chrX'): (750, 2),
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
