
import glob
import bisect
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tqdm import tqdm

import sys
import pdb

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
import argparse

###training augmented
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def process_calls_df(args):
    fs, training_df, calls_idx = args
    calls_df = pd.read_csv(fs[0])
    if len(fs) == 2:
        calls2_df = pd.read_csv(fs[1])
        calls_df = pd.concat([calls_df, calls2_df])
    calls_df = calls_df.sort_values(['chr', 'coord'])
    calls_df = calls_df.groupby('chr').agg(list)
    calls_locs_df = calls_df.loc[:,'coord']
    calls_purities_df = calls_df.loc[:,'purity']

    enhancer_calls_counts = np.zeros((len(training_df),))
    promoter_calls_counts = np.zeros((len(training_df),))
    for training_idx, row in enumerate(training_df.iterrows()):
        row = row[1]
        enhancer_chrom, promoter_chrom = row['enhancer_chrom'], row['promoter_chrom']
        enhancer_start, enhancer_end = row['enhancer_start'], row['enhancer_end']
        promoter_start, promoter_end = row['promoter_start'], row['promoter_end']

        chr_calls = calls_locs_df.get(enhancer_chrom, None)
        chr_purities = calls_purities_df.get(enhancer_chrom, None)
        if chr_calls is not None:
            start_idx = bisect.bisect(chr_calls, enhancer_start)
            for idx in range(start_idx, len(chr_calls)):
                if chr_calls[idx] < enhancer_end:
                    enhancer_calls_counts[training_idx] = max(enhancer_calls_counts[training_idx], chr_purities[idx])
        
        chr_calls = calls_locs_df.get(promoter_chrom, None)
        chr_purities = calls_purities_df.get(promoter_chrom, None)
        if chr_calls is not None:
            start_idx = bisect.bisect(chr_calls, promoter_start)
            for idx in range(start_idx, len(chr_calls)):
                if chr_calls[idx] < promoter_end:
                    promoter_calls_counts[training_idx] = max(promoter_calls_counts[training_idx], chr_purities[idx])
    name = fs[0].split('/')[-1].split('-')[1]
    tqdm.write('Completing ' + str(calls_idx) + ' ' + name)
    return (calls_idx, name, enhancer_calls_counts, promoter_calls_counts)

def augmented(): #all of this was originally not in a function and just sort of operated on its own. 
	# Main
	cell_line = sys.argv[1]
	pool = Pool(70)
	training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-eep/training.h5', 'training')
	calls_dir = './data/'+cell_line+'/calls/'
	calls_files = sorted(glob.glob(calls_dir + "*-calls.csv"))
	inputs = []
	for name in calls_files:
	    if inputs and inputs[-1][0].split('/')[-1].split('-')[0] == name.split('/')[-1].split('-')[0]:
	        inputs[-1].append(name)
	    else:
	        inputs.append([name])
	input_args = [(inputs[i], training_df, i) for i in range(len(inputs))]
	for ret in tqdm(pool.imap(process_calls_df, input_args), total=len(input_args)):
	    calls_idx, name, enhancer_calls_counts, promoter_calls_counts = ret
	    col_name = str(calls_idx) + ' ' + name
	    training_df[col_name + ' (Enhancer)'] = pd.Series(enhancer_calls_counts, index=training_df.index)
	    training_df[col_name + ' (Promoter)'] = pd.Series(promoter_calls_counts, index=training_df.index)

	print('Saving...')
	training_df.to_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-eep/augmented_training.h5', key='training')
	print('Done')


#### training epw and training eep.
def eep_epw(eep=True): #originally did not have its own function. 
	# Get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode')
	parser.add_argument('--cell_line')
	parser.add_argument('--fixed_params')
	args = parser.parse_args()
	mode = args.mode
	cell_line = str(args.cell_line)
	fixed_params = args.fixed_params
	print("Training cell_line={} in mode {}".format(cell_line, mode))

	# Get training data
	nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window', 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

	#for augmented
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
	return training_df, predictors_df, labels

def cv_mini(training_df, predictors_df, labels, test_chrom):
	idxs_per_chrom = {}
	chroms = pd.unique(training_df['enhancer_chrom'])
	for chrom in chroms:
		training_df['enhancer_chrom'] == chrom
		idxs_per_chrom[chrom] = np.where(training_df['enhancer_chrom'] == chrom)[0]
		chroms = pd.unique(training_df['enhancer_chrom'])
		#print (set(range(len(predictors_df))))
	#import pdb; pdb.set_trace()
	train_idxs = np.array(list(set(range(len(predictors_df)))-set(idxs_per_chrom[test_chrom])),dtype=int)
	val_idxs = idxs_per_chrom[test_chrom].astype(int)
	yield train_idxs, val_idxs

def cv(training_df, predictors_df, labels, chroms, fixed_params = True):
	# Get idxs per chrom
	idxs_per_chrom = {}
	chroms = pd.unique(training_df['enhancer_chrom'])
	for chrom in chroms:
	    training_df['enhancer_chrom'] == chrom
	    idxs_per_chrom[chrom] = np.where(training_df['enhancer_chrom'] == chrom)[0]
	    yield train_idxs, val_idxs

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
	    print("Proper cross validation")
	    estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.1, max_depth = 5, max_features = 'log2', random_state = 0)
	    def cv():
	        for chrom in set(chroms):
	            train_idxs = np.array(list(set(range(len(predictors_df)))-set(idxs_per_chrom[chrom])),dtype=int)
	            val_idxs = idxs_per_chrom[chrom].astype(int)
	            yield train_idxs, val_idxs
	    scores = cross_val_score(estimator, predictors_df, labels, scoring = 'roc_auc', cv = cv(), n_jobs = -1)
	    print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

	    estimator.fit(predictors_df, labels)
	    importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
	    print(importances.head(16))

	    print("Improper cross validation")
	    estimator = GradientBoostingClassifier(n_estimators = 4000, learning_rate = 0.1, max_depth = 5, max_features = 'log2', random_state = 0)
	    cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

	    scores = cross_val_score(estimator, predictors_df, labels, scoring = 'roc_auc', cv = cv, n_jobs = -1)
	    print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

	    estimator.fit(predictors_df, labels)
	    importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
	    print(importances.head(16))


#https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

# #for training augmented
# X = 
# Y = 

#for training eep
# training_df, X,y = eep_epw()
# cv = cv(training_df, X, y)
# parameters = = {
#     'n_estimators': [100, 200, 500, 1000, 4000],
#     'max_depth': [2, 5, 10]
# }
# test_chrom = 14

# #for train epw
training_df, X,y = eep_epw()
cv = cv(training_df, X, y, "chr14")
parameters = {
    'n_estimators': [500],
    'max_depth': [5]
}
test_chrom = "chr14"

# Create the RFE object and compute a cross-validated score.
gbt = GradientBoostingClassifier(learning_rate=0.1,max_features='log2',random_state=0, n_estimators =500, max_depth = 5)
                            #parameters, cv=cv_mini(training_df, X,y, test_chrom), scoring='roc_auc', iid=True, n_jobs=-1# tree
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=gbt, step=500, cv=cv_mini(training_df, X,y,test_chrom),
              scoring='roc_auc', verbose=1) 
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
