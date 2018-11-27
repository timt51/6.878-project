import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'window_chrom', 'window_start', 'window_end', 'window_name', 'active_promoters_in_window', 'interactions_in_window', 'enhancer_distance_to_promoter', 'bin', 'label']

training_df = pd.read_hdf('./targetfinder/paper/targetfinder/K562/output-eep/augmented_training.h5', 'training').set_index(['enhancer_name', 'promoter_name'])
predictors_df = training_df.drop(nonpredictors, axis = 1)
predictors_df = predictors_df.iloc[:,272:]
# predictors_df = predictors_df.iloc[:,:272]
labels = training_df['label']

estimator = GradientBoostingClassifier(n_estimators = 1000, learning_rate = 0.1, max_depth = 10, max_features = 'log2', random_state = 0)
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

scores = cross_val_score(estimator, predictors_df, labels, scoring = 'f1', cv = cv, n_jobs = -1)
print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

estimator.fit(predictors_df, labels)
importances = pd.Series(estimator.feature_importances_, index = predictors_df.columns).sort_values(ascending = False)
print(importances.head(16))
