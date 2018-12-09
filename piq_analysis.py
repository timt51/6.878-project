import glob
import bisect
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score

from tqdm import tqdm

import sys

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

# Main
cell_line = sys.argv[1]
pool = Pool(6)
training_df = pd.read_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-eep/training.h5', 'training')
calls_dir = './data/'+cell_line+'/calls/'
calls_files = [
    calls_dir + '139-MA01391CTCF-calls.all.csv',
    calls_dir + '139-MA01391CTCF.RC-calls.all.csv',
    calls_dir + '459-CN00021LM2-calls.all.csv',
    calls_dir + '459-CN00021LM2.RC-calls.all.csv',
    calls_dir + '464-CN00071LM7-calls.all.csv',
    calls_dir + '464-CN00071LM7.RC-calls.all.csv',
    calls_dir + '480-CN00231LM23-calls.all.csv',
    calls_dir + '480-CN00231LM23.RC-calls.all.csv'
]
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

training_df = training_df[
    ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'CTCF (enhancer)',
    '0 MA01391CTCF (Enhancer)', '1 CN00021LM2 (Enhancer)', '2 CN00071LM7 (Enhancer)', '3 CN00231LM23 (Enhancer)',
    'promoter_chrom', 'promoter_start', 'promoter_end', 'CTCF (promoter)',
    '0 MA01391CTCF (Promoter)', '1 CN00021LM2 (Promoter)', '2 CN00071LM7 (Promoter)', '3 CN00231LM23 (Promoter)']]

# collaspe by enhancer/promoter
enhancer_stats_dict = {}
promoter_stats_dict = {}
for _, row in training_df.iterrows():
    enhancer_chrom, enhancer_start, enhancer_end, ctcf_chip_seq, ma_139, lm_2, lm_7, lm_23 = row[:8]
    enhancer_stats_dict[(enhancer_chrom, enhancer_start, enhancer_end)] = [ctcf_chip_seq, ma_139, lm_2, lm_7, lm_23]
    promoter_chrom, promoter_start, promoter_end, ctcf_chip_seq, ma_139, lm_2, lm_7, lm_23 = row[8:]
    promoter_stats_dict[(promoter_chrom, promoter_start, promoter_end)] = [ctcf_chip_seq, ma_139, lm_2, lm_7, lm_23]

# make np arrays
enhancer_stats = []
for v in enhancer_stats_dict.values():
    enhancer_stats.append(v)
enhancer_stats = np.array(enhancer_stats)
promoter_stats = []
for v in promoter_stats_dict.values():
    promoter_stats.append(v)
promoter_stats = np.array(promoter_stats)

# get metrics
# what is proportion of +/-, what happens when guess randomly, guess majority
purity_thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
f1s, prs, res = [], [], []
enhancer_labels = enhancer_stats[:,0] > 0
for threshold in purity_thresholds:
    ma_139_pred = enhancer_stats[:,1] >= threshold
    ma_139_lm_pred = np.any(enhancer_stats[:,1:] >= threshold, axis=1)

    f1 = f1_score(enhancer_labels, ma_139_pred)
    pr = precision_score(enhancer_labels, ma_139_pred)
    re = recall_score(enhancer_labels, ma_139_pred)
    f1s.append(f1)
    prs.append(pr)
    res.append(re)
    print(threshold, f1, pr, re)

best_f1_idx = np.argmax(f1s)
sz = 50
plt.figure()
plt.step(res, prs)
plt.scatter(res[0], prs[0], label='Motif Match Only F1='+str(f1s[0])[:6], color='orange', s=sz)
plt.scatter(res[best_f1_idx], prs[best_f1_idx], label='PIQ Best F1              ='+str(f1s[best_f1_idx])[:6], color='green', s=sz)
plt.xlim([0.5, 1])
plt.ylim([0, 1])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall For CTCF Binding Calls Using PIQ")
plt.legend()
plt.savefig("./figs/Pr vs Re.png")
import pdb; pdb.set_trace()