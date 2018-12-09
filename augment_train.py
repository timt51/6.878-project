import glob
import bisect
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tqdm import tqdm

import sys
import pdb

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
    window_calls_counts = np.zeros((len(training_df),))
    for training_idx, row in enumerate(training_df.iterrows()):
        row = row[1]
        enhancer_chrom, promoter_chrom, window_chrom = row['enhancer_chrom'], row['promoter_chrom'], row['window_chrom']
        enhancer_start, enhancer_end = row['enhancer_start'], row['enhancer_end']
        promoter_start, promoter_end = row['promoter_start'], row['promoter_end']
        window_start, window_end = row['window_start'], row['window_end']

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

        chr_calls = calls_locs_df.get(window_chrom, None)
        chr_purities = calls_purities_df.get(window_chrom, None)
        if chr_calls is not None:
            start_idx = bisect.bisect(chr_calls, window_start)
            for idx in range(start_idx, len(chr_calls)):
                if chr_calls[idx] < window_end:
                    window_calls_counts[training_idx] = max(window_calls_counts[training_idx], chr_purities[idx])
    name = fs[0].split('/')[-1].split('-')[1]
    tqdm.write('Completing ' + str(calls_idx) + ' ' + name)
    return (calls_idx, name, enhancer_calls_counts, promoter_calls_counts, window_calls_counts)

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
df_no_window = training_df.copy()
df_window = training_df.copy()
for ret in tqdm(pool.imap(process_calls_df, input_args), total=len(input_args)):
    calls_idx, name, enhancer_calls_counts, promoter_calls_counts, window_calls_counts = ret
    col_name = str(calls_idx) + ' ' + name
    df_no_window[col_name + ' (Enhancer)'] = pd.Series(enhancer_calls_counts, index=df_no_window.index)
    df_no_window[col_name + ' (Promoter)'] = pd.Series(promoter_calls_counts, index=df_no_window.index)

    df_window[col_name + ' (Enhancer)'] = pd.Series(enhancer_calls_counts, index=df_window.index)
    df_window[col_name + ' (Promoter)'] = pd.Series(promoter_calls_counts, index=df_window.index)
    df_window[col_name + ' (Window)'] = pd.Series(window_calls_counts, index=df_window.index)

print('Saving...')
df_no_window.to_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-epw/augmented_no_window_training.h5', key='training')
df_window.to_hdf('./targetfinder/paper/targetfinder/'+cell_line+'/output-epw/augmented_window_training.h5', key='training')
print('Done')
