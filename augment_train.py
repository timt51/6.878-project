import glob
import bisect
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tqdm import tqdm

def process_calls_df(args):
    calls_files, training_df, calls_idx = args
    calls_df = pd.read_csv(calls_files[calls_idx])
    calls2_df = pd.read_csv(calls_files[calls_idx+1])
    calls_df = pd.concat([calls_df, calls2_df])
    calls_locs_df = calls_df.groupby('chr')['coord'].apply(lambda x: sorted(list(x)))

    enhancer_calls_counts = np.zeros((len(training_df),))
    promoter_calls_counts = np.zeros((len(training_df),))
    for training_idx, row in tqdm(enumerate(training_df.iterrows()), total=len(training_df)):
        row = row[1]
        enhancer_chrom, promoter_chrom = row['enhancer_chrom'], row['promoter_chrom']
        enhancer_start, enhancer_end = row['enhancer_start'], row['enhancer_end']
        promoter_start, promoter_end = row['promoter_start'], row['promoter_end']

        chr_calls = calls_locs_df.get(enhancer_chrom, None)
        if chr_calls is not None:
            start_idx = bisect.bisect(chr_calls, enhancer_start)
            for idx in range(start_idx, len(chr_calls)):
                if chr_calls[idx] < enhancer_end:
                    enhancer_calls_counts[training_idx] += 1
        
        if chr_calls is not None:
            chr_calls = calls_locs_df.get(promoter_chrom, None)
            start_idx = bisect.bisect(chr_calls, promoter_start)
            for idx in range(start_idx, len(chr_calls)):
                if chr_calls[idx] < promoter_end:
                    promoter_calls_counts[training_idx] += 1
    return (calls_idx, enhancer_calls_counts, promoter_calls_counts)

# For K562
pool = Pool(6)
training_df = pd.read_hdf('./targetfinder/paper/targetfinder/K562/output-eep/training.h5', 'training')
calls_dir = './data/K562/calls/'
calls_files = sorted(glob.glob(calls_dir + "*-calls.csv"))
inputs = [(calls_files, training_df, i) for i in range(0,len(calls_files),2)]
for ret in tqdm(pool.imap(process_calls_df, inputs), total=len(calls_files)//2):
    calls_idx, enhancer_calls_counts, promoter_calls_counts = ret
    col_name = str(calls_idx//2)
    training_df[col_name + ' (Enhancer)'] = pd.Series(enhancer_calls_counts, index=training_df.index)
    training_df[col_name + ' (Promoter)'] = pd.Series(promoter_calls_counts, index=training_df.index)

print('Saving...')
training_df.to_hdf('.targetfinder/paper/targetfinder/K562/output-eep/augmented_training.h5', key='training')
print('Done')