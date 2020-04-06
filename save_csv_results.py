import numpy as np
import pandas as pd
import pickle 
import os

def get_theta_gt(idx, table=None):
    if table is None:
        table = pd.read_csv("/content/new_3d_dataset/all_pairs_3d.csv")
    values = table.iloc[idx, -3:].values
    return values

def get_theta_model(idx, stats_dict):
    return np.array([
        stats_dict['rotate_value'][idx],
        stats_dict['scale_value'][idx],
        stats_dict['shift_value'][idx]
    ])

def get_results_pandas(ref_table, stats, siam_back=False):
    newtable = pd.DataFrame(columns = ref_table.columns.values)
    print(stats['rotate_value'].shape)
    for i in range(len(ref_table)):
        prefix = ref_table.iloc[i, :-3]
        if siam_back:
            newtable.loc[i] = [
                               *prefix, 
                               stats['rotate_value'][i][1], 
                               stats['scale_value'][i][1], 
                               stats['shift_value'][i][1],
                              ]
        else:
            newtable.loc[i] = [
                               *prefix, 
                               stats['rotate_value'][i][0], 
                               stats['scale_value'][i][0], 
                               stats['shift_value'][i][0],
                              ]
    return newtable

def lower_art_name(s):
    return '_'.join([i.lower() for i in s.split()])

def calculate_all_diff(ref_table, newtable):
    for art_name in ['Rotate value', 'Scale value', 'Shift value']:
        gt = ref_table[art_name].values
        res = newtable[art_name].values
        # res = stats[lower_art_name(art_name)].squeeze()
        diff = np.abs(gt - res)
        print('=== %s ===' % art_name)
        print('mean error:', np.mean(diff))
        print('median error:', np.median(diff))
        print('variance error:', np.std(diff))
        print()


RESULTS_DIR = r"C:\Users\22k_koz\Desktop\work\results"
LOGDIR =  r"C:\Users\22k_koz\Desktop\work\logdir"
MODEL_FN = "APR06_01"

def main():
    with open(os.path.join(LOGDIR, MODEL_FN, 'stats.pkl'), 'rb') as f:
        stats = pickle.load(f)
    mytable = pd.read_csv(os.path.join(RESULTS_DIR, 'all_pairs_3d.csv'))
    mystats = stats['affine_simple']
    newtable = get_results_pandas(mytable, mystats, siam_back=False)
    calculate_all_diff(mytable, newtable)
    newtable.to_csv()
    newtable.to_csv(os.path.join(RESULTS_DIR, MODEL_FN + '.csv'), index=False)

if __name__ == "__main__":
    main()