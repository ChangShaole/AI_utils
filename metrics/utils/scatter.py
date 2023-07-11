import argparse
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def parser_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameters for the script config.")
    parser.add_argument('-f', '--file', type=str, 
                        help="predict file")
    parser.add_argument('-c1', '--column_1', type=str,
                        help="ground truth label column")
    parser.add_argument('-c2', '--column_2', type=str, 
                        help="pred proba column.")
    parser.add_argument('-t', '--title', type=str, default='scatter',
                        required=False, help="the out figure title. (default: scatter)")
    parser.add_argument('-o', '--output', type=str, default='scatter.png',
                        required=False, help="the out figure name. (default: scatter.png)")
    return parser.parse_args(argv)


args = parser_args()


df = pd.read_csv(args.file)

xlim_min = df[args.column_1].min()
xlim_max = df[args.column_1].max()
ylim_min = df[args.column_2].min()
ylim_max = df[args.column_2].max()

lim_min = min(xlim_min, ylim_min)
lim_max = max(xlim_max, ylim_max)
extend = 0.05*(lim_max-lim_min)


pcc, p_value = pearsonr(df[args.column_1], df[args.column_2])

plt.figure()
plt.scatter(df[args.column_1], df[args.column_2], color='darkorange', lw=2, label='PCC: {}'.format(round(pcc,3)))
plt.plot([lim_min-extend, lim_max+extend], [lim_min-extend, lim_max+extend], color='navy', lw=2, linestyle='--')
plt.xlim([lim_min-extend, lim_max+extend])
plt.ylim([lim_min-extend, lim_max+extend])
plt.xlabel('Ground Truth Label')
plt.ylabel('Regression Prediction')
plt.title(args.title)
plt.legend(loc="lower right")
plt.savefig(args.output)
