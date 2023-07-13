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
                        help="the input features column list (split with ,)")
    parser.add_argument('-c2', '--column_2', type=str,
                        help="the fit target column.")
    parser.add_argument('-rc', '--regression_coefficient', type=float,
                        help="the regression coefficient value.")
    parser.add_argument('-i', '--intercept', type=float,
                        help="the intercept value.")
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

x_extend1 = 0.2*(xlim_max-xlim_min)
y_extend1 = 0.2*(ylim_max-ylim_min)
x_extend2 = 0.05*(xlim_max-xlim_min)
y_extend2 = 0.05*(ylim_max-ylim_min)


pcc, p_value = pearsonr(df[args.column_1], df[args.column_2])

plt.figure()
plt.scatter(df[args.column_1], df[args.column_2], color='darkorange', lw=0.8, label='PCC: {}'.format(round(pcc,3)))
plt.plot([xlim_min-x_extend2, xlim_max+x_extend2],
         [(xlim_min-x_extend2)*(args.regression_coefficient)+args.intercept,
          (xlim_max+x_extend2)*(args.regression_coefficient)+args.intercept],
          color='navy', lw=2, linestyle='--')
plt.xlim([xlim_min-x_extend1, xlim_max+x_extend1])
plt.ylim([ylim_min-y_extend1, ylim_max+y_extend1])
plt.xlabel(args.column_1)
plt.ylabel(args.column_2)
plt.title(args.title)
plt.legend(loc="lower right")
plt.savefig(args.output)
