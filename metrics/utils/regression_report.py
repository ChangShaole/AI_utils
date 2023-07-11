import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def parser_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameters for the script config.")
    parser.add_argument('-f', '--file', type=str, 
                        help="predict file")
    parser.add_argument('-c1', '--column_1', type=str,
                        help="ground truth label column")
    parser.add_argument('-c2', '--column_2', type=str, 
                        help="pred proba column.")
    parser.add_argument('-r', '--pearson', type=bool, default=False, 
                        required=False, help="if calulate the pearson correlation coefficient.")
    parser.add_argument('-r2', '--r2_score', type=bool, default=False, 
                        required=False, help="if calulate the r2 score.")
    parser.add_argument('-mae', '--mean_absolute_error', type=bool, default=False, 
                        required=False, help="if calulate the mean absolute error.")
    parser.add_argument('-rmse', '--root_mean_squared_error', type=bool, default=False, 
                        required=False, help="if calulate the root mean squared error.")
    return parser.parse_args(argv)


args = parser_args()


df = pd.read_csv(args.file)

if args.pearson:
    pcc, p_value = pearsonr(df[args.column_1], df[args.column_2])
    print('Pearson Correlation Coefficient:', round(pcc,3), '\tP Value:', round(p_value,3))
if args.r2_score:
    print('R2 Score:',
          round(r2_score(df[args.column_1], df[args.column_2]),3))
if args.mean_absolute_error:
    print('Mean Absolute Error:',
          round(mean_absolute_error(df[args.column_1], df[args.column_2]),3))
if args.root_mean_squared_error:
    print('Root Mean Squared Error:',
          round(np.sqrt(mean_squared_error(df[args.column_1], df[args.column_2])),3))
