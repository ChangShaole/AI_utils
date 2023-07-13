import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="script config")
    parser.add_argument('-f','--file', type=str,
                        required=True, help="the under-processing file.")
    parser.add_argument('-ci','--column_inputs', type=str,
                        required=True, help="the input columns (split with ,)")
    parser.add_argument('-ct','--column_targets', type=str,
                        required=True, help="the target columns (split with ,)")
    parser.add_argument('-o','--output', type=str, default='formula',
                        required=False, help= "the output file.")
    return parser.parse_args(argv)

args = parse_args()


df = pd.read_csv(args.file)
column_inputs = args.column_inputs.split(',')
column_targets = args.column_targets.split(',')

model = LinearRegression()
X = pd.DataFrame(df, columns=column_inputs).values
Y = pd.DataFrame(df, columns=column_targets).values

model.fit(X, Y)

with open(args.output, 'w') as file:
    for index1, target in enumerate(column_targets):
        for index2, input in enumerate(column_inputs):
            print(round(model.coef_[index1][index2],3), '*', input, file=file)
        print(round(model.intercept_[index1],3), file=file)
        print('= {}'.format(target), file=file)
