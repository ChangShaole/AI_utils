import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def parser_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameters for the script config.")
    parser.add_argument('-f', '--file', type=str, 
                        help="predict file")
    parser.add_argument('-c1', '--column_1', type=str,
                        help="ground truth label column")
    parser.add_argument('-c2', '--column_2', type=str, 
                        help="pred column.")
    return parser.parse_args(argv)


args = parser_args()


df = pd.read_csv(args.file)

y_true = np.array(df[args.column_1])
y_pred = np.array(df[args.column_2])

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
