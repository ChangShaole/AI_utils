import pandas as pd
import numpy as np
import argparse
import pickle
import os
import json


def parser_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameters for the rf model predicting.")
    parser.add_argument('-et', '--test_file', type=str, help="file incluing the internal test dataset. (csv)")
    parser.add_argument('-w', '--weights', type=str, help="weights file. (pkl, required)")
    parser.add_argument('-i', '--input_feature_file', type=str, help="the feature file")
    parser.add_argument('-rf', '--results_folder', type=str, default='res',
                        required=False, help="folder for keeping the result.")
    parser.add_argument('-r', '--results', type=str, default='pred.csv',
                        required=False, help="result file name.")
    parser.add_argument('-ip', '--if_pred_proba', type=bool, default=False,
                        required=False, help="the generate the proba prediction")
    parser.add_argument('-c1', '--column_1', type=str, default='pred_proba',
                        required=False, help="the pred proba column")
    parser.add_argument('-c', '--column', type=str, default='pred_class',
                        required=False, help="the pred class column")
    return parser.parse_args(argv)


args = parser_args()

test_df = pd.read_csv(args.test_file)


with open(args.input_feature_file, 'r') as f:
    json_list = f.read()

input_feature_list = json.loads(json_list)

test_input_df = pd.DataFrame(test_df, columns=[header for header in test_df.columns if header in input_feature_list])
np_test_X = test_input_df.values.astype(float)

model = pickle.load(open(args.weights, "rb"))

if args.if_pred_proba:
    pred_proba = model.predict_proba(np_test_X)[:,1]
    test_df[args.column_1] = pd.Series(pred_proba)


pred_class = model.predict(np_test_X)
test_df[args.column] = pd.Series(pred_class)
out_path = os.path.join(args.results_folder, args.results)
print('writing the results to {}'.format(out_path))
test_df.to_csv(out_path, header=True, index=None)
