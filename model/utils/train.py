import argparse
import json
import pandas as pd
from clf import binary_clf


def parser_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameters for the script config.")
    parser.add_argument('-f1', '--modeling_file', type=str, help="the modeling file")
    parser.add_argument('-f2', '--test_file', type=str, help="the test file")
    parser.add_argument('-t', '--target_column', type=str, help="the target column")
    parser.add_argument('-i', '--input_feature_file', type=str, help="the feature file")
    parser.add_argument('-a', '--algorithm', type=str, default='NaiveBayes',
                        required=False, help="the algorithm")
    parser.add_argument('-io', '--if_opt', type=bool, default=False,
                        required=False, help="if opt")
    parser.add_argument('-s', '--score', type=str, default='f1_score',
                        required=False, help="opt metric.")
    parser.add_argument('-od', '--opt_direction', type=str, default='maximize',
                        required=False, help="opt direction.")
    parser.add_argument('-wd', '--weights_dir', type=str, help="weights dir for keeping the weights.")
    return parser.parse_args(argv)

args = parser_args()

modeling_file = args.modeling_file
test_file = args.test_file
target_column = args.target_column
input_feature_file = args.input_feature_file


df_train = pd.read_csv(modeling_file)
df_test = pd.read_csv(test_file)

with open(input_feature_file, 'r') as file:
    input_features_list = json.load(file)

train_X = pd.DataFrame(df_train, columns=[column for column in df_train.columns if column in input_features_list]).values
train_y = df_train[target_column].values
test_X = pd.DataFrame(df_test, columns=[column for column in df_train.columns if column in input_features_list]).values
test_y = df_test[target_column].values

print('train feature shape: {}\ntrain target shape: {}'.format(train_X.shape, train_y.shape))
print('test feature shape: {}\ntest target shape: {}'.format(test_X.shape, test_y.shape))

model = binary_clf(algorithm=args.algorithm, if_opt=args.if_opt,
            opt_metric=args.score, opt_direction=args.opt_direction, random_state=42, save_model_num=5,
            weights_dir=args.weights_dir)
model.train(train_X, train_y, test_X, test_y)
