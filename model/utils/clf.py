import pickle
import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import product
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score


class binary_clf(object):
    def __init__(self, algorithm: str,  weights_dir: str, save_model_num: int,
                 if_opt: bool, opt_metric: str, opt_direction: str, random_state:None):
        
        self.algorithm = algorithm
        self.if_opt = if_opt
        self.weights_dir = weights_dir
        self.save_model_num = save_model_num
        self.opt_metric = opt_metric
        self.opt_direction = opt_direction


        if self.algorithm == "RandomForest":
            self.model = RandomForestClassifier(random_state=random_state)
            if self.if_opt:
                self.param_grid = {
                    "n_estimators": [50, 100, 200, 500, 800, 1000],
                    "max_depth": [None, 3, 5, 10, 20],
                    "min_samples_leaf": [1, 3, 5, 7, 9],
                    "max_features": ["sqrt", "log2", None],
                    "criterion":["gini", "entropy", "log_loss"],
                    "class_weight": [None, "balanced", "balanced_subsample"]
                }
        elif self.algorithm == "XGBoost":
            self.model = xgb.XGBClassifier(random_state=random_state)
            if self.if_opt:
                self.param_grid = {
                    "n_estimators": [50, 100, 200, 500, 800, 1000],
                    "max_depth": [None, 3, 5, 7, 9],
                    "subsample": [0.6, 0.7, 0.8, 0.9],
                    "objective": ['binary:logistic', 'binary:hinge'],
                    "use_label_encoder": [False]
                }
        elif self.algorithm == "MLP":
            self.model = MLPClassifier(random_state=random_state)
            if self.if_opt:
                self.param_grid = {
                    "hidden_layer_sizes": [(128,64), (128,64,32), (128,64,16), (256,128,64), (512,128,32)],
                    "activation": ['logistic', 'tanh', 'relu'],
                    "alpha": [0.0001, 0.001, 0.01], # L2 regularization term
                    "learning_rate": ['constant', 'invscaling', 'adaptive'],
                    "learning_rate_init": [0.001, 0.0001],
                    "max_iter": [200, 500, 800, 1000, 3000]
                }
        elif self.algorithm == "KNN":
            self.model = KNeighborsClassifier()
            if self.if_opt:
                self.param_grid = {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                }
        elif self.algorithm == "SVM":
            self.model = SVC(random_state=random_state, probability=True)
            if self.if_opt:
                self.param_grid = {
                    "C": [1, 10, 100],
                    "kernel": ["linear", "rbf"]
                }
        elif self.algorithm == "NaiveBayes":
            self.model = GaussianNB()
            if self.if_opt:
                self.param_grid = None  # No hyperparameters to tune for Naive Bayes
        else:
            raise ValueError("Invalid algorithm name!")

    def train(self, train_X: np.ndarray, train_y: np.ndarray, valid_X: None, valid_y: None):
        if self.if_opt and (self.param_grid is not None):
            print('performing {} parameter optimization ...'.format(self.algorithm))
            para_list = []
            combination_list = []
            for key in self.param_grid.keys():
                para_list.append(key)
                combination_list.append(self.param_grid[key])
            combination = list(product(*combination_list))

            best_model_dict = {}
            for combin in tqdm(combination):
                para_dict_ = {}
                for para, value in zip(para_list, combin):
                    para_dict_[para] = value

                self.model.set_params(**para_dict_)
                self.model.fit(train_X, train_y)

                if self.opt_metric == 'accuracy':
                    valid_pred_class = self.predict(valid_X)
                    score = accuracy_score(valid_y, valid_pred_class)
                elif self.opt_metric == 'auc':
                    valid_pred_proba = self.predict_proba(valid_X)
                    fpr, tpr, _ = roc_curve(valid_y, valid_pred_proba)
                    score = auc(fpr, tpr)
                elif self.opt_metric == 'f1_score':
                    valid_pred_class = self.predict(valid_X)
                    score = 1
                    for label in np.unique(valid_y):
                        single_f1_score = f1_score(valid_y, valid_pred_class, pos_label=label)
                        score *= single_f1_score
                else:
                    raise ValueError("Invalid metric name!")
                
                if len(best_model_dict) < self.save_model_num:
                    model_list = ['{}_{}'.format(self.algorithm, i) for i in range(1, self.save_model_num+1)]
                    for model in model_list:
                        if model not in best_model_dict.keys():
                            best_model_dict[model] = score
                            self.save_model(os.path.join(self.weights_dir, model), score)
                            break
                else:
                    if self.opt_direction == 'maximize':
                        if score > min(best_model_dict.values()):
                            min_value = np.inf
                            for key in best_model_dict.keys():
                                if best_model_dict[key] < min_value:
                                    selected_key = key
                                    min_value = best_model_dict[key]
                            best_model_dict[selected_key] = score
                            self.save_model(os.path.join(self.weights_dir, selected_key), score)
                    elif self.opt_direction == 'minimize':
                        if score < max(best_model_dict.values()):
                            max_value = -np.inf
                            for key in best_model_dict.keys():
                                if best_model_dict[key] > max_value:
                                    selected_key = key
                                    max_value = best_model_dict[key]
                            best_model_dict[selected_key] = score
                            self.save_model(os.path.join(self.weights_dir, selected_key), score)
                    else:
                        raise ValueError("Invalid optimization direction!")
        else:
            if self.if_opt == False:
                print('{} single model fitting ...'.format(self.algorithm))
            elif self.param_grid is None:
                print('{} has no parameter can be optimization!'.format(self.algorithm))

            self.model.fit(train_X, train_y)

            if self.opt_metric == 'accuracy':
                valid_pred_class = self.predict(valid_X)
                score = accuracy_score(valid_y, valid_pred_class)
            elif self.opt_metric == 'auc':
                valid_pred_proba = self.predict_proba(valid_X)
                fpr, tpr, _ = roc_curve(valid_y, valid_pred_proba)
                score = auc(fpr, tpr)
            elif self.opt_metric == 'f1_score':
                valid_pred_class = self.predict(valid_X)
                score = 1
                for label in np.unique(valid_y):
                    single_f1_score = f1_score(valid_y, valid_pred_class, pos_label=label)
                    score *= single_f1_score
            else:
                raise ValueError("Invalid metric name!")
            
            self.save_model(os.path.join(self.weights_dir, self.algorithm), score)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)[:,1]

    def save_model(self, filename: str, score: float):
        with open(filename+'.pkl', "wb") as file:
            pickle.dump(self.model, file)
        with open(filename+'.json', 'w') as file:
            json.dump(self.model.get_params(), file, indent=4)
        with open(filename+'_opt_score', 'w') as file:
            dict_ = {}
            dict_[self.opt_metric] = score
            json.dump(dict_, file, indent=4)


if __name__ == '__main__':
    import argparse
    def parser_args(argv=None):
        parser = argparse.ArgumentParser(
            description="Parameters for the script config.")
        parser.add_argument('-a', '--algorithm', type=str, default='NaiveBayes',
                            required=False, help="the algorithm")
        parser.add_argument('-io', '--if_opt', type=bool, default=False,
                            required=False, help="if opt")
        parser.add_argument('-s', '--score', type=str, default='f1_score',
                            required=False, help="opt metric.")
        parser.add_argument('-od', '--opt_direction', type=str, default='maximize',
                            required=False, help="opt direction.")
        return parser.parse_args(argv)


    args = parser_args()

    df = pd.read_csv('../data/exp_binary_classification.csv')

    train_df = df[df['set']=='train'].drop('set', axis=1).copy()
    test_df = df[df['set']=='test'].drop('set', axis=1).copy()
    train_X = train_df.drop('Y', axis=1).values
    train_Y = train_df['Y'].values
    test_X = test_df.drop('Y', axis=1).values
    test_Y = test_df['Y'].values

    model = binary_clf(algorithm=args.algorithm, if_opt=args.if_opt,
                        opt_metric=args.score, opt_direction=args.opt_direction, random_state=42,
                        save_model_num=5, weights_dir='../weights')
    model.train(train_X, train_Y, test_X, test_Y)
