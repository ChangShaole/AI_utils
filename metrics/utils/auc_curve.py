import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc
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
    parser.add_argument('-tl', '--threshold_lookup', type=bool, default=False,
                        required=False, help="if print the threshold lookup table. (default: False)")
    parser.add_argument('-di', '--drop_intermediate', type=bool, default=False,
                        required=False, help="if drop intermediate. (default: False)")
    parser.add_argument('-t', '--title', type=str, default='Receiver Operating Characteristic',
                        required=False, help="the out figure title. (default: Receiver Operating Characteristic)")
    parser.add_argument('-o', '--output', type=str, default='ROC-AUC.png',
                        required=False, help="the out figure name. (default: ROC-AUC.png)")
    return parser.parse_args(argv)


args = parser_args()


df = pd.read_csv(args.file)
if args.drop_intermediate:
    fpr, tpr, thresholds = roc_curve(df[args.column_1], df[args.column_2], drop_intermediate=True)
else:
    fpr, tpr, thresholds = roc_curve(df[args.column_1], df[args.column_2], drop_intermediate=False)
roc_auc = auc(fpr, tpr)


if args.threshold_lookup:
    ftpr_threshold = pd.DataFrame()
    ftpr_threshold['fpr'] = fpr
    ftpr_threshold['tpr'] = tpr
    ftpr_threshold['threshold'] = thresholds
    print(ftpr_threshold)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(args.title)
plt.legend(loc="lower right")
plt.savefig(args.output)
