import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def parser_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameters for the script config.")
    parser.add_argument('-f', '--file', type=str, 
                        help="predict file")
    parser.add_argument('-c1', '--column_1', type=str,
                        help="ground truth label column")
    parser.add_argument('-c2', '--column_2', type=str, 
                        help="pred column.")
    parser.add_argument('-xa', '--x_axis', type=str, default='x-axis',
                        required=False, help="the x-axis name. (default: x-axis)")
    parser.add_argument('-ya', '--y_axis', type=str, default='y-axis',
                        required=False, help="the y-axis name. (default: y-axis)")
    parser.add_argument('-xl', '--x_label', type=str, default='class-0,class-1',
                        required=False, help="the x-label name. split with , (default: class-0,class-1)")
    parser.add_argument('-yl', '--y_label', type=str, default='class-0,class-1',
                        required=False, help="the y-label name. split with , (default: class-0,class-1)")
    parser.add_argument('-t', '--title', type=str, default='Confusion Matrix',
                        required=False, help="the out figure title. (default: Confusion Matrix)")
    parser.add_argument('-o', '--output', type=str, default='confusion-matrix.png',
                        required=False, help="the out figure name. (default: confusion-matrix.png)")
    return parser.parse_args(argv)


args = parser_args()


sns.set()

df = pd.read_csv(args.file)

y_true = np.array(df[args.column_1])
y_pred = np.array(df[args.column_2])

x_label = args.x_label.split(',')
y_label = args.y_label.split(',')


fig = plt.figure()
ax = fig.add_subplot(111)
conf_mat = confusion_matrix(y_true, y_pred)

sns.heatmap(conf_mat,annot=True,xticklabels=x_label,yticklabels=y_label, fmt= '.20g')
ax.set_xlabel(args.x_axis)
ax.set_ylabel(args.y_axis)
plt.title(args.title)
plt.savefig(args.output)
