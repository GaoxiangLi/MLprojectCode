import numpy as np
import argparse


def main(args):
    SVM = args.SVM
    RF = args.RF
    if SVM:
        import SVM
        SVM.SVM(args)
    if RF:
        import RF
        RF.RF(args)


def parse_arguments(parser):
    parser.add_argument('--SVM', dest='SVM', action='store_true', help='Use this option for SVM test')
    parser.set_defaults(train=False)

    parser.add_argument('--RF', dest='RF', action='store_true', help='Use this option for Random Forest test')
    parser.set_defaults(train=False)

    parser.add_argument('--training_feature', default='./data/training_feature_all.csv',
                        help='Training feature file')

    parser.add_argument('--training_label', default='./data/ML_training_label.csv',
                        help='Training label file')

    parser.add_argument('--testing_feature', default='./data/t1_imbanlance_feature.csv',
                        help='Testing feature file')

    parser.add_argument('--testing_label', default='./data/t1_imbanlance_label.csv',
                        help='Testing label file')

    parser.add_argument('--result_score_file', default='score1_svm.csv',
                        help='Result score file name')

    parser.set_defaults(evaluation=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='other_test')
    args = parse_arguments(parser)
    main(args)
