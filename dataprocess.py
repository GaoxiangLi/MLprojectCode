import numpy as np
import argparse


def main(args):

        import DataPreprocess
        DataPreprocess.dataprocess(args)


def parse_arguments(parser):
    parser.add_argument('--header', dest='header', action='store_true', help='If header included')
    parser.set_defaults(train=False)
    parser.add_argument('--index', dest='index', action='store_true', help='If index included')
    parser.set_defaults(train=False)
    parser.add_argument('--label', dest='label', action='store_true', help='If label included')
    parser.set_defaults(train=False)

    parser.add_argument('--filename', default='./data/dsQTL.csv',
                        help='Training feature file')

    parser.add_argument('--Consequence_index', type=int, default=0,
                        help='The Consequence column number')

    parser.add_argument('--Segway_index', type=int, default=0,
                        help='The Segway column number')

    parser.add_argument('--Label_index', type=float, default=0,
                        help='The Label column number')

    parser.add_argument('--feature_file', default='./data/dp_feature.csv',
                        help='The feature to be saved')

    parser.add_argument('--label_file', default='./data/dp_label.csv',
                        help='The label to be saved')


    parser.set_defaults(evaluation=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dataprocess.py')
    args = parse_arguments(parser)
    main(args)
