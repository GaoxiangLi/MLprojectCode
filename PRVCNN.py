import numpy as np
import argparse


def main(args):
    training = args.train
    evaluation = args.evaluation
    testing = args.test
    if training:
        import PRVCNN_train
        PRVCNN_train.train(args)
    if evaluation:
        import PRVCNN_eval
        PRVCNN_eval.eval(args)
    if testing:
        import PRVCNN_test
        PRVCNN_test.test(args)


def parse_arguments(parser):
    parser.add_argument('--train', dest='train', action='store_true', help='Use this option for train model')
    parser.set_defaults(train=False)

    parser.add_argument('--model_name', default='PRVCNN',
                        help='The name saved for model and checkpoint')

    parser.add_argument('--training_feature', default='./data/training_feature_all.csv',
                        help='Training feature file')

    parser.add_argument('--training_label', default='./data/training_label_all.csv',
                        help='Training label file')

    parser.add_argument('--test', dest='test', action='store_true', help='Use this option for test model')
    parser.set_defaults(test=False)

    parser.add_argument('--test_name', default='PRVCNN',
                        help='The name of model to be test')

    parser.add_argument('--testing_feature', default='./data/t1_imbanlance_feature.csv',
                        help='Testing feature file')

    parser.add_argument('--testing_label', default='./data/t1_imbanlance_label.csv',
                        help='Testing label file')

    parser.add_argument('--result_score_file', default='score1.csv',
                        help='Result score file name')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size for training')

    parser.add_argument('--max_epoch', type=int, default=20,
                        help='The max epoch for training')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='The learning rate for training')

    parser.add_argument('--learning_rate_decay', type=float, default=0.99,
                        help='The learning rate for training')

    parser.add_argument('--dropout_rate', type=float, default=1,
                        help='The dropout rate for training')

    parser.add_argument('--L2_regularizer', type=float, default=0.05,
                        help='The L2 regularizer lambda')

    parser.add_argument('--kernel_size', nargs='+', type=int, default=5,
                        help='The kernel size for convolutional layers')

    parser.add_argument('--evaluation', dest='evaluation', action='store_true',
                        help='Use this option for evaluate model')

    parser.add_argument('--eval_score_file', default='./result/score1.csv',
                        help='The score file to evaluate')

    parser.add_argument('--eval_true_label', default='./data/t1_imbanlance_label.csv',
                        help='The true label to evaluate')

    parser.add_argument('--eval_name', default='Your method',
                        help='The name of curve showed in figure')

    parser.add_argument('--eval_dataset', type=int, default=1,
                        help='The compared dataset to be evaluated,1 for allelic imbalance'
                             '2 for eQTL, 3 for dsQTL, 4 for validated')

    parser.set_defaults(evaluation=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PRVCNN')
    args = parse_arguments(parser)
    main(args)
