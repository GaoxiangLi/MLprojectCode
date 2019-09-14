import numpy as np
import argparse


def main(args):
    training = args.train
    evaluation = args.evaluation
    testing1 = args.test1
    testing2 = args.test2
    testing3 = args.test3
    testing4 = args.test4
    if training:
        import PRVCNN_train
        PRVCNN_train.train(args)
    if evaluation:
        import PRVCNN_eval
        PRVCNN_eval.eval(args)
    if testing1:
        import PRVCNN_test1
        PRVCNN_test1.test1(args)
    if testing2:
        import PRVCNN_test2
        PRVCNN_test2.test2(args)
    if testing3:
        import PRVCNN_test3
        PRVCNN_test3.test3(args)
    if testing4:
        import PRVCNN_test4
        PRVCNN_test4.test4(args)


def parse_arguments(parser):
    parser.add_argument('--train', dest='train', action='store_true', help='Use this option for train model')
    parser.set_defaults(train=False)

    parser.add_argument('--model_name', default='PRVCNN',
                        help='The name saved for model and checkpoint')

    parser.add_argument('--test_name', default='PRVCNN',
                        help='The name of model to be test')

    parser.add_argument('--test1', dest='test1', action='store_true', help='Use this option for test model1')
    parser.set_defaults(test=False)

    parser.add_argument('--test2', dest='test2', action='store_true', help='Use this option for test model2')
    parser.set_defaults(test=False)

    parser.add_argument('--test3', dest='test3', action='store_true', help='Use this option for test model3')
    parser.set_defaults(test=False)

    parser.add_argument('--test4', dest='test4', action='store_true', help='Use this option for test model4')
    parser.set_defaults(test=False)

    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size for training')

    parser.add_argument('--max_epoch', type=int, default=2000,
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

    parser.add_argument('--training_feature', default='./data/training_feature_all.csv',
                        help='Training feature file')

    parser.add_argument('--training_label', default='./data/training_label_all.csv',
                        help='Training label file')

    parser.add_argument('--testing1_feature', default='./data/t1_imbanlance_feature.csv',
                        help='Testing1 feature file')

    parser.add_argument('--testing1_label', default='./data/t1_imbanlance_label.csv',
                        help='Testing1 label file')

    parser.add_argument('--testing2_feature', default='./data/t2_eQTL_feature.csv',
                        help='Testing2 feature file')

    parser.add_argument('--testing2_label', default='./data/t2_eQTL_label.csv',
                        help='Testing2 label file')

    parser.add_argument('--testing3_feature', default='./data/t3_dsQTL_feature.csv',
                        help='Testing3 feature file')

    parser.add_argument('--testing3_label', default='./data/t3_dsQTL_label.csv',
                        help='Testing3 label file')

    parser.add_argument('--testing4_feature', default='./data/t4_validated_feature.csv',
                        help='Testing4 feature file')

    parser.add_argument('--testing4_label', default='./data/t4_validated_label.csv',
                        help='Testing4 label file')

    parser.add_argument('--evaluation', dest='evaluation', action='store_true',
                        help='Use this option for evaluate model')

    parser.add_argument('--eval_score_file', default='./result/train_score.csv',
                        help='The score file to evaluate')

    parser.add_argument('--eval_true_label', default='./data/test_label3.csv',
                        help='The true label to evaluate')

    parser.add_argument('--eval_dataset', type=int, default=0,
                        help='The compared dataset to be evaluated, 0 for training set, 1 for allelic imbalance'
                             '2 for eQTL, 3 for dsQTL, 4 for validated')

    parser.set_defaults(evaluation=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PRVCNN')
    args = parse_arguments(parser)
    main(args)
