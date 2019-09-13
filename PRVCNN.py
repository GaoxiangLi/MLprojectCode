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

    parser.add_argument('--model_name',  default='PRVCNN',
                        help='The name saved for model and checkpoint')

    parser.add_argument('--evaluation', dest='evaluation', action='store_true',
                        help='Use this option for evaluate model')
    parser.set_defaults(evaluation=False)

    parser.add_argument('--eval_name', default='PRVCNN',
                        help='The name of model to be eval')

    parser.add_argument('--test_name', default='PRVCNN',
                        help='The name of model to be test')

    parser.add_argument('--test1', dest='test1', action='store_true', help='Use this option for test model1')
    parser.set_defaults(test=False)

    parser.add_argument('--test2', dest='test2', action='store_true', help='Use this option for test model1')
    parser.set_defaults(test=False)

    parser.add_argument('--test3', dest='test3', action='store_true', help='Use this option for test model1')
    parser.set_defaults(test=False)

    parser.add_argument('--test4', dest='test4', action='store_true', help='Use this option for test model1')
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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PRVCNN')
    args = parse_arguments(parser)
    main(args)
