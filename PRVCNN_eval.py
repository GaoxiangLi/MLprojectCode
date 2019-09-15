import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 16
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
NUM_EXAMPLE = 60977


def eval(args):
    eval_score_file = args.eval_score_file
    eval_true_label = args.eval_true_label
    eval_dataset = args.eval_dataset
    y_prob = np.loadtxt("%s" % (eval_score_file), delimiter=',')
    y_true = np.loadtxt("%s" % (eval_true_label), delimiter=',')
    y_true = np.argmax(y_true, axis=1)
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    if eval_dataset == 1:
        print("Draw roc curve for allelic imbalance dataset")
        all_label = np.loadtxt("./data/t1_label.csv", delimiter=',')
        y_pred2 = np.loadtxt('./data/t1_Fathmm_MKL.csv', delimiter=',')
        y_pred3 = np.loadtxt('./data/t1_GWAS3D.csv', delimiter=',')
        y_pred4 = np.loadtxt('./data/t1_Funseq.csv', delimiter=',')
        y_pred5 = np.loadtxt('./data/t1_Funseq2.csv', delimiter=',')
        y_pred6 = np.loadtxt('./data/t1_GWAVA_TSS.csv', delimiter=',')
        y_pred7 = np.loadtxt('./data/t1_SuRFR.csv', delimiter=',')
        y_pred8 = np.loadtxt('./data/t1_DANN.csv', delimiter=',')
        y_pred9 = np.loadtxt('./data/t1_CADD.csv', delimiter=',')
        y_pred10 = np.loadtxt('./data/t1_Composite.csv', delimiter=',')
        #
        fpr2, tpr2, thresholds2 = roc_curve(all_label, y_pred2, pos_label=1)
        fpr3, tpr3, thresholds3 = roc_curve(all_label, y_pred3, pos_label=1)
        fpr4, tpr4, thresholds4 = roc_curve(all_label, y_pred4, pos_label=1)
        fpr5, tpr5, thresholds5 = roc_curve(all_label, y_pred5, pos_label=1)
        fpr6, tpr6, thresholds6 = roc_curve(all_label, y_pred6, pos_label=1)
        fpr7, tpr7, thresholds7 = roc_curve(all_label, y_pred7, pos_label=1)
        fpr8, tpr8, thresholds8 = roc_curve(all_label, y_pred8, pos_label=1)
        fpr9, tpr9, thresholds9 = roc_curve(all_label, y_pred9, pos_label=1)
        fpr10, tpr10, thresholds9 = roc_curve(all_label, y_pred10, pos_label=1)
        #
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)
        roc_auc4 = auc(fpr4, tpr4)
        roc_auc5 = auc(fpr5, tpr5)
        roc_auc6 = auc(fpr6, tpr6)
        roc_auc7 = auc(fpr7, tpr7)
        roc_auc8 = auc(fpr8, tpr8)
        roc_auc9 = auc(fpr9, tpr9)
        roc_auc10 = auc(fpr10, tpr10)

        # # Plot ROC curve
        name = args.eval_name
        plt.plot(fpr, tpr, label='%s (area = %0.3f)' % (name, roc_auc))
        plt.plot(fpr2, tpr2, label='Fathmm-MKL (area = %0.3f)' % roc_auc2)
        plt.plot(fpr3, tpr3, label='GWAS3D (area = %0.3f)' % roc_auc3)
        plt.plot(fpr4, tpr4, label='Funseq (area = %0.3f)' % roc_auc4)
        plt.plot(fpr5, tpr5, label='Funseq2 (area = %0.3f)' % roc_auc5)
        plt.plot(fpr6, tpr6, label='GWAVA_TSS (area = %0.3f)' % roc_auc6)
        plt.plot(fpr7, tpr7, label='SuRFR (area = %0.3f)' % roc_auc7)
        plt.plot(fpr8, tpr8, label='DANN (area = %0.3f)' % roc_auc8)
        plt.plot(fpr9, tpr9, label='CADD (area = %0.3f)' % roc_auc9)
        plt.plot(fpr10, tpr10, label='Composite (area = %0.3f)' % roc_auc10)
        # plt.plot(fpr11, tpr11, label='Combined (area = %0.3f)' % roc_auc11)

        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('allelc imbalanced')
        plt.legend(loc="lower right")
        plt.show()
        print("finished")
    if eval_dataset == 2:
        print("Draw roc curve for eQTL dataset")
        all_label = np.loadtxt("./data/t2_label.csv", delimiter=',')
        y_pred2 = np.loadtxt('./data/t2_Fathmm_MKL.csv', delimiter=',')
        y_pred3 = np.loadtxt('./data/t2_GWAS3D.csv', delimiter=',')
        y_pred4 = np.loadtxt('./data/t2_Funseq.csv', delimiter=',')
        y_pred5 = np.loadtxt('./data/t2_Funseq2.csv', delimiter=',')
        y_pred6 = np.loadtxt('./data/t2_GWAVA_TSS.csv', delimiter=',')
        y_pred7 = np.loadtxt('./data/t2_SuRFR.csv', delimiter=',')
        y_pred8 = np.loadtxt('./data/t2_DANN.csv', delimiter=',')
        y_pred9 = np.loadtxt('./data/t2_CADD.csv', delimiter=',')
        y_pred10 = np.loadtxt('./data/t2_Composite.csv', delimiter=',')
        #
        fpr2, tpr2, thresholds2 = roc_curve(all_label, y_pred2, pos_label=1)
        fpr3, tpr3, thresholds3 = roc_curve(all_label, y_pred3, pos_label=1)
        fpr4, tpr4, thresholds4 = roc_curve(all_label, y_pred4, pos_label=1)
        fpr5, tpr5, thresholds5 = roc_curve(all_label, y_pred5, pos_label=1)
        fpr6, tpr6, thresholds6 = roc_curve(all_label, y_pred6, pos_label=1)
        fpr7, tpr7, thresholds7 = roc_curve(all_label, y_pred7, pos_label=1)
        fpr8, tpr8, thresholds8 = roc_curve(all_label, y_pred8, pos_label=1)
        fpr9, tpr9, thresholds9 = roc_curve(all_label, y_pred9, pos_label=1)
        fpr10, tpr10, thresholds9 = roc_curve(all_label, y_pred10, pos_label=1)
        #
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)
        roc_auc4 = auc(fpr4, tpr4)
        roc_auc5 = auc(fpr5, tpr5)
        roc_auc6 = auc(fpr6, tpr6)
        roc_auc7 = auc(fpr7, tpr7)
        roc_auc8 = auc(fpr8, tpr8)
        roc_auc9 = auc(fpr9, tpr9)
        roc_auc10 = auc(fpr10, tpr10)

        # print("thread2:", thresholds2, len(thresholds2))
        # print("thread_ours:", thresholds, len(thresholds))
        # # Plot ROC curve
        name = args.eval_name
        plt.plot(fpr, tpr, label='%s (area = %0.3f)' % (name, roc_auc))
        plt.plot(fpr2, tpr2, label='Fathmm-MKL (area = %0.3f)' % roc_auc2)
        plt.plot(fpr3, tpr3, label='GWAS3D (area = %0.3f)' % roc_auc3)
        plt.plot(fpr4, tpr4, label='Funseq (area = %0.3f)' % roc_auc4)
        plt.plot(fpr5, tpr5, label='Funseq2 (area = %0.3f)' % roc_auc5)
        plt.plot(fpr6, tpr6, label='GWAVA_TSS (area = %0.3f)' % roc_auc6)
        plt.plot(fpr7, tpr7, label='SuRFR (area = %0.3f)' % roc_auc7)
        plt.plot(fpr8, tpr8, label='DANN (area = %0.3f)' % roc_auc8)
        plt.plot(fpr9, tpr9, label='CADD (area = %0.3f)' % roc_auc9)
        plt.plot(fpr10, tpr10, label='Composite (area = %0.3f)' % roc_auc10)
        # plt.plot(fpr11, tpr11, label='Combined (area = %0.3f)' % roc_auc11)

        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('eQTL')
        plt.legend(loc="lower right")
        plt.show()
        print("finished")
    if eval_dataset == 3:
        print("Draw roc curve for dsQTL dataset")
        all_label = np.loadtxt("./data/t3_label.csv", delimiter=',')
        y_pred2 = np.loadtxt('./data/t3_Fathmm_MKL.csv', delimiter=',')
        y_pred3 = np.loadtxt('./data/t3_GWAS3D.csv', delimiter=',')
        y_pred4 = np.loadtxt('./data/t3_Funseq.csv', delimiter=',')
        y_pred5 = np.loadtxt('./data/t3_Funseq2.csv', delimiter=',')
        y_pred6 = np.loadtxt('./data/t3_GWAVA_TSS.csv', delimiter=',')
        y_pred7 = np.loadtxt('./data/t3_SuRFR.csv', delimiter=',')
        y_pred8 = np.loadtxt('./data/t3_DANN.csv', delimiter=',')
        y_pred9 = np.loadtxt('./data/t3_CADD.csv', delimiter=',')
        y_pred10 = np.loadtxt('./data/t3_Composite.csv', delimiter=',')
        #
        fpr2, tpr2, thresholds2 = roc_curve(all_label, y_pred2, pos_label=1)
        fpr3, tpr3, thresholds3 = roc_curve(all_label, y_pred3, pos_label=1)
        fpr4, tpr4, thresholds4 = roc_curve(all_label, y_pred4, pos_label=1)
        fpr5, tpr5, thresholds5 = roc_curve(all_label, y_pred5, pos_label=1)
        fpr6, tpr6, thresholds6 = roc_curve(all_label, y_pred6, pos_label=1)
        fpr7, tpr7, thresholds7 = roc_curve(all_label, y_pred7, pos_label=1)
        fpr8, tpr8, thresholds8 = roc_curve(all_label, y_pred8, pos_label=1)
        fpr9, tpr9, thresholds9 = roc_curve(all_label, y_pred9, pos_label=1)
        fpr10, tpr10, thresholds9 = roc_curve(all_label, y_pred10, pos_label=1)
        #
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)
        roc_auc4 = auc(fpr4, tpr4)
        roc_auc5 = auc(fpr5, tpr5)
        roc_auc6 = auc(fpr6, tpr6)
        roc_auc7 = auc(fpr7, tpr7)
        roc_auc8 = auc(fpr8, tpr8)
        roc_auc9 = auc(fpr9, tpr9)
        roc_auc10 = auc(fpr10, tpr10)

        # # Plot ROC curve
        name = args.eval_name
        plt.plot(fpr, tpr, label='%s (area = %0.3f)' % (name, roc_auc))
        plt.plot(fpr2, tpr2, label='Fathmm-MKL (area = %0.3f)' % roc_auc2)
        plt.plot(fpr3, tpr3, label='GWAS3D (area = %0.3f)' % roc_auc3)
        plt.plot(fpr4, tpr4, label='Funseq (area = %0.3f)' % roc_auc4)
        plt.plot(fpr5, tpr5, label='Funseq2 (area = %0.3f)' % roc_auc5)
        plt.plot(fpr6, tpr6, label='GWAVA_TSS (area = %0.3f)' % roc_auc6)
        plt.plot(fpr7, tpr7, label='SuRFR (area = %0.3f)' % roc_auc7)
        plt.plot(fpr8, tpr8, label='DANN (area = %0.3f)' % roc_auc8)
        plt.plot(fpr9, tpr9, label='CADD (area = %0.3f)' % roc_auc9)
        plt.plot(fpr10, tpr10, label='Composite (area = %0.3f)' % roc_auc10)
        # plt.plot(fpr11, tpr11, label='Combined (area = %0.3f)' % roc_auc11)

        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('dsQTL')
        plt.legend(loc="lower right")
        plt.show()
        print("finished")
    if eval_dataset == 4:
        print("Draw roc curve for validated dataset")
        all_label = np.loadtxt("./data/t4_label.csv", delimiter=',')
        y_pred2 = np.loadtxt('./data/t4_Fathmm_MKL.csv', delimiter=',')
        y_pred3 = np.loadtxt('./data/t4_GWAS3D.csv', delimiter=',')
        y_pred4 = np.loadtxt('./data/t4_Funseq.csv', delimiter=',')
        y_pred5 = np.loadtxt('./data/t4_Funseq2.csv', delimiter=',')
        y_pred6 = np.loadtxt('./data/t4_GWAVA_TSS.csv', delimiter=',')
        y_pred7 = np.loadtxt('./data/t4_SuRFR.csv', delimiter=',')
        y_pred8 = np.loadtxt('./data/t4_DANN.csv', delimiter=',')
        y_pred9 = np.loadtxt('./data/t4_CADD.csv', delimiter=',')
        y_pred10 = np.loadtxt('./data/t4_Composite.csv', delimiter=',')
        #
        fpr2, tpr2, thresholds2 = roc_curve(all_label, y_pred2, pos_label=1)
        fpr3, tpr3, thresholds3 = roc_curve(all_label, y_pred3, pos_label=1)
        fpr4, tpr4, thresholds4 = roc_curve(all_label, y_pred4, pos_label=1)
        fpr5, tpr5, thresholds5 = roc_curve(all_label, y_pred5, pos_label=1)
        fpr6, tpr6, thresholds6 = roc_curve(all_label, y_pred6, pos_label=1)
        fpr7, tpr7, thresholds7 = roc_curve(all_label, y_pred7, pos_label=1)
        fpr8, tpr8, thresholds8 = roc_curve(all_label, y_pred8, pos_label=1)
        fpr9, tpr9, thresholds9 = roc_curve(all_label, y_pred9, pos_label=1)
        fpr10, tpr10, thresholds9 = roc_curve(all_label, y_pred10, pos_label=1)
        #
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)
        roc_auc4 = auc(fpr4, tpr4)
        roc_auc5 = auc(fpr5, tpr5)
        roc_auc6 = auc(fpr6, tpr6)
        roc_auc7 = auc(fpr7, tpr7)
        roc_auc8 = auc(fpr8, tpr8)
        roc_auc9 = auc(fpr9, tpr9)
        roc_auc10 = auc(fpr10, tpr10)

        # # Plot ROC curve
        name = args.eval_name
        plt.plot(fpr, tpr, label='%s (area = %0.3f)' % (name, roc_auc))
        plt.plot(fpr2, tpr2, label='Fathmm-MKL (area = %0.3f)' % roc_auc2)
        plt.plot(fpr3, tpr3, label='GWAS3D (area = %0.3f)' % roc_auc3)
        plt.plot(fpr4, tpr4, label='Funseq (area = %0.3f)' % roc_auc4)
        plt.plot(fpr5, tpr5, label='Funseq2 (area = %0.3f)' % roc_auc5)
        plt.plot(fpr6, tpr6, label='GWAVA_TSS (area = %0.3f)' % roc_auc6)
        plt.plot(fpr7, tpr7, label='SuRFR (area = %0.3f)' % roc_auc7)
        plt.plot(fpr8, tpr8, label='DANN (area = %0.3f)' % roc_auc8)
        plt.plot(fpr9, tpr9, label='CADD (area = %0.3f)' % roc_auc9)
        plt.plot(fpr10, tpr10, label='Composite (area = %0.3f)' % roc_auc10)
        # plt.plot(fpr11, tpr11, label='Combined (area = %0.3f)' % roc_auc11)

        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('validated regulatory')
        plt.legend(loc="lower right")
        plt.show()
        print("finished")
