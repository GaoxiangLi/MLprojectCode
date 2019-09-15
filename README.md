# PRVCNN
Predicting regulatory variants with Convolutional Neural Network

Gaoxiang Li

Auburn University

Advisor:Li Chen, Xiao Qin 

How to use:

(1)After download and unzip the file,  run:   python PRVCNN.py   in this directory.
command:  python PRVCNN.py --h (for help to show all command line instruction)

1.training:  --train       training model
             --model_name  the model name you want to save default:PRVCNN
             
             Path for training input
             --training_feature  default='./data/training_feature_all.csv'
             --training_label    default='./data/training_label_all.csv'

2.testing:   --test       testing process
             --test_name    the model name you want to test default:PRVCNN
             
             Path for testing input:
             --testing_feature   default='./data/t1_imbanlance_feature.csv'
             --testing_label     default='./data/t1_imbanlance_label.csv'
             --result_score_file default='score1.csv' Result score file name you want to save
             
3.evaluation --evaluation   evaluation model
             --eval_name    the model name you want to eval default:PRVCNN
             --eval_dataset default=0 The compared dataset to be evaluated,1 for allelic imbalance'
                             '2 for eQTL, 3 for dsQTL, 4 for validated
             Path for eval input
             --eval_score_file  default=default='./result/score1.csv' The score file to evaluate
             --eval_true_label  default=default='./data/t1_imbanlance_label.csv'   The true label to evaluate
             --eval_name        default='Your method'  The name of curve showed in figure
             
4.Parameter  --batch_size default=16
             --max_epoch  default=2000
             --learning_rate default=0.0001
             --learning_rate_decay default=0.99
             --dropout_rate default=1
             --L2_regularizer default=0.05
             --kernel_size default=5


(2)Fpr other test:
Run: python other_test.py
             --SVM   SVM method
             --RF    Random Forest method
             --training_feature    default='./data/training_feature_all.csv'
             --training_label      default='./data/ML_training_label.csv'    (only one column)
             --testing_feature     default='./data/t1_imbanlance_feature.csv'
             --testing_label       default='./data/t1_imbanlance_label.csv'
             --result_score_file   default='score1_svm.csv'   Result score file name to be saved

Prediction Score：

file score1.csv Allelic imbalance

file socre2.csv eQTL

file socre3.csv dsQTL

file score4.csv validated regulatory

Training code: PRVCNN_train

Testing code: PRVCNN_test1-4

notes:input must be reshaped and change the string feature to one-hot code


