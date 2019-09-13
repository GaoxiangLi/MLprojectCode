# PRVCNN
Predicting regulatory variants with Convolutional Neural Network

Gaoxiang Li

Auburn University

Advisor:Li Chen, Xiao Qin 

How to use:
After download and unzip the file,  run:   python PRVCNN.py   in this directory.
command:  python PRVCNN.py --h (for help to show all command line instruction)

1.training:  --train       training model
             --model_name  the model name you want to save default:PRVCNN

2.evaluation --evaluation   evaluation model
             --eval_name    the model name you want to eval default:PRVCNN

3.testing:   --test1        test for allelic imbalance dataset
             --test2        test for eQTL dataset
             --test3        test for dsQTL dataset
             --test4        test for validated regulatory dataset
             --test_name    the model name you want to test default:PRVCNN

4.Parameter  --batch_size default=16
             --max_epoch  default=2000
             --learning_rate default=0.0001
             --learning_rate_decay default=0.99
             --dropout_rate default=1
             --L2_regularizer default=0.05
             --kernel_size default=5





Prediction Score：

file score1.csv Allelic imbalance

file socre2.csv eQTL

file socre3.csv dsQTL

file score4.csv validated regulatory

Training code: PRVCNN_train

Testing code: PRVCNN_test1-4

notes:input must be reshaped and change the string feature to one-hot code


