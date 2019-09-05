# PRVCNN
Predicting regulatory variants with Convolutional Neural Network

Gaoxiang Li

Auburn University

Advisor:Li Chen, Xiao Qin 

Prediction Score：

file score1.csv Allelic imbalance

file socre2.csv eQTL

file socre3.csv dsQTL

file score4.csv validated regulatory

Training code: PRVCNN_train

Testing code: PRVCNN_test1-4

notes:input must be reshaped and change the string feature to one-hot code

How to use:

1.training: After download and unzip the file, run PRVCNN_train.py (./PRVCNN_train.py) in this directory.

2.testing:Run PRVCNN_test1.py (./PRVCNN_test1.py) PRVCNN_test2.py  PRVCNN_test3.py  PRVCNN_test4.py  four python files one by one.
The result prediction score will be saved as score1.csv score2.csv score3.csv score4.csv.
The result figure will be drawed automaticaly.
