% pure SVM with kernel
clear all;
load a4data;

XTrain = double(data_train);
TTrain = double(labels_train);

XTest = double(data_test);
TTest = double(labels_train(1:1200,:));

model = svmtrain(TTrain, XTrain, '-t 1 -d 3 -g 0');
[TTest] = svmpredict(TTest, XTest, model);

write_kaggle_csv('prediction', TTest);
