
%% Load model
load('GIST_59.00percent.mat','model');
[test_label,test_inst]=libsvmread('test.txt');
[labels_test] = predict(test_label, test_inst, model);
save('mypredictions.mat','labels_test');
write_kaggle_csv('prediction', labels_test);