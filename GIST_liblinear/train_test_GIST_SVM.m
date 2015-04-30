[train_label,train_inst]=libsvmread('data.txt');
model = train(train_label,train_inst, '-s 1');
save('linearmodel.mat','model');

load('linearmodel.mat','model');
[test_label,test_inst]=libsvmread('test.txt');
[labels_test] = predict(test_label, test_inst, model);
% save('prediction.mat','labels_test');

write_kaggle_csv('prediction', labels_test);