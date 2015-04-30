load a4data2.mat;

load('linearmodel.mat','model');

[labels_test] = svmpredict(double(ones(1200,1)),data_test,model);
