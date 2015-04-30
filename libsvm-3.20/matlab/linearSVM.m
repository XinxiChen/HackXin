load a4data2.mat
model = svmtrain(labels_train, data_train,'-c 1 -g 2');

save('linearmodel.mat','model');

