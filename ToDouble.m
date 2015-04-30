load a4data.mat

%uint8 to double, for the convenience of libsvm API

data_test=double(data_test);
data_train = double(data_train);
labels_train=double(labels_train);

save('a4data2.mat')