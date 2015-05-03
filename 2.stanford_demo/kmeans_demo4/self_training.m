% load train_XC.mat;
% load a4data.mat;
% trainY = double([labels_train]) + 1; % ADD 1 to LABELS!!!!!!!!!!!
% 
% % standardize training data
% trainXC_mean = mean(trainXC);
% trainXC_sd = sqrt(var(trainXC)+0.01);
% trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
% trainXCs = [trainXCs, ones(size(trainXCs,1),1)]; %append one after the original data?
% 
% % train classifier using SVM
% C = 100;
% theta = train_svm(trainXCs, trainY, C);

%self-training
load kmeans_model0503_2.mat;
trainX_nolabel=double(data_nolabel);

rfSize = 6; %receptive field size w*w
numCentroids=2000; %number of features, i.e. K in K-means
whitening=true; %whitening needed
numPatches = 3000000; 
CIFAR_DIM=[32 32 3]; %data set dimension



trainXC_nolabel = extract_features(trainX_nolabel, centroids, rfSize, CIFAR_DIM, M,P);

trainXC_nolabel_mean = mean(trainXC_nolabel);
trainXC_nolabel_sd = sqrt(var(trainXC_nolabel)+0.01);
trainXCs_nolabel = bsxfun(@rdivide, bsxfun(@minus, trainXC_nolabel, trainXC_nolabel_mean), trainXC_nolabel_sd);
trainXCs_nolabel = [trainXCs_nolabel, ones(size(trainXCs_nolabel,1),1)]; %append one after the original data?
[val,trainY_nolabel] = max(trainXCs_nolabel*theta, [], 2);
% train classifier twice using SVM
C = 100;
theta = train_svm([trainXCs;trainXCs_nolabel], [trainY;trainY_nolabel], C);

save('kmeans_model0503_3.mat','centroids','M','P','trainXC_mean','trainXC_sd','theta');
