% 
% %% Load CIFAR test data
% clear all;
% load trainXCsY.mat;
% 
% %% Configuration
% addpath minFunc;
% rfSize = 6;
% numCentroids=5000;
% whitening=true;
% CIFAR_DIM=[32 32 3];
% 
% %% Load model

load a4data.mat;
load trainXCs.mat;
load kmeans_model0505.mat;

% train classifier using SVM
C = 10;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));
save('kmeans_model0505_test_feature_C10_C1_1st.mat','centroids','M','P','trainXC_mean','trainXC_sd','theta');

trainX_nolabel=double(data_nolabel);
%with whitening
trainXC_nolabel = extract_features(trainX_nolabel, centroids, rfSize, CIFAR_DIM, M,P);
trainXC_nolabel_mean = mean(trainXC_nolabel);
trainXC_nolabel_sd = sqrt(var(trainXC_nolabel)+0.01);
trainXCs_nolabel = bsxfun(@rdivide, bsxfun(@minus, trainXC_nolabel, trainXC_nolabel_mean), trainXC_nolabel_sd);
trainXCs_nolabel = [trainXCs_nolabel, ones(size(trainXCs_nolabel,1),1)]; %append one after the original data?
[val,trainY_nolabel] = max(trainXCs_nolabel*theta, [], 2);
save('trainXCs_nolabel.mat','trainXCs_nolabel','trainY_nolabel');

% train classifier twice using SVM
C = 1;
theta = train_svm([trainXCs;trainXCs_nolabel], [trainY;trainY_nolabel], C);

save('kmeans_model_test_feature_C10_C1_2nd.mat','centroids','M','P','trainXC_mean','trainXC_sd','theta');
