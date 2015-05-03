clear all;
CIFAR_DIR='';
addpath minFunc;
%% Configuration, parameters to tune

% extracting random sub-patches from unlabeled input images. 
% Each patch has dimension w-by-w and has d channels 
% Each w-by-w patch can be represented as a vector in R^N of pixel intensity values, with N = w · w · d. 
% construct a dataset of m randomly sampled patches, X = {x(1) , ..., x(m) }, where x(i) ? R^N . 
% Given this dataset, we apply the pre-processing and unsupervised learning steps.
rfSize = 6; %receptive field size w*w

numCentroids=5000; %number of features, i.e. K in K-means

whitening=true; %whitening needed

numPatches = 5000000; 

CIFAR_DIM=[32 32 3]; %data set dimension

%% Load CIFAR training data
fprintf('Loading training data...from both labeled and unlabeled\n');
f1=load([CIFAR_DIR '/a4data.mat']);

trainFeature=double([f1.data_train;f1.data_nolabel]);
trainX = double([f1.data_train]);
trainY = double([f1.labels_train]) + 1; % ADD 1 to LABELS!!!!!!!!!!!

clear f1;

% extract random patches
patches = zeros(numPatches, rfSize*rfSize*3); %initialize numPatches????patches


%i=1, extract patches from 1st image in data_train
%i=2, extract patches from 2nd image in data_train
%......always extract the i-th patches from the mod(i-1,size(trainX,1))+1
%image in data_train
for i=1:numPatches
  if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end
  
  %Uniform Distribution (Discrete) to get a random upper left row number
  %from 32-6+1=27
  r = random('unid', CIFAR_DIM(1) - rfSize + 1); 
  
  %Uniform Distribution (Discrete) to get a random upper left column number
  %from 32-6+1=27
  c = random('unid', CIFAR_DIM(2) - rfSize + 1); 
  
  %get the ith patch from the mod(i-1,size(trainX,1))+1 image in train_data
  patch = reshape(trainFeature(mod(i-1,size(trainFeature,1))+1, :), CIFAR_DIM);
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
  patches(i,:) = patch(:)';
end

% pre-processing1: normalize for contrasts; Mean subtraction and scale normalization
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% pre-processing2: whitening 
if (whitening)
  C = cov(patches);
  M = mean(patches);
  [V,D] = eig(C);
  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
  patches = bsxfun(@minus, patches, M) * P;
end

% run K-means
centroids = run_kmeans(patches, numCentroids, 50); %iterate for 50 times
show_centroids(centroids, rfSize); %cetroid: 1600*(6*6*3), height=6; width=6, depth/channel=3
drawnow; 

% extract features from training set
if (whitening)
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);
else
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
end

% standardize training data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)]; %append one after the original data?

% train classifier using SVM
C = 100;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));
save('kmeans_model_1st_train.mat','centroids','M','P','trainXC_mean','trainXC_sd','theta');


%%%%%%%% self-training starts => make unlabeled to roughly labeled
trainX_nolabel=double(data_nolabel);

trainXC_nolabel = extract_features(trainX_nolabel, centroids, rfSize, CIFAR_DIM, M,P);
trainXC_nolabel_mean = mean(trainXC_nolabel);
trainXC_nolabel_sd = sqrt(var(trainXC_nolabel)+0.01);
trainXCs_nolabel = bsxfun(@rdivide, bsxfun(@minus, trainXC_nolabel, trainXC_nolabel_mean), trainXC_nolabel_sd);
trainXCs_nolabel = [trainXCs_nolabel, ones(size(trainXCs_nolabel,1),1)]; %append one after the original data?
[val,trainY_nolabel] = max(trainXCs_nolabel*theta, [], 2);

save('trainXCs.mat','train_XCs','trainXCs_nolabel','trainY','trainY_nolabel');

% train classifier twice using SVM
C = 100;
theta = train_svm([trainXCs;trainXCs_nolabel], [trainY;trainY_nolabel], C);

save('kmeans_model_2nd_train.mat','centroids','M','P','trainXC_mean','trainXC_sd','theta');
