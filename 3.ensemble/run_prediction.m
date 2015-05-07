%% Load CIFAR test data
clear all;
CIFAR_DIR='';

%% Configuration
addpath minFunc;
rfSize = 6;
whitening=true;
CIFAR_DIM=[32 32 3];

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1=load([CIFAR_DIR '/a4data.mat']);
testX = double(f1.data_test);
clear f1;

%% Load model, for kmeans_model series, please change the model name accordingly
load kmeans_model_70.67percent.mat;

% compute testing features and standardize
if (whitening)
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
end

testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
labels1 = labels - 1;
headers = {'id';'label'};
A = linspace(1,1200,1200);
data2 = [A',labels1];

save('mypredictions.mat','labels1');
write_kaggle_csv('prediction', labels1);