%% Load CIFAR test data
clear all;
CIFAR_DIR='';

%% Configuration
addpath minFunc;
rfSize = 6;
numCentroids=1600;
whitening=true;
numPatches = 400000;
CIFAR_DIM=[32 32 3];

%% Load CIFAR test data
fprintf('Loading test data...\n');
%f1=load([CIFAR_DIR '/test_batch.mat']);

%Modified by Xinxi
f1=load([CIFAR_DIR '/a4data.mat']);

testX = double(f1.data_test);
% testY = double(f1.labels);
clear f1;

%% Load model
f2=load([CIFAR_DIR '/kmeans_model1.mat']);
centroids = f2.centroids; 
M = f2.M; 
P = f2.P; 
trainXC_mean = f2.trainXC_mean;
trainXC_sd = f2.trainXC_sd;
theta = f2.theta;

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
%data1 = ['id';'label'];
%celldata = cellstr(data);
labels1 = labels - 1;
headers = {'id';'label'};
A = linspace(1,1200,1200);
data2 = [A',labels1];
%csvwrite_with_headers('prediction.csv',data2,headers);
csvwrite_with_headers('prediction.csv',data2,headers);

