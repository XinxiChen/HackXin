C = 1;
f2=load([ '/kmeans_model_1st_train.mat']);
centroids = f2.centroids; 
M = f2.M; 
P = f2.P; 
trainXC_mean = f2.trainXC_mean;
trainXC_sd = f2.trainXC_sd;

load trainXCsY.mat;
theta = train_svm(trainXCs, trainY, C);

save('kmeans_model_1st_C1.mat','centroids','M','P','trainXC_mean','trainXC_sd','theta');
