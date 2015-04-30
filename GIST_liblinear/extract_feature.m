load ../a4data
XTrain = double(data_train);
TTrain = double(labels_train);
XTest = double(data_test);
TTest=double(labels_train(1:1200,1));

allgist=[];
% use Gist for feature extraction
for i=1:3000
    clear R
    clear G
    clear B
    clear A
    clear param
    R=XTrain(i,1:1024);
    G=XTrain(i,1025:2048);
    B=XTrain(i,2049:3072);
    Image(:,:,1)=reshape(R,32,32);
    Image(:,:,2)=reshape(G,32,32);
    Image(:,:,3)=reshape(B,32,32);
    param.imageSize = [32 32]; % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    [gist1, param] = LMgist(Image, '', param);
    allgist = [allgist;gist1];
end

fid=fopen('data.txt','w'); 
for i=1:3000
   fprintf(fid,'%d ',TTrain(i,1));
   for j=1:512
    fprintf(fid, '%d:%g ',j,allgist(i, j));
   end
   fprintf(fid, '\n');
end
fclose(fid);

allgist=[];
% use Gist for feature extraction
for i=1:1200
    clear R
    clear G
    clear B
    clear A
    clear param
    R=XTest(i,1:1024);
    G=XTest(i,1025:2048);
    B=XTest(i,2049:3072);
    Image(:,:,1)=reshape(R,32,32);
    Image(:,:,2)=reshape(G,32,32);
    Image(:,:,3)=reshape(B,32,32);
    param.imageSize = [32 32]; % it works also with non-square images
    param.orientationsPerScale = [8 8 8 8];
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    [gist1, param] = LMgist(Image, '', param);
    allgist = [allgist;gist1];
end

fid=fopen('test.txt','w'); 
for i=1:1200
   fprintf(fid,'%d ',TTest(i,1));
   for j=1:512
    fprintf(fid, '%d:%g ',j,allgist(i, j));
   end
   fprintf(fid, '\n');
end
fclose(fid);