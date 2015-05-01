Data = load('train.mat');
DataField = fieldnames(Data);
dlmwrite('FileName1.txt', Data.(DataField{1}));
dlmwrite('FileName2.txt', Data.(DataField{2}));
dlmwrite('FileName3.txt', Data.(DataField{3}));
dlmwrite('FileName4.txt', Data.(DataField{4}));
dlmwrite('FileName5.txt', Data.(DataField{4}));