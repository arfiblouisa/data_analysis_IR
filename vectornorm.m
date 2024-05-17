function [matrix]=vectornorm(datasetmatrix)

ss=size(datasetmatrix);
%subtracts mean
meandata=zeros(ss);
for i=1:ss(1),
    meandata(i,:)=datasetmatrix(i,:)/mean(datasetmatrix(i,:));
end
%normalizes
matrix=zeros(ss);
for i=1:ss(1),
    matrix(i,:)=meandata(i,:)/norm(meandata(i,:));
end
