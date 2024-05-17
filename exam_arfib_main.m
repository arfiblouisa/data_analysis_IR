clear all
close all 

%load dataset
load Exam2024data.mat

figure;plot(xaxis,IR_for_exam);
xlabel('Wavenumbers cm^{-1}');
title('Raw Cannabis dataset')

y=zeros(87,3);
for i=1:87
    if class(i)==1
        y(i,1)=1;
    elseif class(i)==2
        y(i,2)=1;
    else
        y(i,3)=1;
    end
end


%% PCA on the raw dataset

[coeff,score,latent,tsquared,explained,mu] = pca(IR_for_exam,'NumComponents',4);

figure;plot(explained,'LineWidth',2);
xlabel('Principal Components')
ylabel('variance explained')

%plot loadings
figure;
plot(xaxis,coeff)
xlabel('Loadings with axis of Wavenumbers cm^{-1}');
legend('PC1','PC2','PC3','PC4')

%scoreplot
figure;gscatter(score(:,1), score(:,2), class);
xlabel('PC 1');ylabel('PC 2')
legend('Class 1','Class 2','Class 3')

%% PCA on baseline corrected dataset

for i=1:87
     [baseline_d,baseline]=whittaker_baseline(IR_for_exam(i,:)', 10^4, 10^-3);
      baselinedataset(i,:)=baseline_d';
end

figure;plot(xaxis,baselinedataset');
xlabel('Wavenumbers cm^{-1}');
title('Baseline corrected dataset')

[coeff,score,latent,tsquared,explained,mu] = pca(baselinedataset,'NumComponents',4);

figure;plot(explained(1:10),'LineWidth',2);
xlabel('Principal Components')
ylabel('variance explained')

%plot loadings
figure;
plot(xaxis,coeff)
xlabel('Loadings with axis of Wavenumbers cm^{-1}');
legend('PC1','PC2','PC3','PC4')

%scoreplot
figure;gscatter(score(:,1), score(:,2), class);
xlabel('PC 1');ylabel('PC 2')
legend('Class 1','Class 2','Class 3')

%clusters
Z=linkage(score(:,1:2),'ward');

%plot dendrogram
figure;dendrogram(Z)
title 'Dendrogram'
%cluster using three clusters 

T = cluster(Z,'MaxClust',3);

figure;gscatter(score(:,1),score(:,2),T)
for ii = 1:length(score(:,1))
text(score(ii,1),score(ii,2),string(class(ii)),'Color','black')
end
legend('Class 1 found','Class 2 found','Class 3 found')
xlabel('PC1');ylabel('PC2')


%% PCA on baseline corrected and vector normalized

% vectornorm
baselined_vnorm=vectornorm(baselinedataset);
figure;plot(xaxis,baselined_vnorm');
xlabel('Wavenumbers cm^{-1}');
title('Baseline corrected and normalized dataset')

figure;plot(xaxis,baselinedataset');
xlabel('Wavenumbers cm^{-1}');
title('Baseline corrected dataset')

[coeff,score,latent,tsquared,explained,mu] = pca(baselined_vnorm,'NumComponents',4);

figure;plot(explained(1:10),'LineWidth',2);
xlabel('Principal Components')
ylabel('variance explained')

%plot loadings
figure;
plot(xaxis,coeff)
xlabel('Loadings with axis of Wavenumbers cm^{-1}');
legend('PC1','PC2','PC3','PC4')

%scoreplot
figure;gscatter(score(:,1), score(:,2), class);
xlabel('PC 1');ylabel('PC 2')
legend('Class 1','Class 2','Class 3')

%clusters
Z=linkage(score(:,1:2),'ward');

%plot dendrogram
figure;dendrogram(Z)
title 'Dendrogram'
%cluster using three clusters 

T = cluster(Z,'MaxClust',3);

figure;gscatter(score(:,1),score(:,2),T)
for ii = 1:length(score(:,1))
text(score(ii,1),score(ii,2),string(class(ii)),'Color','black')
end
legend('Class 1 found','Class 2 found','Class 3 found')
xlabel('PC1');ylabel('PC2')


%% Outliers removal with T^2 and Q

%T^2
tsqreduced = mahal(score,score);

%Q residual
ss=size(baselined_vnorm);
I = eye(ss(2));
for i=1:ss(1)
Q(i)=baselined_vnorm(i,:)*(I-(coeff*coeff'))*baselined_vnorm(i,:)';
end

%plot against each other
figure;gscatter(tsqreduced,Q,class)
legend('Class 1', 'Class 2','Class 3')
xlabel('Hotelling T^{2}')
ylabel('Q residuals')
xline(0);yline(mean(Q));

% remove outliers in Q statistics 
limitQ=0.9;
limitT=15;
k=1;
for i=1:87
    if Q(i)>limitQ
        if tsqreduced(i)<limitT
            dataset_outlier(k,:)=baselined_vnorm(i,:);
            y_outlier(k,:)=y(i,:);
            k=k+1;
        end
    end
end

%figure;plot(dataset_outlier')

%reperform PCA with outliers removed. 
[coeff,score,latent,tsquared,explained,mu] = pca(dataset_outlier,'NumComponents',4);

figure;plot(coeff);


figure;gscatter(score(:,1), score(:,2), y_outlier);
legend('Class 1','Class 2','Class 3')

clear tsqreduced
clear Q;

%T^2
tsqreduced = mahal(score,score);
%Q residual 
ss=size(dataset_outlier);
I = eye(ss(2));
for i=1:ss(1)
Q(i)=dataset_outlier(i,:)*(I-(coeff*coeff'))*dataset_outlier(i,:)';
end

%plot against each other
figure;gscatter(tsqreduced,Q,y_outlier)
xline(0);yline(mean(Q));
legend('Class 1', 'Class 2','Class 3')

 
%% PLS on baselined

y_norm=normalize(THCA,1,'center','mean');

[Xl,Yl,Xs,Ys,beta,pctVar,PLSmsep] = plsregress(baselinedataset,y_norm,10,'CV',5);
[XL,YL,XS,YS,BETA,PCTVAR,MSE] = plsregress(baselinedataset,y_norm,10);


%plot both errors in same plot 
figure;plot(0:10,sqrt(MSE(2,:)),'b-o',0:10,sqrt(PLSmsep(2,:)),'r-o');
xlabel('Number of components');
ylabel('Estimated Root Mean Squared Prediction Error');
legend({'PLSR RMSEV','PLSR RMSECV' },'location','NE');

X=baselinedataset;
y=THCA;

[n,p] = size(X);

ncomp=6; %found with cross fold validation

[Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar] = plsregress(X,y,ncomp);

plot(1:ncomp,cumsum(100*PLSPctVar(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in Y');
yfitPLS = [ones(n,1) X]*betaPLS;

% plot loadings 
figure;plot(xaxis,Xloadings);
legend('PC1', 'PC2', 'PC3', 'PC4','PC5','PC6')

TSS = sum((y-mean(y)).^2);
RSS_PLS = sum((y-yfitPLS).^2);
rsquaredPLS = 1 - RSS_PLS/TSS

figure;plot(1:ncomp,100*cumsum(PLSPctVar(1,:)));
xlabel('Number of Principal Components');
ylabel('Percent Variance Explained in X');
legend({'PLSR'},'location','SE');

figure;plot(y,y,'-k',y,yfitPLS,'bo');
xlabel('Observed Response');
ylabel('Fitted Response');
legend({'y','PLS with 6 Components'},  ...
	'location','NW');

%% PLS on baselined_vnorm

y_norm=normalize(THCA,1,'center','mean');

[Xl,Yl,Xs,Ys,beta,pctVar,PLSmsep] = plsregress(baselined_vnorm,y_norm,10,'CV',5);
[XL,YL,XS,YS,BETA,PCTVAR,MSE] = plsregress(baselined_vnorm,y_norm,10);

%plot both errors in same plot 
figure;plot(0:10,sqrt(MSE(2,:)),'b-o',0:10,sqrt(PLSmsep(2,:)),'r-o');
xlabel('Number of components');
ylabel('Estimated Root Mean Squared Prediction Error');
legend({'PLSR RMSEV','PLSR RMSECV' },'location','NE');

X=baselinedataset;
y=THCA;

[n,p] = size(X);

ncomp=7; %found with cross fold validation

[Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar] = plsregress(X,y,ncomp);

plot(1:ncomp,cumsum(100*PLSPctVar(2,:)),'-bo');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in Y');
yfitPLS = [ones(n,1) X]*betaPLS;

% plot loadings 
figure;plot(xaxis,Xloadings);
legend('PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7')

TSS = sum((y-mean(y)).^2);
RSS_PLS = sum((y-yfitPLS).^2);
rsquaredPLS = 1 - RSS_PLS/TSS

figure;plot(1:ncomp,100*cumsum(PLSPctVar(1,:)));
xlabel('Number of Principal Components');
ylabel('Percent Variance Explained in X');
legend({'PLSR'},'location','SE');

figure;plot(y,y,'-k',y,yfitPLS,'bo');
xlabel('Observed Response');
ylabel('Fitted Response');
legend({'y','PLS with 7 Components'},  ...
	'location','NW');