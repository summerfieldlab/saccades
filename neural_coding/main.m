% This script runs all the neural coding analyses.  Calculating response
% curves, tuning curves and spatial response fields for each unit in the
% recurrent layer of our dual-stream RNN. The relevant activations can be
% found on the associated OSF repository: https://osf.io/h6evt/

clear all
close all

%% load data
folder = 'activations/';
model = 'dual-stream';
checkpoints = {'convergence'}; % load only activations from the end of training
% checkpoints = {'init'};
%checkpoints = {'init','50','75','convergence'}; % OLD - 4 checkpoints
% stim_set = 'ignore-distractors_logpolar_new';
stim_set = 'simple-counting_logpolar_new';

for k = 1:length(checkpoints)
    disp(['loading ',model,'_',stim_set,'_both_at-',checkpoints{k},'.mat']);
    saccades{k} = load([folder, model,'_',stim_set,'-both_at-',checkpoints{k},'.mat']);
    saccades{k}.prediction = saccades{k}.predicted_num;
    saccades{k}.numerosity =saccades{k}.numerosity + 1;
    saccades{k} = rmfield(saccades{k},'predicted_num');
    if k~=5
    for i = 1:5000
        saccades{k}.predicted_num(i) = find(saccades{k}.prediction(i,:)==max(saccades{k}.prediction(i,:)));
    end
    else
        saccades{k}.predicted_num = saccades{k}.prediction;
    end
end

%% calculate confusion

figure('color',[1 1 1],'position',[616 638 885 379]);
for k = 1:length(saccades)
    
    conf_matrix = NaN(5,5);
    
    for i = 1:5
        for j = 1:5
            conf_matrix(i,j) =  sum(saccades{k}.numerosity==i & saccades{k}.predicted_num==j);
        end
    end
    
    conf_matrix = conf_matrix./sum(conf_matrix,2);
    subplot(2,length(saccades),k);
    imagesc(conf_matrix)
    colormap viridis
    set(gca,'ydir','normal');
    set(gca,'clim',[0 1]);
    
    
    XX = [saccades{k}.numerosity' saccades{k}.num_distractor'  saccades{k}.num_distractor'.*saccades{k}.numerosity'];
    [bb dev stats] = glmfit(XX,saccades{k}.correct','binomial','link','logit');
    disp([checkpoints{k},': t = ',num2str(stats.t(2:end)'),' p = ',num2str(stats.p(2:end)')]);
    [x_vals,y_vals] = ndgrid(linspace(0,5,20),linspace(0,5,20));
    xy_vals = [x_vals(:) y_vals(:) x_vals(:).*y_vals(:)];
    yy = glmval(bb,xy_vals,'logit');
    % disp(['range = ',num2str(ranger(yy))]);
    
    subplot(2,length(saccades),k+length(saccades));
    contourf(linspace(0,5,20),linspace(0,5,20),reshape(yy,20,20)');
    set(gca,'clim',[0 1]);
    set(gca,'ydir','normal');
    xlabel('num. targets','FontSize',14);
    ylabel('num. distracters','FontSize',14);
    
end

%% plot some tuning curves
close all;

for k = 1:length(saccades)
    layer = 'hidden';
    eval(['data = saccades{k}.act_',layer,';']);
    data_mean = squeeze(mean(data,2));
    
    num_mat = unique(saccades{k}.numerosity);
    
    for n = 1:length(num_mat)
        
        indx = (find(saccades{k}.numerosity==num_mat(n) & saccades{k}.num_distractor==0));
        
        indx_1 = indx(1:floor(length(indx)/2));
        indx_2 = indx(floor(length(indx)/2):end);
        
        tuning_1{k}(:,n,:) = squeeze(mean(data(indx_1,:,:)))';
        base = mean(tuning_1{k}(:,:,1),2);
        tuning_1{k} = tuning_1{k}-base;
        
        tuning_2{k}(:,n,:) = squeeze(mean(data(indx_2,:,:)))';
        base = mean(tuning_2{k}(:,:,1),2);
        tuning_2{k} = tuning_2{k}-base;
        
        % indx = mean(mean(tuning(:,:,7:11),2),3)<0;
        % tuning(indx,:,:) = tuning(indx,:,:)*-1;
    end
    
    
    figure('color',[1 1 1],'position',[425 290 768 727]);
    getcolz
    rows = 6;
    cols = 8;
    for i = 1:rows*cols
        subplot(rows,cols,i);
        hold on;
        for s = 1:12
            plot(tuning_1{k}(i,:,s),'color',1-[s/12 s/12 s/12],'linewidth',3);
            set(gca,'xtick',1:5,'yticklabel',{''});
            %ylim([-0.5 0.5]);
            set(gca,'FontSize',16);
        end
    end
    
    
end

return

%% plot average curves

close all;
figure('color',[1 1 1]);
mat_colz = {'b','c','g',[1 0.64 0],'r'};

for k = 1:length(saccades)
    for i = 1:size(tuning_1{k},1)
        mtune = mean(tuning_1{k}(i,:,1:12),3);
        max_tune(i) = find(mtune==max(mtune));
        num_tune(i) = (max(mtune)-min(mtune))./(max(mtune)+min(mtune));
    end
    
    for n = 1:length(num_mat)
       
        indx = find(max_tune==n);
        tune_plot(n,:) = squeeze(mean(mean(tuning_2{k}(indx,:,1:12),3)));
        tune_plot(n,:) = scaler(tune_plot(n,:)).*100;
    end
    
    subplot(1,length(saccades),k);
    
    for n = 1:length(num_mat);
        
        % plot(1:5,tune_plot(n,:),'-o','linewidth',7,'color',mat_colz{n},'MarkerSize',10);
        semilogx(1:5,tune_plot(n,:),'-o','linewidth',7,'color',mat_colz{n},'MarkerSize',10);
        set(gca,'xtick',1:5);
        set(gca,'ytick',0:20:100);
        ylim([0 100]);
        
        xlabel('Number of items (log)','FontSize',44);
        % ylabel({'Normalised'; 'response (%)'},'FontSize',36);
        ylabel({'Neuronal'; 'population'; 'activity (%)'},'FontSize',44);
        hold on;
        
    end
    set(gca,'FontSize',38);
    grid on
    ax = gca;
    box(ax,'off');
    ax.LineWidth=3;
    
end

%% Compare linear vs log normal tuning
% Fit Gaussian tuning curves in space of linear n
ssq0 = 0;

subplot(2,1,1)
hold on
for n = 1:length(num_mat)
    [f, s] = fit([1:5].', (tune_plot(n,:)/100).','gauss1');
    
    plot(tune_plot(n,:)/100, 'linewidth', 3, 'color', mat_colz{n})
    plot([1:5], f([1:5]),':', 'linewidth', 4, 'color', mat_colz{n})
    ssq0 = ssq0 + s.sse;
    xlabel("n")
    ylim([0, 1])
end
hold off
set(gca,'FontSize',32);
ylabel([{'Normalised'}, {'response (%)'}])

% Fit Gaussian tuning curves in space of log n
ssq1 = 0;
% legend({'1 neurons', 'Gaussian fit','2 neurons', 'Gaussian fit','3 neurons',...
    % 'Gaussian fit','4 neurons', 'Gaussian fit','5 neurons', 'Gaussian fit',}, 'Location', 'eastoutside')
subplot(2,1,2)

for n = 1:length(num_mat)
    [f, s] = fit(log([1:5]).', (tune_plot(n,:)/100).','gauss1');
    
    semilogx(1:5, tune_plot(n,:)/100, 'linewidth', 3, 'color', mat_colz{n})
    hold on
    semilogx([1:5], f(log([1:5])),':', 'linewidth', 4, 'color', mat_colz{n})
    ssq1 = ssq1 + s.sse;
    ylim([0, 1])
    xlabel("log(n)")
end
hold off
ylabel([{'Normalised'}, {'response (%)'}])
set(gca,'FontSize', 32);

% Calculate F-ratio test according to https://sites.duke.edu/bossbackup/files/2013/02/FTestTutorial.pdf
n_datapoints = 25;
n_params = 5*2;  % 5 gaussians each with two params
df = n_datapoints-n_params % same df for both models
f_ratio = ssq0/ssq1
p = 1 - fcdf(f_ratio,df,df)
% 

%% Plot tuning as a function of numerical distance from preferred numerosity (all units pooled)

diffmat = -4:4;
tmp = zeros(size(tuning_1{k},1),length(diffmat));

for i = 1:size(tuning_1{k},1);
    for n = 1:length(num_mat);
        diff = max_tune(i)-num_mat(n);
        ind = find(diff==diffmat);
        filter_data(i,ind) = tmp(i,ind) + mean(tuning_1{k}(i,n,:));
    end
end

figure('color',[1 1 1]);
set(gca,'DefaultLineLineWidth',20)
normalised = (filter_data-min(mean(filter_data),[],"all"));
normalised = (normalised./max(mean(normalised))).*100;
stpn(normalised,{'k'});
set(gca,'xtick',1:9,'xticklabel',diffmat);
set(gca,'FontSize',32);
ylabel({'Normalised'; 'response (%)'},'FontSize',36);
xlabel('Numerical Distance', 'FontSize', 36)
ax = gca;
box(ax,'off');
ax.LineWidth=3;

%% frequences of selectivity

figure('color',[1 1 1]);
[n x] = hist(max_tune,1:5);
n = (n./sum(n))*100;
b = bar(x,n);
b.FaceColor = 'flat';
% b.CData(1,:) = [1 0 0]; %red
% b.CData(2,:) = [0 0 1]; % blue
% b.CData(3,:) = [0 1 0]; % green
% b.CData(4,:) = [1 1 0];
% b.CData(5,:) = [.5 0 .5];
b.CData(1,:) = [0 0 1]; % blue
b.CData(2,:) = [0 1 1]; % cyan
b.CData(3,:) = [0 1 0]; % green
b.CData(4,:) = [1 0.64 0];
b.CData(5,:) = [1 0 0]; %red

ylabel('Percentage of units');
xlabel('Preferred numerosity');
set(gca,'FontSize',32);
ax = gca;
box(ax,'off');
    
    
return
%% spatial position

fspace = linspace(0.05,0.95,21);
% fspace = linspace(-0.05,1.05,21); % Quite a lot of random high mean activation at the very extremes so perhaps best to exclude
k = 1;

activations = NaN(length(fspace)-1,length(fspace)-1,1024);

for x = 1:length(fspace)-1;
    disp(['x = ',num2str(x)]);
    for y = 1:length(fspace)-1;
        % disp(['x = ',num2str(x),' y = ',num2str(y)]);
        indx = find(saccades{k}.glimpse_xy(:,:,1) > fspace(x) & saccades{k}.glimpse_xy(:,:,1) < fspace(x+1) & saccades{k}.glimpse_xy(:,:,2) > fspace(y) & saccades{k}.glimpse_xy(:,:,2) < fspace(y+1));
        if ~isempty(indx)
            for n = 1:1024;
                act = squelch(saccades{k}.act_hidden(:,:,n));
                activations(x,y,n) = mean(act(indx));
            end
        end
        
        
    end
end

%% Plot sample of units
figure('color',[1 1 1]);
[nex, ngl, nunits] = size(activations)
units = [2, 7, 11, 14, 26, 29, ...
         30, 31, 32, 37, 38, 39,...
         40, 41, 42, 43, 45, ...
         55, 56, 57, 58,...
         59, 61, 62, 63, 64,...
         50, 65, 66, 67, 69, 70];
% units = randsample(nunits, 36, false)
% for n = 1:36;subplot(6,6,n);
for n = 1:25;subplot(5,5,n);
    imagesc(activations(:,:,units(n)));
    set(gca,'clim',[0 1.25]);
    set(gca,'xticklabel',{''},'yticklabel',{''});
end
% colorbar()
% colormap jet;


%% fit Gaussian tuning model to spatial RFs
% The set of centroids to test
[x_vals,y_vals] = ndgrid(medi(fspace),medi(fspace));
xy_vals = [x_vals(:) y_vals(:)];

% The set of dispersions to test
sigma_mat = exp(linspace(log(0.001),log(0.5),40));

% For each neuron, fit Gaussians at each candiate centroid with each
% candidate dispersion
for n = 1:1024
    disp(['neuron...',num2str(n)]);
    scale_act = scaler(activations(:,:,n));
    clear dev;
    for xy = 1:size(xy_vals,1)
        for s = 1:length(sigma_mat)
            fit = mvnpdf(xy_vals,xy_vals(xy,:),[sqrt(sigma_mat(s)) 0;0 sqrt(sigma_mat(s))]);
            fit = scaler(fit);
            dev(xy,s) = sum((fit-scale_act(:)).^2);
        end
    end
    % Save indices and params of best fit for each neuron
    ii = find4(dev==min(dev(:)));
    neuron_min(n) = min(dev(:));
    neuron_param(n,:) = ii;
    
end

%% Plot Gaussian fit for neurons with best fit
[~,sort_ii] = sort(neuron_min,'ascend');
%sort_ii = 1:36;

figure('color',[1 1 1]);
for k = 1:36;
    subplot(6,6,k);
    imagesc((activations(:,:,sort_ii(k))));
    set(gca,'xticklabel',{''},'yticklabel',{''});
    
end
colormap jet

figure;
for k = 1:36;
    subplot(6,6,k);
    best_fit =  mvnpdf(xy_vals,xy_vals(neuron_param(sort_ii(k),1),:),[(sigma_mat(neuron_param(sort_ii(k),2))) 0;0 (sigma_mat(neuron_param(sort_ii(k),2)))]);
    imagesc(reshape(best_fit,length(fspace)-1,length(fspace)-1));
    set(gca,'clim',[0 1]);
end


%% Assess relationship between eccentricty and width of spatial response fields
[~,sort_ii] = sort(neuron_min,'ascend');
ecc = pdist2(xy_vals(neuron_param(sort_ii,1),:),[0.5 0.5]);
sig = sigma_mat(neuron_param(sort_ii,2));
% ecc = pdist2(xy_vals(neuron_param(sort_ii(1:512),1),:),[0.5 0.5]); % look
% only at neurons with strongest gaussian fit
% sig = sigma_mat(neuron_param(sort_ii(1:512),2)); 
% ind  = ecc<0.6; % analyse only those with eccentricity less than 0.6
% ecc = ecc(ind);
% sig = sig(ind);
figure;
% plot(ecc,sig,'ko');
% h = histogram2(ecc(:),sig(:),20,'DisplayStyle','tile','ShowEmptyBins','on')
Xedges = linspace(0, 0.62, 40);
Yedges = sigma_mat;
h = histogram2(ecc(:),sig(:),Xedges,Yedges,'DisplayStyle','tile','ShowEmptyBins','on')
% h = histogram2(ecc(sort_ii(1:512)),sig(sort_ii(1:512))',Xedges,Yedges,'DisplayStyle','tile','ShowEmptyBins','on')
set(gca, "YScale", "log")
% ylim([0 1]);
xlabel('RF eccentricity')
ylabel('RF width');
set(gca,'ColorScale','log')
cb= colorbar();
ylabel(cb,'Number of units','FontSize',24,'Rotation',270)
[y,newecc] = fitxy(ecc,sig',1);
hold on;
plot(newecc,y,'r-', 'LineWidth',2);
% ylim([0, 0.4])
set(gca,'FontSize',32);

[r,p] = corrcoef(ecc,sig)
% [r,p] = corrcoef(ecc(sort_ii(1:512)),sig(sort_ii(1:512) ))
text(0.16,0.1,'r=0.23, p<0.001','Color','white','FontSize',20)

% %% relationship between numerical tuning and spatial tuning
% 
% figure('color',[1 1 1]);
% hold on;
% 
% scale_num_tune = scaler(abs(num_tune));
% 
% for n = 1:1024;
%     plot(xy_vals(neuron_param(n,1),1)+randn*0.01,xy_vals(neuron_param(n,1),2)+randn*0.01,'o','markerfacecolor',mat_colz{max_tune(n)},'markeredgecolor','w','markersize',5+(scale_num_tune(n)*10));
% end





%% plot ramping activity

mat_colz = {'b','c','g',[1 0.64 0],'r'};

figure('color',[1 1 1]);
% for n = 1:36;
%     subplot(6,6,n);
for n = 1:4*4;
    subplot(4,4,n);
    for i = 1:5;
        hold on;
        indx = find(saccades{1}.numerosity==i);
        tmp_ramp = mean((saccades{1}.act_hidden(indx,:,n+5)));
        plot(1:12,tmp_ramp,'linewidth',3,'color',mat_colz{i});
        set(gca,'FontSize',16);
    end
   % ylim([0 0.2]);
    xlim([1 12]);
    % if n < 31;
    if n < 13;
        xticklabels("")
    end
end
% legend('1', '2', '3', '4', '5')




%% Neural Geometry

num_dist = unique(saccades{1}.num_distractor);

for n = 1:length(num_mat);
    for d = 1:length(num_dist);
        indx = (saccades{1}.numerosity==num_mat(n) & saccades{1}.num_distractor==num_dist(d));
        d_tuning(:,:,n,d) = squeeze(mean(data(indx,:,:)))';
        % base = mean(d_tuning(:,:,1,d),2);
        % d_tuning = d_tuning-base;
    end
    % indx = mean(mean(tuning(:,:,7:11),2),3)<0;
    % tuning(indx,:,:) = tuning(indx,:,:)*-1;
end

%%

rsa_tuning = mean(d_tuning,2); % Average over glimpses
% rsa_tuning = d_tuning(:,end,:); % Take just the last glimpse
rsa_tuning = rsa_tuning(:,:);

RSA = squareform(pdist(rsa_tuning'));
MDS = cmdscale(RSA,3);

f1 = figure('color',[1 1 1]);
hold on;
mat_colz = {'b','c','g',[1 0.64 0],'r'};
% Plot separately for each number of distractors (if present)
if size(MDS, 1) > 5
    markz = {'o','s','d','o','s','d','o','s','d','o','s','d','o','s','d','o','s','d'};
    cmat = [mat_colz mat_colz mat_colz];
    
    for k = 1:15
        
        plot3(MDS(k,1),MDS(k,2),MDS(k,3),'marker',markz{k},'markersize',15,'markerfacecolor',cmat{k},'markeredgecolor',cmat{k});
        % set(gca,'xlim',[-4 3]);
        % set(gca,'ylim',[-4 3]);
        % set(gca,'zlim',[-4 3]);
    end
    axis equal
else
    for k = 1:5
        % plot3(MDS(k,1),MDS(k,2),MDS(k,3),'marker', 'o', 'markersize',15,'markerfacecolor',mat_colz{k},'markeredgecolor',mat_colz{k});
        plot(MDS(k,1),MDS(k,2),'marker', 'o', 'markersize',30,'markerfacecolor',mat_colz{k},'markeredgecolor',mat_colz{k});
        % set(gca,'xlim',[-5 7]);
        % set(gca,'ylim',[-5 7]);
        % set(gca,'zlim',[-5 7]);
    end
    textscatter(MDS(:,1), MDS(:,2), {'1', '2', '3', '4', '5'}, 'ColorData', [1, 1, 1], 'FontSize', 25, 'FontWeight', 'bold')
    axis equal
end
set(gca,'FontSize',32);
xlabel('Dimension 1')
ylabel('Dimension 2')
zlabel('Dimension 3')
%animate_mds(f1,1,[1 1],1);


% figure('color',[1 1 1],'position',[425 290 768 727]);
% getcolz
% for i = 1:100;
%     subplot(10,10,i);
%     hold on;
%     for d = 1:length(num_dist)
%         plot(d_tuning(i,:,12,d),'color',colz{d},'linewidth',3);
%         set(gca,'xtick',1:5,'yticklabel',{''});
%
%     end
% end

%% Split trials into two halves, attempt to reconstruct one half from dim reduced version of the other
% Split into two halves, 2500 observations each
parpool(4)
for rep=1:1000
    rep
n_obs = size(data, 1);
indx = 1:n_obs;
indx_shuff = indx(randperm(n_obs));

data1 = data(indx_shuff(1:n_obs/2), :, :);
data2 = data(indx_shuff((n_obs/2)+1:end), :, :);
% d_tuning1 = d_tuning(indx_shuff(1:n_neurons/2), :, :);
% d_tuning2 = d_tuning(indx_shuff((n_neurons/2)+1:end), :, :);

% Calculate similarity matrices for each half
rsa_tuning1 = mean(data1,2); % Average over glimpses
rsa_tuning2 = mean(data2,2); % Average over glimpses
[rsq1_avg, rsq2_avg] = reconstruct_from_dim_reduce(rsa_tuning1, rsa_tuning2);

% rsa_tuning1 = data1(:, end, :);  % Take just the last glimpse
% rsa_tuning2 = data2(:, end, :);  % Take just the last glimpse
% [rsq1_last, rsq2_last] = reconstruct_from_dim_reduce(rsa_tuning1, rsa_tuning2);


% plot(1:5, mean_rsq)
% set(gca,'FontSize',32);

mean_rsq1_avg(rep, 1:5) = mean(rsq1_avg, 2); % Average over neurons
stderr_rsq1_avg(rep, 1:5) = std(rsq1_avg, 0, 2)/sqrt(size(rsq1_avg, 2));
mean_rsq2_avg(rep, 1:5) = mean(rsq2_avg, 2);
stderr_rsq2_avg(rep, 1:5) = std(rsq2_avg, 0, 2)/sqrt(size(rsq2_avg, 2));
end
% save("dimensionality.mat","mean_rsq1_avg","stderr_rsq1_avg", "mean_rsq2_avg", "stderr_rsq2_avg")

%% Plot
meanmean_rsq2_avg = mean(mean_rsq2_avg, 1);
stdmean_rsq2_avg = std(mean_rsq2_avg, 0, 1)/sqrt(1000);
errorbar(1:5, meanmean_rsq2_avg, stdmean_rsq2_avg, 'LineWidth', 3)


% mean_rsq1_last = mean(rsq1_last, 2);
% stderr_rsq1_last = std(rsq1_last, 0, 2)/sqrt(size(rsq1_last, 2));
% mean_rsq2_last = mean(rsq2_last, 2);
% stderr_rsq2_last = std(rsq2_last, 0, 2)/sqrt(size(rsq2_last, 2));
% errorbar(1:5, mean_rsq1_avg, stderr_rsq1_avg, 'LineWidth', 3)
% hold on
% errorbar(1:5, mean_rsq2_avg, stderr_rsq2_avg, 'LineWidth', 3, 'color', 'red')

% errorbar(1:5, mean_rsq1_last, stderr_rsq1_last, 'LineWidth', 3)
% errorbar(1:5, mean_rsq2_last, stderr_rsq2_last, 'LineWidth', 3, 'color', 'blue')
% legend({'Average glimpse', 'Last glimpse',})
% legend({'Average train', 'Average heldout',  'Last train', 'Last heldout'})
set(gca,'FontSize',32);
xlabel('Number of dimensions')
ylabel('R^2 of reconstruction')
% title('Last glimpse only')
% title('Average over glimpses')
set(gca,'xtick',1:5);

% hold off

%%
function [rsq1, rsq2] = reconstruct_from_dim_reduce(rsa_tuning1, rsa_tuning2)
% Dimensionality reduce both halves. Train a reconstruction model on the
% first half. Assess how well it generalises to the second half. Repeat for
% dimensionalities 1 through 5. Use Lasso regularised regression.
    n_neurons = size(rsa_tuning1, 3);
    rsa_tuning1 = rsa_tuning1(:,:);
    RSA1 = squareform(pdist(rsa_tuning1));
    rsa_tuning2 = rsa_tuning2(:,:);
    RSA2 = squareform(pdist(rsa_tuning2));
    for dim=1:5
        disp(dim)
        % Reduce the dimensionality
        MDS1 = cmdscale(RSA1,dim);
        MDS2 = cmdscale(RSA2,dim);
        parfor n=1:n_neurons
            % fit a linear model to reconstruct rsa1 from the dimensionality
            % reduced version. Then test how well this model can reconstruct
            % rsa 2
    
            % [b,bint,r,rint,stats] = regress(rsa_tuning1(:,n), MDS1);
            % pred = MDS2*b
            % mdl = fitlm(MDS1,rsa_tuning1(:,n));
            [mdl, FitInfo] = fitrlinear(MDS1,rsa_tuning1(:,n), 'Learner', 'leastsquares', 'Regularization','ridge');
            pred = mdl.predict(MDS1);
            r = corrcoef(pred, rsa_tuning1(:, n));
            rsq1(dim, n) = r(1,2)^2; % mdl.Rsquared.Ordinary;
            
            % How well does the reconstruction model transfer to the
            % heldout half?
            pred = mdl.predict(MDS2);
            r = corrcoef(pred, rsa_tuning2(:, n));
            rsq2(dim, n) = r(1,2)^2;
        end
        % mean_rsq(dim) = mean(rsq)
    end
end
