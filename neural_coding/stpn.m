function stpn(x,colz,linewid,xbins,barsonly);
% function stpn(x,colz,linewid,xbins);


if nargin<1;
    x = rand(10,2,3);
end

if nargin<2
 colz = {[0.8500    0.3250    0.0980],[0.4940    0.1840    0.5560],[0.3010    0.7450    0.9330],[ 0    0.4470    0.7410],[112 173 71]/255,'b','c'};
    
end

if nargin<3
    linewid = 3;
end

if nargin<4;
    xbins = 1:size(x,ndims(x));
end

if nargin<5
    barsonly = 0;
end

if ndims(x) ==2;
    y = squeeze(nanmean(x(:,:)))';
    e = (squeeze(nanstd(x(:,:)))')./sqrt(size(x,1));
    if barsonly
    errorbar(xbins,y,e,'color',colz{1},'linewidth',linewid,'linestyle','none');
    else        
    errorbar(xbins,y,e,'color',colz{1},'linewidth',linewid);
    end
end
    

if ndims(x) ==3;
% plot lines
for n = 1:size(x,2);
    hold on;
    y = squeeze(squeeze(nanmean(x(:,n,:))))';
    e = (squeeze(nanstd(x(:,n,:)))')./sqrt(size(x,1));
    if barsonly
    errorbar(xbins,y,e,'color',colz{n},'linewidth',linewid,'linestyle','none');
    else        
    errorbar(xbins,y,e,'color',colz{n},'linewidth',linewid);
    end
end

if max(xbins)==size(x,ndims(x));
xlim([0.5 length(y)+0.5]);
end
set(gca,'xtick',xbins);
set(gca,'FontSize',14);
end


