%range of a matrix
function r=range(x)

if ~iscell(x)

x=x(:);
r(1)=min(x);
r(2)=max(x);
else
% cell
for n=1:length(x);
    xx=x{n};
    if ~iscell(xx)    
    rr(n,1)=min(xx(:));
    rr(n,2)=max(xx(:));
    else
        % cell within celll
        for n1=1:length(xx)        
        xxx=xx{n1};
        rrr(n,n1,1)=min(xxx(:));
        rrr(n,n1,2)=max(xxx(:));
        end
        rr(n,1)=min(rrr(n,:,1));
        rr(n,2)=max(rrr(n,:,2));
    end
        
end

r(1)=min(rr(:,1));
r(2)=max(rr(:,2));
end

