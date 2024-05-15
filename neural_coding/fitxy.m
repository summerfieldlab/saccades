function [f newx p s]=fitxy(x,y,polynom);
% function [f newx]=fitxy(x,y,polynom);
% figure;scatter(x,y);
% hold on;plot(newx,f)

rx=ranger(x);
intv=(max(x)-min(x))./(length(x)-1);
newx=rx(1):intv:rx(2);
%newx=linspace(x(1),x(end),length(x)-1);

[p s]=polyfit(x,y,polynom);
f=polyval(p,newx);



