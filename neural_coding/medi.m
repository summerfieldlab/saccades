function x = medi(y);
% medi finds the midpoints of a vector

if ~isinf(y(1));
x = y(1)+cumsum(diff(y)) - (diff(y)/2);
else
    yy=y(2:end-1);
    x = yy(1)+cumsum(diff(yy)) - (diff(yy)/2);
    x=[-Inf x Inf];
end
    