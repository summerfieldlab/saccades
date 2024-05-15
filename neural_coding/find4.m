function [i]=find4(in);


if ndims(in)==2;

[ii jj]=find(in);
i=[ii(1) jj(1)];

end

if ndims(in)==3;

for n=1:3;
    j=setdiff(1:3,n);
    dog=mean(mean(in,j(1)),j(2));
    ii=find(dog>0);
    i(n)=ii(1);
end

end

if ndims(in)==4;

    
for n=1:4;
    j=setdiff(1:4,n);
    dog=mean(mean(mean(in,j(1)),j(2)),j(3));
    ii=find(dog>0);
    i(n)=ii(1);
end

end

if ndims(in)==5;
    
    for n=1:5;
    j=setdiff(1:5,n);
    dog=mean(mean(mean(mean(in,j(1)),j(2)),j(3)),j(4));
    i(n)=first(find(dog>0));
    end


end


if ndims(in)==6;
    
    for n=1:6;
    j=setdiff(1:6,n);
    dog=mean(mean(mean(mean(mean(in,j(1)),j(2)),j(3)),j(4)),j(5));
    i(n)=find(dog>0);
    end


end


if ndims(in)==7;
    
    for n=1:7;
    j=setdiff(1:7,n);
    dog=mean(mean(mean(mean(mean(mean(in,j(1)),j(2)),j(3)),j(4)),j(5)),j(6));
    i(n)=find(dog>0);
    end


end