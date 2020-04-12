  
function [L,num,aa,ca,ma,da,ec,stats] = seg_area(Ia)
%function [L,num,aa,ca,ma,ec,pe] = seg_area(Ia)
%Pick up parameters from bwimages
[L,num] = bwlabel(Ia,8);
stats = regionprops(L,'Area','ConvexArea','Eccentricity');
aa = cat(1,stats.Area);
ca = cat(1,stats.ConvexArea);
ma = ca-aa;
da = ma./aa;
ec = cat(1,stats.Eccentricity);
end
