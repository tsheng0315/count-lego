function [numA,numB] = count_lego(I)
% Using several methods to detect the number of target lego bricks
% References:
% Segment by hsv image
% http://wenku.baidu.com/link?url=aJiQ2a0DufrqsXQwtdsbNFECNvUGdZYwaJ-Kf-Go381FnOnQ95t7zAjdUHMoRTX-F2bg-vZ9f0jIjNF0Uq88ZJYrb4Xab_Qfg8AebZ-wbIe
% Segment by watershed (Page 3)
% http://www.ilovematlab.cn/thread-221402-1-1.html
% Rigionprops
% https://cn.mathworks.com/help/images/ref/regionprops.html
I = im2double(I);
Ig = rgb2gray(I);
Ig = medfilt2(Ig,[3 3]);

g = I(:,:,2);
%% Transfer to hsv
hsv=rgb2hsv(I);
h=hsv(:,:,1);
s=hsv(:,:,2);
%v=hsv(:,:,3);
% figure(2),subplot(221);imshow(I)
% figure(2),subplot(222);imshow(h)
% figure(2),subplot(223);imshow(s)
% figure(2),subplot(224);imshow(g)

%% =============================Lego_A==============================
%% Morphological operations
a1 = h>0.35&h<0.55;
a1 = a1.*(s>0.5&s<0.95);
se = strel('disk',4);
a2=imdilate(a1,se);
a2=imerode(a2,se);
a3 = imfill(a2,'holes');
a3 = imerode(a3,strel('disk',10));
a3 = imopen(a3,strel('disk',10));
a3 = bwareaopen(a3,1000);
a3 = imerode(a3,strel('disk',8));
% figure(3),subplot(221);imshow(a3)

%% Segementation
[L,num,aa,~,ma,~,~,~] = seg_area(a3);
L1 = zeros(size(L));
m1 = L1; m2 = m1; m3 = m2;
% Pick up the areas which have multipe bricks
 for i = 1:num
     if ma(i)>7000&&aa(i)>36000
         idx = find(L == i);
         L(idx) = 0;
         L1(idx) = 1;
     end
 end
 % Prior processing
 me = L1.*Ig;
 idx0 = find(L1==0);
 ne = edge(me,'sobel','nothinning');
 ne = imclose(ne,strel('disk',2));
 n2 = L1 - ne;
 n2(find(n2==-1))=0;
 n2 = imerode(n2,strel('disk',4));
 [L1,num1,~,~,ma1,da1,ec1,~] = seg_area(n2);
 for j = 1:num1
     idx = find(L1==j);
     if da1(j)>5 
     L1(idx) = 0;
     n2(idx) = 0;
     end
     if ma1(j)<100&&ec1(j)<0.88
         m1(idx) = 1;
     end
 end
 n3 = imdilate(m1,strel('disk',30));
 n3 = n3|n2;
 n3 = imfill(n3,'holes');
 n3(idx0)=0;
 n3 = bwareaopen(n3,500);
%  figure(6),imagesc(n3)
 
% Segement bricks by using Watershed
 [L2,num2,~,~,ma2,~,~,~] = seg_area(n3);
for k = 0.4:0.1:0.9
    for i = 1:num2
        idx = find(L2==i);
        if ma2(i)>5000
            m2(idx)=1;
            L2(idx) = 0;
        else
            L2(idx) = 1;
        end
    end
    if max(max(m2))>0
    sw = seg_watershed(m2,k);
    L2 = L2|sw;
    [L2,num2,~,~,ma2,~,~,~] = seg_area(L2);
    else
        break
    end
end

L = L|L2;
L = imerode(L,strel('disk',5));
L = bwareaopen(L,1000);
%figure(7),imshow(L)
[L3,num3,aa3,~,~,da3,ec3,~] = seg_area(L);
for j = 1:num3
    idx = find(L3==j);
    if ec3(j)>0.81&&ec3(j)<0.95&&da3(j)<0.21&&aa3(j)>3000
        m3(idx)=1;
    end
end
%figure(8),imshow(m3)
m_a = m3;
[~,numA] = bwlabel(m3,8);
disp(numA)

%% =============================Lego_B==============================
%% Morphological operations
b1 = h>0.05&h<0.165;
b1 = b1.*(s>0.55);
b1 = b1.*(g>0.3);
se = strel('disk',4);
b2=imdilate(b1,se);
b2=imerode(b2,se);
b3 = imfill(b2,'holes');
b3 = imopen(b3,se);
b3 = bwareaopen(b3,10000);
%figure(3);subplot(221);imshow(b3)

%% Segementation
m1 = zeros(size(b3));
[L,num,aa,~,ma,da,~,~] = seg_area(b3);
% Pick up the areas which have multipe bricks
m = zeros(size(b3));
 for i = 1:num
     if (da(i)*ma(i)>4000||da(i)>0.14&&aa(i)>40000)
         idx = find(L == i);
         L(idx) = 0;
         m(idx) = 1;
     end
 end
if max(max(m))>0
 m = seg_watershed(m,0.7);
 m = imerode(m,strel('disk',10));
 L = L|m;
[L,num,~,~,ma,da,~,~] = seg_area(L);
end
m = zeros(size(b3));
 for i = 1:num
     if (da(i)>0.14&&ma(i)>8000)
         idx = find(L == i);
         L(idx) = 0;
         m(idx) = 1;
     end
 end
 L = imerode(L,strel('disk',5));
 %figure,imagesc(m)

 %% Segement by edge
 if max(max(m))>0
 % Prior processing
 me = m.*Ig;
 ne = edge(me,'sobel','nothinning');
 ne = imclose(ne,strel('disk',2));
 n2 = m - ne;
 idx = find(n2==-1);
 n2(idx)=0;
 n2 = imerode(n2,strel('disk',4));
 [L1,num1,aa1,~,~,da1,~,~] = seg_area(n2);
 %figure(5),imagesc(L1)
 for j = 1:num1
     idx = find(L1==j);
     if da1(j)>5
     L1(idx) = 0;
     n2(idx) = 0;
     end
     if aa1(j)<3000
         m1(idx) = 1;
     end
 end
 n3 = imdilate(m1,strel('disk',30));
 n3 = n3|n2;
 n3 = imfill(n3,'holes');
 idx0 = find(m==0);
 n3(idx0)=0;    
 n3 = bwareaopen(n3,500);
 sw = seg_watershed(n3,0.7);
L = sw|L;
%figure(6),imshow(L)

 end
 
%% Select target objects
m4 = zeros(size(L));m3 = m4;
m4(1,:)=1; m4(end,:)=1;m4(:,1)=1;m4(:,end)=1;
[L3,num3,aa3,~,~,da3,ec3,~] = seg_area(L);
for j = 1:num3
    idx = find(L3==j);
    m5 = zeros(size(L));
    m5(idx) = 1;
    m2 = m4&m5;
    idy = find(m2>0);
    if length(idy)>50||aa3(j)<8000
        continue
    end
    if (ec3(j)<0.8&&da3(j)<0.2&&aa3(j)>10000&&aa3(j)<35000)||(ec3(j)<0.9&&da3(j)<0.25&&aa3(j)<15000)||(ec3(j)<0.5&&da3(j)<0.02)
        m3(idx)=1;
    end
end
%figure(8),imshow(m3)
m_b = m3;
[~,numB] = bwlabel(m3,8);
disp(numB)

m_AB = m_b-m_a;
figure,imagesc(m_AB)
end
