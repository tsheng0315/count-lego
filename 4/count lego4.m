%% Explanation of the code contents.
% This code attempts to count the number of blue rectangles (numA) and
% number of red squares (numB).

%% Procedure:
% 1) Download image as I=imread('filename') and invoke count_lego(I) function
% 2) Use K-Means Clustering to segment the image into different colors
% 3) Determine which clusters contain blue and red colors
% 4) Select blue and red clusters containing the most number of blue and
% red elements, respectively
% 5) Edge detection using Canny edge detector for red and blue colors
% 6) Dilation of objects and filling holes
% 7) Apply watershed method to separate touching objects and determine
% boundaries of objects
% 8) Compute properties of each identified object (areas, perimeters, etc.)
% 9) Classify each object as square/rectangle/circle/triangle
% 10) If a blue object is rectangular or red object is square, then
% we can judge whether an object is blue rectangle or red square based on
% their perimeters and area properties
% 11) Count the objects that fall under those categories

%% Main Function
function [numA,numB]=lego_count(I)

close all;
figure(1), imagesc(I);

%% Perform K-means clustering to segment the objects
% Part of the code was taken and/or an idea (or method) was used from
% https://stackoverflow.com/questions/25691735/matlab-color-based-segmentation
cform = makecform('srgb2lab');
lab_he = applycform(I,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 6;
% repeat the clustering 3 times to avoid local minima
rng default;
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', 'Replicates',3);

pixel_labels = reshape(cluster_idx,nrows,ncols);
%imshow(pixel_labels,[]), title('image labeled by cluster index');
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = I;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

figure(2), title('Clustering by Color');
figure(2), subplot(2,3,1), imagesc(segmented_images{1});
figure(2), subplot(2,3,2), imagesc(segmented_images{2});
figure(2), subplot(2,3,3), imagesc(segmented_images{3});
figure(2), subplot(2,3,4), imagesc(segmented_images{4});
figure(2), subplot(2,3,5), imagesc(segmented_images{5});
figure(2), subplot(2,3,6), imagesc(segmented_images{6});

%% Determine segmented images containing red and blue colors

% Part of the code was taken and/or an idea (or method) was used from
% https://stackoverflow.com/questions/37684903/matlab-how-to-detect-green-color-on-image

res_red = {};
res_blue = {};
for i1 = 1:nColors
    %Determine if images contain red and blue colors
    %determine red threshold (305 to 360) in hue channel
    red_range = [305,360]/360;
    blue_range = [200,260]/360;
    intensity = 0.5;

    %convert to HSV color space
    hsv_red = rgb2hsv(segmented_images{i1});
    hsv_blue = rgb2hsv(segmented_images{i1});

    %generate a region of interest (only areas which aren't black)
    relevanceMask_1 = rgb2gray(segmented_images{i1})>0;
    relevanceMask_2 = rgb2gray(segmented_images{i1})>0;
    
    %find pixels within the specified range in the H and V channels
    redAreasMask = hsv_red(:,:,1)>red_range(1) & hsv_red(:,:,1) < red_range(2) & hsv_red(:,:,3) > intensity;
    blueAreasMask = hsv_blue(:,:,1)>blue_range(1) & hsv_blue(:,:,1) < blue_range(2) & hsv_blue(:,:,3) > intensity;

    %return the mean in the relevance mask
    res_red = [res_red, sum(redAreasMask(:)) / sum(relevanceMask_1(:))];
    res_blue = [res_blue, sum(blueAreasMask(:)) / sum(relevanceMask_2(:))];
    i1 = i1 + 1;
end

% Select cluster which contains biggest number of red components
max_red = max(max(max(max(max(res_red{1}, res_red{2}), res_red{3}), res_red{4}), res_red{5}), res_red{6});
for i3 = 1:nColors
    if res_red{i3} == max_red;
        cluster_red = segmented_images{i3};
    end
    i3 = i3 + 1;
end

% Select cluster which contains biggest number of blue components
max_blue = max(max(max(max(max(res_blue{1}, res_blue{2}), res_blue{3}), res_blue{4}), res_blue{5}), res_blue{6});
for i4 = 1:nColors
    if res_blue{i4} == max_blue;
        cluster_blue = segmented_images{i4};
    end
    i4 = i4 + 1;
end

%% Perform edge enhancing and apply watershed method for red legos

% Part of the code was taken and/or an idea (or method) was used from
% https://www.mathworks.com/help/images/detecting-a-cell-using-image-segmentation.html
% https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/
% https://www.mathworks.com/matlabcentral/answers/308183-calculating-perimeter-of-object

% perform canny edge detection
grey_cluster2 = rgb2gray(cluster_red);
[~, threshold] = edge(grey_cluster2, 'canny');
fudgeFactor = 0.2;
BWs = edge(grey_cluster2,'canny', threshold * fudgeFactor);
%figure(5), imagesc(BWs), title('binary gradient mask');

% Perform dilation
se90 = strel('line', 3, 90);
se85 = strel('line', 3, 85);
se80 = strel('line', 3, 80);
se75 = strel('line', 3, 75);
se70 = strel('line', 3, 70);
se65 = strel('line', 3, 65);
se60 = strel('line', 3, 60);
se55 = strel('line', 3, 55);
se50 = strel('line', 3, 50);
se45 = strel('line', 3, 45);
se40 = strel('line', 3, 40);
se35 = strel('line', 3, 35);
se30 = strel('line', 3, 30);
se25 = strel('line', 3, 25);
se20 = strel('line', 3, 20);
se15 = strel('line', 3, 15);
se10 = strel('line', 3, 10);
se05 = strel('line', 3, 5);
se0 = strel('line', 3, 0);

BWsdil = imdilate(BWs, [se90 se85 se80 se75 se70 se65 se60 se55 se50 se45 se40 se35 se30 se25 se20 se15 se10 se05 se0]);
%figure(6), imshow(BWsdil), title('dilated gradient mask');

% Fill Interior Gaps
BWdfill = imfill(BWsdil, 'holes');

% Apply Watershed Method
bw = BWdfill;
L = watershed(bw);
Lrgb = label2rgb(L);
figure(4), imagesc(Lrgb)

figure(4), imagesc(imfuse(bw,Lrgb))
axis([10 175 15 155])

bw2 = ~bwareaopen(~bw, 10);
figure(4), imagesc(bw2)

D = -bwdist(~bw);
figure(4), imshow(D,[])

Ld = watershed(D);
figure(4), imagesc(label2rgb(Ld))

bw2 = bw;
bw2(Ld == 0) = 0;
figure(4), imagesc(bw2)

mask = imextendedmin(D,2);
imshowpair(bw,mask,'blend')

D2 = imimposemin(D,mask);
Ld2 = watershed(D2);
bw3 = bw;
bw3(Ld2 == 0) = 0;
figure(4), imagesc(bw3), title('Red objects in the image');

%% Determine the shapes of red legos and number of red squared legos

% Part of the code was taken and/or an idea (or method) was used from
% https://www.mathworks.com/matlabcentral/answers/116793-how-to-classify-shapes-of-this-image-as-square-rectangle-triangle-and-circle
% https://www.mathworks.com/matlabcentral/fileexchange/34767-a-suite-of-minimal-bounding-objects

%get outlines of each RED object
[B,L,N] = bwboundaries(bw3);
%get stats including perimeter, area and metrics for each shape
stats=  regionprops(L, 'Centroid', 'Area', 'Perimeter');
Centroid = cat(1, stats.Centroid);
Perimeter = cat(1,stats.Perimeter);
Area = cat(1,stats.Area);
CircleMetric = (Perimeter.^2)./(4*pi*Area);  %circularity metric
SquareMetric = NaN(N,1);
TriangleMetric = NaN(N,1);
%for each boundary, fit to bounding box, and calculate parameters
for k=1:N,
   boundary = B{k};
   [rx,ry,boxArea] = minboundrect( boundary(:,2), boundary(:,1));  %x and y are flipped in images
   %get width and height of bounding box
   width = sqrt( sum( (rx(2)-rx(1)).^2 + (ry(2)-ry(1)).^2));
   height = sqrt( sum( (rx(2)-rx(3)).^2+ (ry(2)-ry(3)).^2));
   aspectRatio = width/height;
   if aspectRatio > 1,  
       aspectRatio = height/width;  %make aspect ratio less than unity
   end
   SquareMetric(k) = aspectRatio;    %aspect ratio of box sides
   TriangleMetric(k) = Area(k)/boxArea;  %filled area vs box area
end
%define thresholds for each metric
%do in order of circle, triangle, square, rectangle to avoid assigning the
%same shape to multiple objects
isCircle =   (CircleMetric < 1.1);
isTriangle = ~isCircle & (TriangleMetric < 0.6);
isSquare =   ~isCircle & ~isTriangle & (SquareMetric > 0.75);
isRectangle= ~isCircle & ~isTriangle & ~isSquare;  %rectangle isn't any of these
%assign shape to each object
whichShape = cell(N,1);  
whichShape(isCircle) = {'Circle'};
whichShape(isTriangle) = {'Triangle'};
whichShape(isSquare) = {'Square'};
whichShape(isRectangle)= {'Rectangle'};
%label
RGB = label2rgb(L);
figure(4), imshow(RGB), title('Red objects'); hold on;
Combined = [CircleMetric, SquareMetric, TriangleMetric];
number_of_red_cubes1 = 0;
for k=1:N
   %display metric values and shape name next to an object
   Txt = sprintf('C=%0.3f S=%0.3f T=%0.3f',  Combined(k,:));
   text( Centroid(k,1)-20, Centroid(k,2), Txt);
   text( Centroid(k,1)-20, Centroid(k,2)+20, whichShape{k});
   if isSquare(k) == 1
      if Perimeter(k) > 400;
          if Perimeter(k) < 650;
              number_of_red_cubes1 = number_of_red_cubes1 + 1;
          end
      end
   end
end

%% Perform edge enhancing and apply watershed method for blue legos

% Part of the code was taken and/or an idea (or method) was used from
% https://www.mathworks.com/help/images/detecting-a-cell-using-image-segmentation.html
% https://blogs.mathworks.com/steve/2013/11/19/watershed-transform-question-from-tech-support/
% https://www.mathworks.com/matlabcentral/answers/308183-calculating-perimeter-of-object

% perform canny edge detection
grey_cluster_blue = rgb2gray(cluster_blue);
[~, threshold_blue] = edge(grey_cluster_blue, 'canny');
fudgeFactor = .5;
BWs_blue = edge(grey_cluster_blue,'canny', threshold_blue * fudgeFactor);
%figure(8), imshow(BWs_blue), title('binary gradient mask for blue objects');

% Perform dilation
se90 = strel('line', 3, 90);
se85 = strel('line', 3, 85);
se80 = strel('line', 3, 80);
se75 = strel('line', 3, 75);
se70 = strel('line', 3, 70);
se65 = strel('line', 3, 65);
se60 = strel('line', 3, 60);
se55 = strel('line', 3, 55);
se50 = strel('line', 3, 50);
se45 = strel('line', 3, 45);
se40 = strel('line', 3, 40);
se35 = strel('line', 3, 35);
se30 = strel('line', 3, 30);
se20 = strel('line', 3, 20);
se15 = strel('line', 3, 15);
se10 = strel('line', 3, 10);
se05 = strel('line', 3, 5);
se0 = strel('line', 3, 0);

BWsdil_blue = imdilate(BWs_blue, [se90 se85 se80 se75 se70 se65 se60 se55 se50 se45 se40 se35 se30 se20 se15 se10 se0]);
%figure(6), imshow(BWsdil), title('dilated gradient mask');

% Fill Interior Gaps
BWdfill_blue = imfill(BWsdil_blue, 'holes');

% Apply Watershed Method
bw_blue = BWdfill_blue;

L_blue = watershed(bw_blue);

Lrgb_blue = label2rgb(L_blue);
figure(5), imshow(Lrgb_blue)

imshow(imfuse(bw_blue,Lrgb_blue))
axis([10 175 15 155])

bw2_blue = ~bwareaopen(~bw_blue, 10);
imshow(bw2)

D_blue = -bwdist(~bw_blue);
imshow(D_blue,[])

Ld_blue = watershed(D_blue);
imshow(label2rgb(Ld_blue))

bw2_blue = bw_blue;
bw2_blue(Ld_blue == 0) = 0;
imshow(bw2_blue)

mask_blue = imextendedmin(D_blue,2);
imshowpair(bw_blue,mask_blue,'blend')

D2_blue = imimposemin(D_blue,mask_blue);
Ld2_blue = watershed(D2_blue);
bw3_blue = bw_blue;
bw3_blue(Ld2_blue == 0) = 0;
imagesc(bw3_blue), title('Blue objects in the image');

%% Determine the shapes of blue legos and number of blue rectangular legos

% Part of the code was taken and/or an idea (or method) was used from
% https://www.mathworks.com/matlabcentral/answers/116793-how-to-classify-shapes-of-this-image-as-square-rectangle-triangle-and-circle
% https://www.mathworks.com/matlabcentral/fileexchange/34767-a-suite-of-minimal-bounding-objects

%get outlines of each object
[B,L,N] = bwboundaries(bw3_blue);
%get stats including perimeter, area and metrics for each shape
stats=  regionprops(L, 'Centroid', 'Area', 'Perimeter');
Centroid = cat(1, stats.Centroid);
Perimeter_b = cat(1,stats.Perimeter);
Area = cat(1,stats.Area);
CircleMetric = (Perimeter_b.^2)./(4*pi*Area);  %circularity metric
SquareMetric = NaN(N,1);
TriangleMetric = NaN(N,1);
%for each boundary, fit to bounding box, and calculate some parameters
for k=1:N,
   boundary = B{k};
   [rx,ry,boxArea] = minboundrect( boundary(:,2), boundary(:,1));  %x and y are flipped in images
   %get width and height of bounding box
   width = sqrt( sum( (rx(2)-rx(1)).^2 + (ry(2)-ry(1)).^2));
   height = sqrt( sum( (rx(2)-rx(3)).^2+ (ry(2)-ry(3)).^2));
   aspectRatio = width/height;
   if aspectRatio > 1,  
       aspectRatio = height/width;  %make aspect ratio less than unity
   end
   SquareMetric(k) = aspectRatio;    %aspect ratio of box sides
   TriangleMetric(k) = Area(k)/boxArea;  %filled area vs box area
end
%define thresholds for each metric
%do in order of circle, triangle, square, rectangle to avoid assigning the
%same shape to multiple objects
isCircle =   (CircleMetric < 1.1);
isTriangle = ~isCircle & (TriangleMetric < 0.6);
isSquare =   ~isCircle & ~isTriangle & (SquareMetric > 0.75);
isRectangle= ~isCircle & ~isTriangle & ~isSquare;  %rectangle isn't any of these
%assign shape to each object
whichShape = cell(N,1);  
whichShape(isCircle) = {'Circle'};
whichShape(isTriangle) = {'Triangle'};
whichShape(isSquare) = {'Square'};
whichShape(isRectangle)= {'Rectangle'};
%label
RGB = label2rgb(L);
figure(5), imshow(RGB), title('Blue objects'); hold on;
Combined = [CircleMetric, SquareMetric, TriangleMetric];
number_of_blue_rectangles1 = 0;
for k=1:N
   %display metric values and shape name next to object
   Txt = sprintf('C=%0.3f S=%0.3f T=%0.3f',  Combined(k,:));
   text( Centroid(k,1)-20, Centroid(k,2), Txt);
   text( Centroid(k,1)-20, Centroid(k,2)+20, whichShape{k});
   if isRectangle(k) == 1
      if Perimeter_b(k) > 650;
          if Perimeter_b(k) < 980;
              number_of_blue_rectangles1 = number_of_blue_rectangles1 + 1;
          end
      end
   end
end

 
%%
numB = number_of_red_cubes1;
numA = number_of_blue_rectangles1;

%% Display number of blue rectangles and red squares 
numA
numB
end
%% 
% Function below was downloaded from: 
% https://www.mathworks.com/matlabcentral/fileexchange/34767-a-suite-of-minimal-bounding-objects

function [rectx,recty,area,perimeter] = minboundrect(x,y,metric)
% minboundrect: Compute the minimal bounding rectangle of points in the plane
% usage: [rectx,recty,area,perimeter] = minboundrect(x,y,metric)
%
% arguments: (input)
%  x,y - vectors of points, describing points in the plane as
%        (x,y) pairs. x and y must be the same lengths.
%
%  metric - (OPTIONAL) - single letter character flag which
%        denotes the use of minimal area or perimeter as the
%        metric to be minimized. metric may be either 'a' or 'p',
%        capitalization is ignored. Any other contraction of 'area'
%        or 'perimeter' is also accepted.
%
%        DEFAULT: 'a'    ('area')
%
% arguments: (output)
%  rectx,recty - 5x1 vectors of points that define the minimal
%        bounding rectangle.
%
%  area - (scalar) area of the minimal rect itself.
%
%  perimeter - (scalar) perimeter of the minimal rect as found
%
%
% Note: For those individuals who would prefer the rect with minimum
% perimeter or area, careful testing convinces me that the minimum area
% rect was generally also the minimum perimeter rect on most problems
% (with one class of exceptions). This same testing appeared to verify my
% assumption that the minimum area rect must always contain at least
% one edge of the convex hull. The exception I refer to above is for
% problems when the convex hull is composed of only a few points,
% most likely exactly 3. Here one may see differences between the
% two metrics. My thanks to Roger Stafford for pointing out this
% class of counter-examples.
%
% Thanks are also due to Roger for pointing out a proof that the
% bounding rect must always contain an edge of the convex hull, in
% both the minimal perimeter and area cases.
%
%
% Example usage:
%  x = rand(50000,1);
%  y = rand(50000,1);
%  tic,[rx,ry,area] = minboundrect(x,y);toc
%
%  Elapsed time is 0.105754 seconds.
%
%  [rx,ry]
%  ans =
%      0.99994  -4.2515e-06
%      0.99998      0.99999
%   2.6441e-05            1
%  -5.1673e-06   2.7356e-05
%      0.99994  -4.2515e-06
%
%  area
%  area =
%      0.99994
%
%
% See also: minboundcircle, minboundtri, minboundsphere
%
%
% Author: John D'Errico
% E-mail: woodchips@rochester.rr.com
% Release: 3.0
% Release date: 3/7/07

% default for metric
if (nargin<3) || isempty(metric)
  metric = 'a';
elseif ~ischar(metric)
  error 'metric must be a character flag if it is supplied.'
else
  % check for 'a' or 'p'
  metric = lower(metric(:)');
  ind = strmatch(metric,{'area','perimeter'});
  if isempty(ind)
    error 'metric does not match either ''area'' or ''perimeter'''
  end
  
  % just keep the first letter.
  metric = metric(1);
end

% preprocess data
x=x(:);
y=y(:);

% not many error checks to worry about
n = length(x);
if n~=length(y)
  error 'x and y must be the same sizes'
end

% start out with the convex hull of the points to
% reduce the problem dramatically. Note that any
% points in the interior of the convex hull are
% never needed, so we drop them.
if n>3
  edges = convhull(x,y);

  % exclude those points inside the hull as not relevant
  % also sorts the points into their convex hull as a
  % closed polygon
  
  x = x(edges);
  y = y(edges);
  
  % probably fewer points now, unless the points are fully convex
  nedges = length(x) - 1;
elseif n>1
  % n must be 2 or 3
  nedges = n;
  x(end+1) = x(1);
  y(end+1) = y(1);
else
  % n must be 0 or 1
  nedges = n;
end

% now we must find the bounding rectangle of those
% that remain.

% special case small numbers of points. If we trip any
% of these cases, then we are done, so return.
switch nedges
  case 0
    % empty begets empty
    rectx = [];
    recty = [];
    area = [];
    perimeter = [];
    return
  case 1
    % with one point, the rect is simple.
    rectx = repmat(x,1,5);
    recty = repmat(y,1,5);
    area = 0;
    perimeter = 0;
    return
  case 2
    % only two points. also simple.
    rectx = x([1 2 2 1 1]);
    recty = y([1 2 2 1 1]);
    area = 0;
    perimeter = 2*sqrt(diff(x).^2 + diff(y).^2);
    return
end
% 3 or more points.

% will need a 2x2 rotation matrix through an angle theta
Rmat = @(theta) [cos(theta) sin(theta);-sin(theta) cos(theta)];

% get the angle of each edge of the hull polygon.
ind = 1:(length(x)-1);
edgeangles = atan2(y(ind+1) - y(ind),x(ind+1) - x(ind));
% move the angle into the first quadrant.
edgeangles = unique(mod(edgeangles,pi/2));

% now just check each edge of the hull
nang = length(edgeangles);
area = inf;
perimeter = inf;
met = inf;
xy = [x,y];
for i = 1:nang
  % rotate the data through -theta 
  rot = Rmat(-edgeangles(i));
  xyr = xy*rot;
  xymin = min(xyr,[],1);
  xymax = max(xyr,[],1);
  
  % The area is simple, as is the perimeter
  A_i = prod(xymax - xymin);
  P_i = 2*sum(xymax-xymin);
  
  if metric=='a'
    M_i = A_i;
  else
    M_i = P_i;
  end
  
  % new metric value for the current interval. Is it better?
  if M_i<met
    % keep this one
    met = M_i;
    area = A_i;
    perimeter = P_i;
    
    rect = [xymin;[xymax(1),xymin(2)];xymax;[xymin(1),xymax(2)];xymin];
    rect = rect*rot';
    rectx = rect(:,1);
    recty = rect(:,2);
  end
end
% get the final rect

% all done

end % mainline end
