function sw = seg_watershed(Is,threshold)
%ws = seg_watershed(Is,threshold)
%Using watershed to segment close bricks
%Parameters: Is = imput bwimage; threshold = segment threshold.
Is1 = bwdist(imcomplement(Is));
Is1 = (mat2gray(Is1));
[c,h] = imcontour(Is1,0.2:0.1:1);
%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2)
Is2 = imimposemin(imcomplement(Is1),Is1>threshold);
sw = watershed(Is2);
sw = sw & Is;
% figure(10),hold on
% subplot(221),imshow(Is)
% subplot(222),imshow(Is1)
% subplot(223),imshow(Is2)
% subplot(224),imshow(sw)
end
