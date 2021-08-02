%% 2021/02/22
close all;
im = imread('D:/1.jpg');

% im = rgb2gray(im);
shape = size(im)
H = shape(1)
W = shape(2)

outDim = [floor(W), floor(H)]
% the correspondences need at least four points
Zp = [217, 39; 204, 95;  174, 223; 648, 402] % (x, y) in each row
Zs = [283, 54; 166, 101; 198, 250; 666, 372]

figure; imshow(im); hold on;
plot(Zp(:, 1), Zp(:, 2), 'ro')
plot(Zs(:, 1), Zs(:, 2), 'bo')
line([Zp(:, 1)'; Zs(:, 1)'], [Zp(:, 2)'; Zs(:, 2)']);

interp.method = 'nearest';
interp.radius = 10;
interp.power = 2;
% Xw, Yw are the transformed coordinates using warp. here I found Xw and Yw are reverse.
[Xw, Yw, imgw, imgwr, map] = tpswarp(im, outDim, Zp, Zs, interp); 

figure; imshow(uint8(imgw)); hold on;
plot(Zs(:, 1), Zs(:, 2), 'bo')
%plot(Yw, Xw);
plot(Yw(Zp(:, 2)*W+Zp(:, 1)), Xw(Zp(:, 2)*W+Zp(:, 1)), 'yo')
figure; imshow(uint8(imgw));

return
