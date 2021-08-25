function [Xw, Yw, imgw, imgwr, map] = tpswarp(img, outDim, Zp, Zs, interp)
%
% Description: Thin-Plane spline warping of the input image (img) to
% output image (imgw). The warping is defined by a set of reference points
% (Zp=[Xp Yp]) on the [img] image and a set of corresponding points (Zs)
% on the (imgw) image. The warping will translate Zp in img to Zs in imgw.
%
% Input:
% img - input image
% outDim - Output canvas ([W H])
% Zp - landmark in img
% Zs - landmark in canvas
% interp.method - interpolation mode('nearest', 'invdist', 'none')
% interp.radius - Max radius for nearest neighbor interpolation or
%                 Radius of inverse weighted interpolation
% interp.power - power for inverse weighted interpolation
%
% Output:
% imgw - warped image with no holes
% imgwr - warped image with holes
% map - Map of the canvas with 0 indicating holes and 1 indicating pixel
%
% Reference:
% F.L. Bookstein, "Principal Warps: Thin-Plate splines and the
% decomposition of deformations", IEEE Transaction on Pattern Analysis and
% Machine Intelligence, Vol. 11, No. 6, June 1989
%
% Author: Fitzgerald J Archibald
% Date: 07-Apr-09

%% Initialization
NPs = size(Zp,1); % number of landmark points

imgH = size(img,1); % height
imgW = size(img,2); % width

outH = outDim(2);
outW = outDim(1);

% landmark in input
Xp = Zp(:,1)';
Yp = Zp(:,2)';

% landmark in output (homologous)
Xs = Zs(:,1)';
Ys = Zs(:,2)';

%% RR: remove zero keypoints
if 1
    idx = Xp>0 & Yp>0;
    Xp = Xp(idx);
    Yp = Yp(idx);
    Xs = Xs(idx);
    Ys = Ys(idx);
    
    NPs = sum(idx);
end
%% \RR

%% Algebra of Thin-plate splines

if 1
    % Compute thin-plate spline mapping [W|a1 ax ay] using landmarks
    [wL]=computeWl(Xp, Yp, NPs);

else  % uncertainty
    d_mean = mean(sqrt((Xp-Xs).^2 + (Yp-Ys).^2))
    d_med = median(sqrt((Xp-Xs).^2 + (Yp-Ys).^2))
    d_max = max(sqrt((Xp-Xs).^2 + (Yp-Ys).^2))
    l =0.5*d_med %d_mean * 2
    beta = 1
    % uncertainty = [1 1 1 1] 
    % uncertainty = [5, 3, 2, 2]
    uncertainty = [1, 5, 1, 1]

    D = 1 ./ uncertainty
    D = D / sum(D)  % normalize
    D = D.^beta  % adjust contrast
    D = D ./ sum(D)   % normalize
    D_inv = 1 ./ D
    D_inv = diag(D_inv)


%     s = sum(uncertainty)
%     D_inv = uncertainty ./ s   % normalize
%     D_inv = D_inv.^beta  % adjust contrast
%     D_inv = D_inv ./ sum(D_inv)   % normalize
%     % D_inv = s .* D_inv
%     D_inv = diag(D_inv)

    if l == 0
        lambda = 0
    else
        lambda = radialBasis(l) % 2 * l * l * log(l)
    end
    np = NPs
    rXp = repmat(Xp(:),1,np); % 1xNp to NpxNp
    rYp = repmat(Yp(:),1,np); % 1xNp to NpxNp

    wR = sqrt((rXp-rXp').^2 + (rYp-rYp').^2); % compute r(i,j)

    wK = radialBasis(wR); % compute [K] with elements U(r)=r^2 * log (r^2)
    %wK = wK + lambda*eye(np);
    wK = wK + lambda*D_inv  % uncertainty
    wP = [ones(np,1) Xp(:) Yp(:)]; % [P] = [1 xp' yp'] where (xp',yp') are n landmark points (nx2)
    wL = [wK wP;wP' zeros(3,3)]; % [L] = [[K P];[P' 0]]
end

wY = [Xs(:) Ys(:); zeros(3,2)]; % Y = ( V| 0 0 0)'   where V = [G] where G is landmark homologous (nx2) ; Y is col vector of length (n+3)
wW = inv(wL)*wY; % (W|a1 ax ay)' = inv(L)*Y

% Thin-plate spline mapping (Map all points in the plane)
% f(x,y) = a1 + ax * x + ay * y + SUM(wi * U(|Pi-(x,y)|)) for i = 1 to n
[Xw, Yw]=tpsMap(wW, imgH, imgW, Xp, Yp, NPs);

%% Warping

% input grid for warping
[X Y] = meshgrid(1:imgW,1:imgH); % HxW

% Nearest neighbor or inverse distance weighted based interpolation
[imgw,imgwr,map] = interp2d(X(:), Y(:), img, Xw, Yw, outH, outW, interp);

return

%% [L] = [[K P];[P' 0]]
% np - number of landmark points
% (xp, yp) - coordinate of landmark points
function [wL]=computeWl(xp, yp, np)

rXp = repmat(xp(:),1,np); % 1xNp to NpxNp
rYp = repmat(yp(:),1,np); % 1xNp to NpxNp

wR = sqrt((rXp-rXp').^2 + (rYp-rYp').^2); % compute r(i,j)

wK = radialBasis(wR); % compute [K] with elements U(r)=r^2 * log (r^2)
wP = [ones(np,1) xp(:) yp(:)]; % [P] = [1 xp' yp'] where (xp',yp') are n landmark points (nx2)
wL = [wK wP;wP' zeros(3,3)]; % [L] = [[K P];[P' 0]]

return


%% Mapping: f(x,y) = a1 + ax * x + ay * y + SUM(wi * U(|Pi-(x,y)|)) for i = 1 to n
% np - number of landmark points
% (xp, yp) - coordinate of landmark points
function [Xw, Yw]=tpsMap(wW, imgH, imgW, xp, yp, np)

[X Y] = meshgrid(1:imgW,1:imgH); % HxW
X=X(:)'; % convert to 1D array by reading columnwise (NWs=H*W)
Y=Y(:)'; % convert to 1D array (NWs)
NWs = length(X); % total number of points in the plane

% all points in plane
rX = repmat(X,np,1); % Np x NWs
rY = repmat(Y,np,1); % Np x NWs

% landmark points
rxp = repmat(xp(:),1,NWs); % 1xNp to Np x NWs
ryp = repmat(yp(:),1,NWs); % 1xNp to Np x NWs

% Mapping Algebra
wR = sqrt((rxp-rX).^2 + (ryp-rY).^2); % distance measure r(i,j)=|Pi-(x,y)|

wK = radialBasis(wR); % compute [K] with elements U(r)=r^2 * log (r^2)
wP = [ones(NWs,1) X(:) Y(:)]'; % [P] = [1 x' y'] where (x',y') are n landmark points (nx2)
wL = [wK;wP]'; % [L] = [[K P];[P' 0]]

Xw  = wL*wW(:,1); % [Pw] = [L]*[W]
Yw  = wL*wW(:,2); % [Pw] = [L]*[W]

% xp
% xp2 = repmat(xp(:), 1, 4)
% yp2 = repmat(yp(:), 1, 4)
% d = sqrt((xp2 - xp2').^2 + (yp2-yp2').^2)
% r = radialBasis(d)
% ph = [ones(4, 1), xp(:), yp(:)]
% l = [r, ph]
% pp = l * wW
% Xw((xp(:)-1)*imgH+yp(:))
% Yw((xp(:)-1)*imgH+yp(:))
return

%% k=(r^2) * log(r)
function [ko]=radialBasis(ri)

r1i = ri;
r1i(find(ri==0))=realmin; % Avoid log(0)=inf, the main diagonal elements are equal to 0
%ko = 2*(ri.^2).*log(r1i);  % namely equal to (ri.^2).*log(r1i.^2)
ko = (ri.^2).*log(r1i);

return
