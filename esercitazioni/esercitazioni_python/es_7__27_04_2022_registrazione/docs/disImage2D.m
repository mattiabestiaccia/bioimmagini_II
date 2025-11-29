% generate a sysntetic MR image with a random misalignment
% dim: image dimension
% dis: 1: impose misalignement
% PhantomImage: Phantom Image
% par: imposed misalignment

function [disImage par]=disImage2D(I)

%genearate tx, ty, theta misalignment in the range [-dimx/10:dimx/10 -dimy/10:dimy/10,-180:180]
sz=size(I);
par=[sz(1)*(rand(1)-0.5)/10,sz(2)*(rand(1)-0.5)/10,120*(rand(1)-0.5)]; %transformation parameters
rad = par(3)*pi/180;


T = [[cos(rad) sin(rad) 0] ; [-sin(rad) cos(rad) 0] ; [par(1) par(2) 1]]; % transform matrix

TM=affine2d(T);
R=imref2d(size(I),[-sz(1) sz(1)],[-sz(1) sz(1) ]); % reference
[disImage RA]=imwarp(I,R,TM,'nearest','OutputView',R);




