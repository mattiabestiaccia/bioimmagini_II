close all
clear all

% create ideal image with 6 patterns
dim=512;
image = zeros(dim,dim);

image(:,:) = 50 ;
image(50:100,50:100)= 120;
image(101:180,101:450)= 200;
image(200:500,200:350)= 90;
image(230:270,230:270)= 250;
image(5:400,450:500)= 150;

% add gaussian noise
sigma =5;
imageN= image + sigma*randn(512,512); 
%display noisy image and histogram
bins=256
figure('NumberTitle', 'off', 'Name', 'Gaussian Noise');
axis('image')
subplot(1,2,1)
colormap gray
imagesc(imageN)
axis('image')
subplot(1,2,2)
hist(imageN(:),bins);


fimage= stdfilt(imageN,true(5)); % compute SD map
figure('NumberTitle', 'off', 'Name', 'SD Map');
axis('image') % preserve image proportions
subplot(1,2,1)
colormap gray
imagesc(fimage)
%axis('image') % preserve image proportions
subplot(1,2,2)
[h,x]=hist(fimage(:),100) % cumpute histogram
plot(x,h)

sigma  % true value
sigmaMean=mean(fimage(:))  % mean of SD map
sigmaMedian=median(fimage(:)) % median of SD map
[m,id]=max(h); % position of histogram maximum
sigmaMax = x(id)

figure('NumberTitle', 'off', 'Name', 'Sigma Calculation Methods');
c = categorical({'True','Mean','Median','Max Hist'});
values = [sigma sigmaMean sigmaMedian sigmaMax];
bar(c,values)
