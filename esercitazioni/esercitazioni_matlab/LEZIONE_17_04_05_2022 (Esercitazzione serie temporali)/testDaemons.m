% esemplification of daemons based registration
%% 
close all
clear all


A= zeros(512,512);       % create two sample images
A(100:400,100:450)=255;
A(350:360,270:360)=0;
A(260:360,350:360)=0;

B=A;
B(100:200,100:300)=0;

s=56;                   % add noise        
A=A+s*rand(512);
B=B+s*rand(512);

figure('Name','Original Images')
colormap gray
subplot(1,2,1)
imagesc(A)
title('Reference Image R')
set(gca,'xtick',[],'ytick',[])
axis image
subplot(1,2,2)
imagesc(B)
axis image
title('Floating Image F')
set(gca,'xtick',[],'ytick',[])
sgtitle('daemons based registration')
%% 


%% 
figure('Name','Registration Procedure')
steps=(1:15).^2;
for i=steps
    [D,C] = imregdemons(B,A,i,'AccumulatedFieldSmoothing',1.5,'PyramidLevels',7);


    colormap gray
    subplot(2,2,1)
    imagesc(A)
    axis image
    set(gca,'xtick',[],'ytick',[])
    title('R image')
    subplot(2,2,2)
    imagesc(C)
    axis image
    set(gca,'xtick',[],'ytick',[])
    title('F image')

    colormap gray
    subplot(2,2,3)
    imshowpair(B,D(:,:,1))
    axis image
    title('Displacement D (X component)')
    subplot(2,2,4)
    imshowpair(B,D(:,:,2))
    axis image
    title('Displacement D (Y component)')
    sgtitle(['Number of Iterations = ' num2str(steps(sqrt(i)))])
    
    pause(5)
    drawnow
end
%% 


