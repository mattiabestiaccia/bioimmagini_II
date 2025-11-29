%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% esempio calcolo rumore su fantoccio MR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

I = double(dicomread('phantom.dcm')); % read DICOM image

nz = find(I ~= 0); % identify zero padding pixels

% ROI-based measurement
figure('NumberTitle', 'off', 'Name', 'Phantom Image');
colormap gray
imagesc(I)
roiOil = drawcircle('Color','g','Label','Oil')      % trace ROIs
roiWater = drawcircle('Color','r','Label','Water')
roiBack = drawcircle('Color','w','Label','Back')

oilMask = createMask(roiOil);                       % ROI mask
waterMask = createMask(roiWater);
backMask = createMask(roiBack);

id = find(oilMask ~= 0);             % calculate SD on masks
stdOil=std(I(id));
id = find(waterMask ~= 0);
stdWater=std(I(id));
id = find(backMask ~= 0);
stdBack=std(I(id));
stdBackCorr = stdBack*1.526;

data = [stdBack stdBackCorr stdWater stdOil]

f = figure;
uit = uitable(f,'Data',data,'Position',[20 120 462 100]);
uit.ColumnName = {'SD Bk','SD Bk Corr','SD Water','SD Oil'};
uit.RowName = {'Manual','Auto'};
%% 


%% 
sdMap=stdfilt(I,ones(3));
figure('NumberTitle', 'off', 'Name', 'SD map analysis');
subplot(1,2,1)
colormap gray
imagesc(sdMap)
bins=256
subplot(1,2,2)
h=histogram(sdMap(nz),bins); % no zero padding

M=mean(sdMap(nz));      % mean
Med=median(sdMap(nz));  % median

[id xm]= max(h.Values);        % max histogram
SDh=(h.BinEdges(xm)+h.BinEdges(xm-1))/2;

data = [M Med SDh]

f = figure;
uit = uitable(f,'Data',data,'Position',[20 120 462 100]);
uit.ColumnName = {'SD Mean','SD Median','Max Hist'};
%% 


%% 
wOpixel = find(I > 100); % identify zero padding and BK pixels


sdMap=stdfilt(I,ones(9));
figure('NumberTitle', 'off', 'Name', 'SD map analysis');
subplot(1,2,1)
colormap gray
imagesc(sdMap)
bins=128
subplot(1,2,2)
h=histogram(sdMap(wOpixel),bins); % no zero padding

M=mean(sdMap(wOpixel));      % mean
Med=median(sdMap(wOpixel));  % median

[id xm]= max(h.Values);        % max histogram
SDh=(h.BinEdges(xm)+h.BinEdges(xm-1))/2;

data = [M Med SDh]

f = figure;
uit = uitable(f,'Data',data,'Position',[20 120 462 100]);
uit.ColumnName = {'SD Mean','SD Median','Max Hist'};
%% 




