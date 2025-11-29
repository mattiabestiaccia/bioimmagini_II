%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D region growing
% Vincenzo Positano HIPPO SW 2021
% adapted from https://it.mathworks.com/matlabcentral/fileexchange/35269-simple-single-seeded-region-growing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Igray: input 3D image
%%% tolerance: tolerance value (gray levels)
%%% seed : seed coordinates [x, y, z ]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Phi = regionGrowing3D(Igray,tolerance,seed)
Phi = false(size(Igray)); % mask of segmented pixels
ref = true(size(Igray));  
PhiOld = Phi;             % mask at the previous step
Phi(seed(1),seed(2),seed(3)) = 1;  % initialize mask with seed value  
while(sum(Phi(:)) ~= sum(PhiOld(:))) % cycle until no pixel is added to the mask
    PhiOld = Phi;                    % store current mask
    segm_val = Igray(Phi);           % voxels currently segmented
    meanSeg = mean(segm_val);        % update mean value of the segmented region 
    posVoisinsPhi = imdilate(Phi,strel('cube',3)) - Phi; % neighbour voxels (26)
    %posVoisinsPhi = imdilate(Phi,strel('sphere',1)) - Phi; % neighbour voxels (6)
    voisins = find(posVoisinsPhi);   % list of neighbour voxels values 
    valeursVoisins = Igray(voisins); 
    Phi(voisins(valeursVoisins > meanSeg - tolerance & valeursVoisins < meanSeg + tolerance)) = 1; %include neighbour voxels basing on tolerance
end
