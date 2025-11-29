close all
clear all

Tsd=22;
Tm=30;

figure
hold on
for k=1:100
    for i = 1:7
        dim(i)=2.^i;
        image=Tm+Tsd*randn(dim(i));
        M(i)=mean(image(:));
    end
    plot(dim,M)
end
xlabel('ROI Dimension')
ylabel('Mean (imposed 30)')

figure
hold on
for k=1:100
    for i = 1:7
        dim(i)=2.^i;
        image=Tm+Tsd*randn(dim(i));
        sd(i)=std(image(:));
    end
    plot(dim,sd)
end
xlabel('ROI Dimension')
ylabel('SD (imposed 22)')