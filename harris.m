clear all;
close all;
A = imread('cast-left.jpg');
I = double(rgb2gray(A));

% Compute the gradient of A
[Ix,Iy]=gradient(I);
[height,width] = size(I);

% Compute the matrix M
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;

% Remove noise
h= fspecial('gaussian',[7 7],2);
x2 = filter2(h, Ix2);
y2 = filter2(h, Iy2);
xy = filter2(h, Ixy);

% Parameter of Harris
k=0.06;
R = zeros(20,20);
% Compute the harris 
for i=1:height
    for j=1:width
        % Compute function R
        M=[x2(i,j) xy(i,j);xy(i,j) y2(i,j)];
        R(i,j)=det(M)-k*(trace(M))^2;
    end
end

% Non-maximum suppression
Rmax = max(max(R));
t=0.01*Rmax;
for i = 1:height
    for j = 1:width
        if R(i,j) < t
            R(i,j) = 0;
        end
    end
end
% Normalization
corner_peaks=imregionalmax(R);

% Get the sparse set of corners
[posr,posc]=find(corner_peaks ==1);

figure
imshow(A);
hold on
for i = 1:length(posr)
    plot(posc(i),posr(i),'r.');
end