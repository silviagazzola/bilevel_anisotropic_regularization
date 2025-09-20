function [m,n,s] = setsize(m,n)
% To avoid circular overlaps, the length of convolution
% has to be reserved on the output array.
% Radix of 2 is the fastest. The length of data is defined as 
% 2^n or its multiple by a few prime factors.
% FFT is fast enough in case of two = 3 or 4.
% if two is small, less primes but larger memory.
% if two is large, more primes but smaller memory.
l = n + m - 1;
two = 4;
if two > 1,
    M = 2.^ceil(log2(l));
    two = 2^two;
    q = (1 + two / 2):two;
    t = q' * M / two;
    for j = 1:size(M,2)
        i = find(t(:,j) >= l(j),1,'first');
        M(j) = t(i,j);
    end
else M = l;
end
k = M - n;
h = floor(k / 2);
n = [h; k - h];
s = floor(m / 2) + h;
m = M - m;
end