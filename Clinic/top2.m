%{
syms x
S = vpasolve(2*x^4 + 3*x^3 - 4*x^2 - 3*x + 2 == 0, x)

curve_memristor = x^5 + x^3 - 6;
curve_software = 0.5*x^3 -2*x^2 + 3*x - 3;
A = vpasolve(curve_memristor == curve_software, x);
A(A~=real(A)) = NaN
%}

%{
x = 0:0.1:10
eq = sin(3*x)
plot(eq)

figure
fs = 300
y = fft(eq)
n = length(eq);          % number of samples
f = (0:n-1)*(fs/n);     % frequency range
power = abs(y).^2/n;    % power of the DFT
plot(f, power)
%}

x = -2:0.25:2;
z1 = x.^exp(-x.^2);
z2 = 2*x.^exp(-x.^2);

real_z1 = real(z1);
imag_z1 = imag(z1);

real_z2 = real(z2);
imag_z2 = imag(z2);

plot(real_z1,imag_z1,'g*',real_z2,imag_z2,'bo')

