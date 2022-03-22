%{
syms x
S = vpasolve(2*x^4 + 3*x^3 - 4*x^2 - 3*x + 2 == 0, x)

curve_memristor = x^5 + x^3 - 6;
curve_software = 0.5*x^3 -2*x^2 + 3*x - 3;
A = vpasolve(curve_memristor == curve_software, x);
A(A~=real(A)) = NaN
%}

result = 'true'
if (result == 'true')
    a = 3
else
    a = 0
end
