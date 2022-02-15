clf
syms x
curve_memristor = 2*x^3 - 3*x^2 + 0.4*x;
curve_software = x^3 - 3;
ezcurve = x^2;

hold on
%fplot(curve_memristor)
%fplot(curve_software)
fplot(ezcurve)
xmin = -10; xmax = 10; ymin = -10; ymax = 10;
axis([xmin xmax ymin ymax])
grid on

%X = [0,0;2,1];
%d = pdist(X,'euclidean')

x = randi([xmin xmax])
y = randi([ymin ymax])
%x = 1;
%y = 0;

plot(x, y, 'x', 'Color', 'black')

[POI, vec] = findClosestDirection(x, y, ezcurve);
plot(POI(1), POI(2), 'o');


function [point, vector] = findClosestDirection(initX, initY, curve)
    syms x y
    dfx = sqrt((x - initX)^2 + (curve - initY)^2)
    D = diff(dfx)
    simplify(D)
    eqn = D == 0
    dist = vpa(solve(eqn, x))
    dist(dist~=real(dist)) = NaN
    x = dist(1)
    y = eval(curve)

% look for 2nd derivative to find shortest vector, AKA min and max (just
% find min)

    point = [x, y];
    vector = [initX - x, initY - y];
end