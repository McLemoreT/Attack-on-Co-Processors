clf
clc

% Random Curve
syms x
curve1 = 2*x^3 - 3*x^2 + 0.4*x;
ezcurve = x^2;
curve_memristor = ezcurve
curve_software = x^3 - 3;

% ===== Old plotting stuff =====
%hold on
%fplot(curve_memristor)
%fplot(curve_software)
%fplot(ezcurve)
xmin = -10; xmax = 10; ymin = -10; ymax = 10;
%axis([xmin xmax ymin ymax])
%grid on

% https://www.mathworks.com/help/matlab/creating_guis/create-and-run-a-simple-programmatic-app.html
% https://www.mathworks.com/help/matlab/ref/uigridlayout.html 
% Create figure window
fig = uifigure;
fig.Name = "Awesome Algorithm";

% Manage app layout
gl = uigridlayout(fig,[2 2]); % grid layout manager. 2x2 grid
gl.RowHeight = {30,'1x'}; % top row is 30px tall, second row fits rest of window
gl.ColumnWidth = {'fit','1x'}; % width of column set to the content it holds, then fit rest of window

% Create UI components
lbl = uilabel(gl);
lbl.Text = 'This is an awesome application';
ax = uiaxes(gl); % graph
    ax.XLim = [xmin xmax];
    ax.YLim = [ymin ymax];
    ax.XGrid = 'on';
    ax.YGrid = 'on';
    hold(ax, 'on');

% UI Components - Button Menu
g2 = uigridlayout(gl, [3 1]); % 2 rows, 1 column (vertical)
g2.RowHeight = {44, 44}; % 44 pixels height per button
%g2.ColumnWidth = 22;
gotoMem = uibutton(g2, 'Text', 'Go to Mem');
gotoSoft = uibutton(g2, 'Text', 'Go to Soft');
newImage = uibutton(g2, 'Text', 'Test new image', 'ButtonPushedFcn', @(event) newImg(xmin, xmax, ymin, ymax, ax));

% Position label 
lbl.Layout.Row = 1;
lbl.Layout.Column = [1 2];
% Position axes
ax.Layout.Row = 2;
ax.Layout.Column = 2; % Spans both columns

% Plotting
fplot(ax, curve_memristor);
[randX, randY] = newImg(xmin, xmax, ymin, ymax, ax);
%legend('show', 'Location', 'best')

[POI, vec] = findClosestDirection(randX, randY, ezcurve);
plot(ax, POI(1), POI(2), 'o', 'Color', 'black'); % Plot the point on the figure

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

function [randX, randY] = newImg(xmin, xmax, ymin, ymax, ax)
    randX = randi([xmin xmax])
    randY = randi([ymin ymax])
    %randX = 1;
    %randY = 0;
    plot(ax, randX, randY, 'x', 'Color', 'black'); % Plot the random point
end