%{
%x = 1:300;
%            curve1 = log(x)
syms x
            curve1 = matlabFunction(x^5 + x^3 - 6)
%            curve2 = 2*log(x)
            curve2 = matlabFunction(0.5*x^3 -2*x^2 + 3*x - 3)
%            plot(x, curve1, 'r', 'LineWidth', 2);
%            hold on;
%            plot(x, curve2, 'b', 'LineWidth', 2);
            x11 = num2cell(curve1)
            x22 = num2cell(curve2)
            x2 = [x, fliplr(x)];
            inBetween = [curve1, fliplr(curve2)];
            fill(app.UIAxes, x2, inBetween, 'g');
%}
%{
syms x
fun = x^5 + x^3 - 6;
f{1} = 'x^2'; % declare as cell array {} of string ''
f{1} = fun

    h3 = ezplot(f{1},[-10,10]);   %the correct way to call ezplot
    x = get(h3, 'XData');          %get the x and y data
    y = get(h3, 'YData');
    area(x,y,'FaceColor',[.7 0 0]);   %plot the (x,y) area in red
%{
figure('Color', 'w');
for ii = 1:4                          %do not use i or j in Matlab
    subplot(2,2,ii);
    h(ii) = ezplot(f{ii},[0,6000]);   %the correct way to call ezplot
    x = get(h(ii), 'XData');          %get the x and y data
    y = get(h(ii), 'YData');
    area(x,y,'FaceColor',[.7 0 0]);   %plot the (x,y) area in red
end
%}
%}
syms x
app.curve_memristor =  x^5 + x^3 - 6;
            f{1} = app.curve_memristor; % declare as cell array {} of string ''
            h3 = ezplot(f{1},[-10,10]);   %the correct way to call ezplot
            x = get(h3, 'XData');          %get the x and y data
            y = get(h3, 'YData');
            area(x,y,'FaceColor',[.7 0 0]);   %plot the (x,y) area in red