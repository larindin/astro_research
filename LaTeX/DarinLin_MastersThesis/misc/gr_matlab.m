%  gr_matlab.m    2021-11-11    Mark Senn <mark@purdue.edu>

% On Linux, I typed
%     lm matlab
%     matlab -nodesktop -nosplash -r gr_matlab
% to run this program.

% See
%     https://www.mathworks.com/matlabcentral/answers/346436-how-to-use-latex-interpreter-for-xticklabels
% Do axis labels using LaTeX.
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
% Do legends using LaTeX.
set(groot, 'defaultLegendInterpreter', 'latex');
% Do text using LaTeX.
set(groot, 'defaulttextinterpreter', 'latex');

x = linspace(0,10,100);
y = x .^ 2;
plot(x, y);
saveas(gcf, '../graphics/gr-matlab.pdf');

% The following line worked for me on Linux.
!pdfcrop ../graphics/gr-matlab.pdf ../graphics/gr-matlab.pdf;

exit;
