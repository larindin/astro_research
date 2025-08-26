%
%  gr-metapost-tally.mp    2022-01-14    Mark Senn    https://bit.ly/marksenn
%
%  Make 20 .pdf files for the tally mark representations for 1 to 20.
%
%  Whenever I use files with numbers in them I like to put leading zeros
%  in the names so they will be listed in order in the directory.
%
%  Do these commands:
%      mpost gr-metapost-tally
%      epspdf gr-metapost-tally-1.mps;   mv gr-metapost-tally-1.pdf  gr-metapost-tally-01.pdf
%      epspdf gr-metapost-tally-2.mps;   mv gr-metapost-tally-2.pdf  gr-metapost-tally-02.pdf
%      ...
%      epspdf gr-metapost-tally-9.mps;   mv gr-metapost-tally-9.pdf  gr-metapost-tally-09.pdf
%      epspdf gr-metapost-tally-20.mps;  mv gr-metapost-tally-20.pdf gr-metapost-tally-20.pdf
%
%  If you use bash you can do
%      for i in $(seq 1 20); \
%          do epspdf gr-tally-$i.mps ; printf -v var "%02d" "$i"; mv gr-tally-$i.pdf gr-tally-$var.pdf; \
%      done
%  to do all the lines that start with epspdf.
%

prologues      := 3;
outputtemplate := "gr-%j-%c.mps";
outputformat   := "eps";

% Define dimensions.
adjust = 0.15cm;  % x and y adjustment
dx     = 0.25cm;  % delta x
h      = 1cm;     % tally height
s      = 1cm;     % spacing between groups of tally marks

% Define number of tally marks.
n = 20;  % number of marks

% Compute the tally mark paths.
path p[];
x := 0;  xstart := x;
for i = 1 upto n:
    if (i mod 5 = 1):
        p[i] = ((x,0)--(x,h));  x := x + dx;
    fi;
    if (i mod 5 = 2):
        p[i] = ((x,0)--(x,h));  x := x + dx;
    fi;
    if (i mod 5 = 3):
        p[i] = ((x,0)--(x,h));  x := x + dx;
    fi;
    if (i mod 5 = 4):
        p[i] = ((x,0)--(x,h));
    fi;
    if (i mod 5 = 0):
        p[i] = ((xstart-adjust,adjust)--(x+adjust,h-adjust));
        x := x + dx + adjust + s;  xstart := x;
    fi;
endfor;
show p[1];

%
%  Write the tally-*.mps files
%

for i = 1 upto n:

    beginfig(i);

        % Define pen---this must go between beginfig and endfig.
        pen mypen;
        mypen = pencircle scaled 3.5;
        pickup mypen;

        % Draw gray tally marks.
        for j = i+1 upto n:
            draw p[j] withcolor 0.8white;
        endfor;

        % Draw black tally marks.
        % Do this last so the dark tally marks
        % will be on top of the light tally marks.
        for j = 1 upto i:
            draw p[j] withcolor black;
        endfor;
  
    endfig;

endfor;

end.
