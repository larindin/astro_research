%  gr-metapost.mp  2021-11-16  Mark Senn  https://bit.ly/marksenn
%
%  Thanks to Yanghyun Kim for asking how to do this figure.
%

prologues := 3;  % make .eps files with embedded fonts

outputtemplate := "%j-%c.mps"; % make output files gr-metapost-1.mps and gr-metapost-2.mps

beginfig(1);
    pickup pencircle scaled 1pt;  % line width
  
    u := 0.5cm;  % unit size

    deltax := 5u;
    deltay := 5u;
    xoff   := 2u;

    pair ll, lr, ul, ur;
    ll := (0, 0);                 % lower left point
    lr := ll + (deltax, 0);       % lower right point
    ul := ll + (xoff,   deltay);  % upper left point
    ur := ul + (deltax, 0);       % upper right point

    draw ll--lr--ur--ul--ll;
    draw ll--ur;

    dotlabel.top(btex 1 etex, ul);  % top label
    dotlabel.top(btex 2 etex, ur);  % top label
    dotlabel.bot(btex 3 etex, ll);  % bottom label
    dotlabel.bot(btex 4 etex, lr);  % bottom label
endfig;

beginfig(2);
    pickup pencircle scaled 1pt;  % line width

    u := 2cm;  % unit size
  
    deltax := 5u;
    deltay := 1u;
    xoff   := 0.5u;

    pair ll, lr, ul, ur;
    ll := (0, 0);                  % lower left point
    ur := ll + (deltax, 0);        % upper right point
    ul := ll + (xoff,   deltay);   % upper left point
    lr := ur + (-xoff,  -deltay);  % lower right point
  
    dotlabel.llft(btex $p_i$ etex,        ll);          % lower left label
    label.lft(    btex $\Delta p_i$ etex, 1/2[ll,ul]);  % left label
    dotlabel.top( btex ${\hat p}_i$ etex, ul);          % upper left label

    dotlabel.lrt(btex ${\hat p}_j$ etex, lr);           % lower rigt label
    label.rt(    btex $\Delta p_j$ etex, 1/2[lr,ur]);   % right label
    dotlabel.urt(btex $p_j$ etex,        ur);           % upper right label
  
    drawarrow ll--ul;
    drawarrow ur--lr;
    draw ul--lr dashed evenly;
    draw ll--3/8[ll,ur];
    draw 3/8[ll,ur]{up}
        for  i = 0 upto 360:
	    ..3/8[ll,ur] + ((i/360)*(deltax/4), (u/20)*sind(12i))
        endfor;
    draw 5/8[ll,ur]--ur;
endfig;

end.
