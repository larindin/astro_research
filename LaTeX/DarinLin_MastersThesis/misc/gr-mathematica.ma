(*
    MaTeX is a Wolfram Language package to use LaTeX fonts in
    Wolfram Language graphics.  See
        http://szhorvat.net/pelican/latex-typesetting-in-mathematica.html
    for instructions on how to use MaTeX.
*)	

(* Install MaTeX. *)
ResourceFunction["MaTeXInstall"][];

(* Load MaTeX. *)
<<MaTeX`;

latexStyle = {FontFamily->"Latin Modern Roman", FontSize->10};

g = Graphics[
  Plot[x^2, 
    {x,0,10},
    BaseStyle->latexStyle,
    PlotRange->{{0,10},{0,100}}
  ]
];
I
Export["../graphics/gr-mathematica.pdf", g];
