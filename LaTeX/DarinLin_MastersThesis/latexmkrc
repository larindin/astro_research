#
#  revised  latexmkrc  2024-10-16  Mark Senn  http://engineering.purdue.edu/~mark
#  created  latexmkrc  2024-10-10  Mark Senn  http://engineering.purdue.edu/~mark
#

# This file is only used by Overleaf.
# Overleaf is used to run the TeX typesetting system on the web.?.

# A latekmkrc file can be used to augment what
# the latexmk program does on Overleaf (run LaTeX
# on the web).  See
#   https://www.overleaf.com/learn/latex/Articles/How_to_use_latexmkrc_with_Overleaf

# So \"o will work in text and {\"o} will work in biblatex bibliographies.
# Without this the accented character is not printed in the bibliography.
$biber = 'biber --output-safechars';

# Set the default timezone to Indianapolis, Indiana.
# That's the same timezone as West Lafayette, Indiana---where
# Purdue's main campus is located.
# See
#   https://www.overleaf.com/learn/latex/Articles/How_to_use_latexmkrc_with_Overleaf
# for more information.
$ENV{'TZ'} = 'America/Indiana/Indianapolis';
