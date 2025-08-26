#!/bin/python2

#  gr-python2.py    2023-08-31    Mark Senn <mark@purdue.edu>

# On Linux, I typed
#     ./gr-python2.py
# to run this program.

import matplotlib as mpl

# From https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
#     You can use TeX to render all of your Matplotlib text by setting
#     rcParams["text.usetex"] (default: False) to True. This requires
#     that you have TeX and the other dependencies described in the
#     Text rendering With LaTeX tutorial properly installed on your
#     system. Matplotlib caches processed TeX expressions, so that
#     only the first occurrence of an expression triggers a TeX
#     compilation. Later occurrences reuse the rendered image from the
#     cache and are thus faster.
mpl.rcParams['text.usetex'] = True

# From
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
#     # Force matplotlib to not use any Xwindows backend.
#     matplotlib.use('Agg')
mpl.use('Agg')

import matplotlib.pyplot as plt

plt.figure(figsize=(4,4))

import numpy as np

x = [ ]
y = [ ]
for i in range(0,1001):
    x.append(i / 100.0)
    y.append((i/100.0) ** 2)

# Normally, there is extra space to the left, right, top, and bottom
# of plotted data in a graph.  This gets rid of that space.
plt.autoscale(enable=None, tight=True)

plt.plot(x, y, 'g')

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.savefig('../graphics/gr-python2.pdf')
