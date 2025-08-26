#  gr-r.R    2021-11-24    Mark Senn    https://bit.ly/marksenn

# See
#     https://texample.net/tikz/examples/tikzdevice-demo/
# for information on tikzDevice.

# Load the tikzDevice package.
# If you don't have it, install with
#     install.packages("tikzDevice", repos="http://R-Forge.R-project.org")
require(tikzDevice)

tikz(file = '../gr-r.tex', height=5, width=5)

curve(x^2, from=0, to=10, xlab="$x$", xlim=c(0,10), ylab="$y$", ylim=c(0,100))

dev.off()
