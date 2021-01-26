#todo take this from https://github.com/NathanKr/machine-learning-plots-lib/blob/main/utils.py
import matplotlib.pyplot as plt


def linear_line(x,intercept,slope):
    return slope * x + intercept


def plot_with_residual(target_plt,X,Y,intercept,slope,color='g',linewidth=3):
    h = linear_line(X,intercept,slope)
    target_plt.plot(X,Y,'o',X,h)
    m = X.size
    i = 0
    while i < m:
        target_plt.plot([X[i],X[i]],[h[i],Y[i]],color=color,linewidth=linewidth)
        i += 1