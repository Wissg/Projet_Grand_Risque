import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
import math
from mpl_toolkits import mplot3d

data = pd.read_excel("data\Base X Y.xlsx")
T = data.iloc[:, 0]
X = data.iloc[:, 1]
Y = data.iloc[:, 2]

if __name__ == '__main__':
    bins = 100
    plt.figure()
    plt.hist(X, bins=bins)
    plt.title("")
    plt.show()

    # Milieu de chaque classe
    y, x = np.histogram(X, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    dist_name = "norm"

    # Paramètres de la loi
    dist = getattr(scipy.stats, dist_name)

    # Modéliser la loi
    param = dist.fit(X)

    loc = param[-2]
    scale = param[-1]
    arg = param[:-2]

    pdf = dist.pdf(x, loc=loc, scale=scale, *arg)

    plt.figure(figsize=(12, 8))
    plt.plot(x, pdf, label=dist_name, linewidth=3)
    plt.plot(x, y, label= "Data")
    plt.legend()
    plt.show()