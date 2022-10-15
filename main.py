import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, genextreme
import scipy
import math
from mpl_toolkits import mplot3d
from datetime import datetime

data = pd.read_excel("data\Base X Y.xlsx")
T = data.iloc[:, 0]
X = data.iloc[:, 1]
Y = data.iloc[:, 2]
data1 = data[["date", "X"]]
data1 = data1.set_index("date")


print(data1)

if __name__ == '__main__':
    fig = plt.figure()
    plt.bar(T, X, 100)
    plt.show()
    bins = 100

    plt.figure()
    plt.hist(X, bins=bins)
    plt.title("")
    plt.show()

    # Milieu de chaque classe
    y, x = np.histogram(X, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0




    # Modélisation loi normal
    param = norm.fit(X)

    loc = param[-2]
    scale = param[-1]
    arg = param[:-2]

    pdf = norm.pdf(x, loc=loc, scale=scale, *arg)

    plt.figure()
    plt.plot(x, pdf, label="loi normal", linewidth=3)
    plt.plot(x, y, label="Data")
    plt.legend()
    plt.show()
    # Modelisation GEV

    # Calcule maxima année.
    year_maxima = data1.resample("Y").max()
    print(year_maxima)
    # Ajuster la distribution GEV aux maxima
    p = genextreme.fit(year_maxima)
    loc = p[-2]
    scale = p[-1]
    c = p[:-2]
    print(p)
    genextreme.pdf(x, *c, loc=loc, scale=scale)
    plt.figure()
    plt.plot(x, pdf, label="loi GEV", linewidth=3)
    plt.plot(x, y, label="Data")
    plt.legend()
    plt.show()


    # Calculer la VaR 99,5% (nécessaire pour le calcul de la CVaR)
    VaR_99 = genextreme.ppf(0.995, *p)

    print(VaR_99)

    # Calculer l'estimation du CVaR à 99 %.
    CVaR_99 = (1 / (1 - 0.995)) * genextreme.expect(lambda x: x,
                                                   args=(p[0],), loc=p[1], scale=p[2], lb=VaR_99)
    print(CVaR_99)