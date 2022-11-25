import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, genextreme
import scipy
import seaborn as sns
import math
from mpl_toolkits import mplot3d
from datetime import datetime
from copulae import EmpiricalCopula, pseudo_obs

data = pd.read_excel("data\Base X Y.xlsx")
T = data.iloc[:, 0]
X = data.iloc[:, 1]
Y = data.iloc[:, 2]
data1 = data[["date", "X"]]
data1 = data1.set_index("date")
data3 = data[["X", "Y"]]

def BIC(x,x_theo,parmater):
    n = len(x)
    residual = np.subtract(x_theo, x)
    SSE = np.sum(np.power(residual, 2))
    return n*np.log(SSE/n) + parmater*np.log(n)

def AIC(x,x_theo,parmater):
    n = len(x)
    residual = np.subtract(x_theo, x)
    rss = np.sum(np.power(residual, 2))
    return n * np.log(rss / n) + 2 * parmater

def Mean_Excess_Function(x, u):
    e = 0
    a = 0
    b = 0
    for i in range(len(x)):
        a = np.max(x[i] - u, 0) + a
        if x[i] > u:
            b = b + 1
    e = a / b
    return e


print(data1)

if __name__ == '__main__':
    fig = plt.figure()
    plt.bar(T, X, 100, label="pertes réalisés sur le portefeuille X ")
    plt.xlabel("date")
    plt.ylabel("X")
    plt.show()
    bins = 100

    plt.figure()
    plt.hist(X, bins=bins)
    plt.title("")
    plt.show()

    # Milieu de chaque classe
    y, x = np.histogram(X, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    cov = np.stack((X, Y), axis=0)
    ecart_type_X = np.std(X)
    ecart_type_Y = np.std(Y)
    Moyenne_X = np.mean(X)
    Moyenne_Y = np.mean(Y)
    Corr = np.cov(cov)[0, 1] / (ecart_type_X * ecart_type_Y)

    print("Cov(X,Y) = ", np.cov(cov))
    print("var(x) = ", ecart_type_X ** 2, "var(y) = ", ecart_type_Y ** 2)
    print("E(x) = ", Moyenne_X, "E(y) = ", Moyenne_Y)
    print("Correlation = ", Corr)

    e = np.zeros(int(max(X)))
    u = np.linspace(0.001, int(max(X)), int(max(X)))

    u = pseudo_obs(data3)
    emp_cop = EmpiricalCopula(u, smoothing="beta")
    df = emp_cop.data
    plt.scatter(df['X'], df['Y'])
    plt.xlabel("Rangs de X")
    plt.ylabel("Rangs de Y")
    plt.title("Copule de X et Y, Rangs de X en fonction des rangs de Y")
    plt.legend()
    plt.show()


    # for i in range(int(max(X))):
    #     e[i] = Mean_Excess_Function(X, u[i])
    #     print(u[i])
    # plt.plot(u, e)
    # plt.ylabel("e(u)")
    # plt.xlabel("u")
    # plt.title("Mean Excess Function")
    # plt.show()

    print(scipy.stats.pearsonr(X, Y)[0])  # Pearson's r correlation de base déja calculé

    print(scipy.stats.spearmanr(X, Y)[0])  # Spearman's rho

    print(scipy.stats.kendalltau(X, Y)[0])  # Kendall's tau

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
    # print(year_maxima)
    # Ajuster la distribution GEV aux maxima
    p = genextreme.fit(year_maxima)
    loc = p[-2]
    scale = p[-1]
    c = p[:-2]
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
