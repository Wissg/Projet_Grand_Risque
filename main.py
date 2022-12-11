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
import copulae
from statsmodels.distributions.copula.api import (
    CopulaDistribution, GumbelCopula, IndependenceCopula, ClaytonCopula,FrankCopula,StudentTCopula,GaussianCopula)

data = pd.read_excel("data\Base X Y.xlsx")
T = data.iloc[:, 0]
X = data.iloc[:, 1]
Y = data.iloc[:, 2]
data1 = data[["date", "X"]]
data1 = data1.set_index("date")
data3 = data[["X", "Y"]]


def BIC(x, x_theo, parmater):
    n = len(x)
    residual = np.subtract(x_theo, x)
    SSE = np.sum(np.power(residual, 2))
    return n * np.log(SSE / n) + parmater * np.log(n)


def AIC(x, x_theo, parmater):
    n = len(x)
    residual = np.subtract(x_theo, x)
    rss = np.sum(np.power(residual, 2))
    return n * np.log(rss / n) + 2 * parmater


def Kolmogorov_Smirnov(x, x_theo):
    return scipy.stats.ks_2samp(x, x_theo)[1]


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


def clayton(theta, u, v):
    a = np.maximum(np.power(u, -theta) + np.power(v, -theta) - 1, 0)
    a = np.power(a, -1 / theta)
    return a


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

    cov = np.cov(np.stack((X, Y), axis=0))
    ecart_type_X = np.std(X)
    ecart_type_Y = np.std(Y)
    Moyenne_X = np.mean(X)
    Moyenne_Y = np.mean(Y)
    Corr = cov[0, 1] / (ecart_type_X * ecart_type_Y)

    print("Cov(X,Y) = ", cov)
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
    plt.savefig("Graph/Copule_empirique.png")
    plt.show()

    sns.jointplot(x=df['X'], y=df['Y'], kind='kde')
    plt.show()

    # Fit a gaussian copula to the data
    _, ndim = data3.shape
    copulaGaussianCopula = copulae.GaussianCopula(dim=ndim)
    paramGaussianCopula = copulaGaussianCopula.fit(data3)

    copulaFrankCopula = copulae.FrankCopula(dim=ndim)
    paramFrankCopula = copulaFrankCopula.fit(data3)

    copulaClaytonCopula = copulae.ClaytonCopula(dim=ndim)
    paramClaytonCopula = copulaClaytonCopula.fit(data3)

    copulaStudentCopula = copulae.StudentCopula(dim=ndim)
    paramStudentCopula = copulaStudentCopula.fit(data3)

    copulaGumbelCopula = copulae.GumbelCopula(dim=ndim)
    paramGumbelCopula = copulaGumbelCopula.fit(data3)

    theta_Frank = paramFrankCopula.params
    theta_Gumbel = paramGumbelCopula.params
    theta_Clayton = paramClaytonCopula.params
    Degree_Freedom_student = paramStudentCopula.params[0]
    corr_student = paramStudentCopula.params[1][0]
    corr_matrix = paramGaussianCopula.params[0]

    print(" theta_Frank =", theta_Frank)
    print(" theta_Gumbel =", theta_Gumbel)
    print(" Degree_Freedom =", corr_student)
    print(" theta_Clayton =", theta_Clayton)
    print(" corr_matrix =", corr_matrix)

    means = [0.500000, 0.500000]
    a = 0.288434 ** 2
    b = 0.288434 * 0.288434 * corr_matrix
    cov_matrix = [[a, b], [b, a]]
    print(cov_matrix)
    copula = GumbelCopula(theta=theta_Gumbel)
    copula.plot_scatter()
    plt.title("Gumbel Copula "+str(theta_Gumbel))
    aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("AIC = ", aic)
    bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("BIC = ", bic)
    ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
    print("Kolmogorov_Smirnov = ", ks)
    plt.annotate("AIC = " + str(aic), xy=(0, 1))
    plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
    plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
    plt.savefig("Graph/Gumbel_Copula.png")
    plt.show()

    copula = ClaytonCopula(theta=theta_Clayton)
    copula.plot_scatter()
    plt.title("Clayton Copula " + str(theta_Clayton))
    aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("AIC = ", aic)
    bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("BIC = ", bic)
    ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
    print("Kolmogorov_Smirnov = ", ks)
    plt.annotate("AIC = " + str(aic), xy=(0, 1))
    plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
    plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
    plt.savefig("Graph/Clayton_Copula.png")
    plt.show()

    copula = FrankCopula(theta=theta_Frank)
    copula.plot_scatter()
    plt.title("Frank Copula " + str(theta_Frank))
    aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("AIC = ", aic)
    bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("BIC = ", bic)
    ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
    print("Kolmogorov_Smirnov = ", ks)
    plt.annotate("AIC = " + str(aic), xy=(0, 1))
    plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
    plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
    plt.savefig("Graph/Frank_Copula.png")
    plt.show()

    copula = StudentTCopula(df=Degree_Freedom_student , corr=corr_student)
    copula.plot_scatter()
    plt.title("Student Copula " + str(corr_student) +" df = "+str(Degree_Freedom_student))
    # print(np.array(copula.rvs(nobs=598, random_state=None)).shape)
    aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("AIC = ", aic)
    bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("BIC = ", bic)
    ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
    print("Kolmogorov_Smirnov = ", ks)
    plt.annotate("AIC = "+str(aic),xy=(0,1))
    plt.annotate("BIC = "+str(bic),xy=(0,0.95))
    plt.annotate("Ks = "+str(ks),xy=(0,0.9))
    plt.savefig("Graph/Student_Copula.png")
    plt.show()

    copula = GaussianCopula(corr=corr_matrix)
    copula.plot_scatter()
    plt.title("Gaussian Copula " + str(corr_matrix))
    aic = AIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("AIC = ", aic)
    bic = BIC(np.array(df.values), np.array(copula.rvs(nobs=598, random_state=None)), 2)
    print("BIC = ", bic)
    ks = Kolmogorov_Smirnov(np.array(df.values)[:,0], np.array(copula.rvs(nobs=598, random_state=None))[:,0])
    print("Kolmogorov_Smirnov = ", ks)
    plt.annotate("AIC = " + str(aic), xy=(0, 1))
    plt.annotate("BIC = " + str(bic), xy=(0, 0.95))
    plt.annotate("Ks = " + str(ks), xy=(0, 0.9))
    plt.savefig("Graph/Gaussian_Copula.png")
    plt.show()

    mvn_dist = scipy.stats.multivariate_normal(mean=means, cov=cov_matrix)
    mvn_rvs = pd.DataFrame(mvn_dist.rvs(598), columns=["Margin 1", "Margin 2"])

    sns.jointplot(x="Margin 1", y="Margin 2", data=mvn_rvs, kind='kde')
    plt.show()

    # for i in range(int(max(X))):
    #     e[i] = Mean_Excess_Function(X, u[i])
    #     print(u[i])
    # plt.plot(u, e)
    # plt.ylabel("e(u)")
    # plt.xlabel("u")
    # plt.title("Mean Excess Function")
    # plt.show()

    #  La fonction des excès moyen est linéaire décroissante à partir d’un certain seuil ?
    # ✓ On aura ξ < 0 donc cela correspondra au domaine d’attraction de Weibull
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
