import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize

df = pd.read_csv(r'A1.csv')
df = df.dropna(how='all')

maturity_time = df["Time_in_month_untiil_MD"] / 12

# Calculate YTM for each bond
ytm_by_time = {}
ytm_by_bond = {}


def ytm_calculation(price, coupon, times):
    bond_tm = maturity_time[:times]
    ytm = lambda y: sum(
        [coupon * 100 * 0.5 * np.exp(-y * t) for t in bond_tm]) + (
                            (100 + coupon * 100 * 0.5) * np.exp(
                        -y * times)) - price
    return optimize.newton(ytm, 0.005)


for i in range(10):
    ytm_bond = []
    price_of_bond = list(df.iloc[i, 7:])
    times = int(df.iloc[i, 6]) + 1
    coupon = df.iloc[i, 2]

    for price in price_of_bond:
        ytm_bond.append(ytm_calculation(price, coupon, times))

    ytm_by_time[i] = ytm_bond.copy()

for i in range(10):
    ytm_by_bond[i] = []
    for bond in ytm_by_time:
        ytm_by_bond[i].append(ytm_by_time[bond][i])

time_axis = (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)
dates = ("Jan 16", "Jan 17", "Jan 18", "Jan 19", "Jan 20",
         "Jan 23", "Jan 24", "Jan 25", "Jan 26", "Jan 27")

# Plot yield curve
plt.figure()
for i in range(10):
    plt.plot(time_axis, ytm_by_bond[i], label=dates[i])

plt.title("Five year yield Curve")
plt.xlabel("Time to Maturity")
plt.ylabel("Yield to Maturity")
plt.legend()
plt.show()


# Calculate spot rate for each bond


def zero_coupon_rate(price, times):
    return -np.log(price / 100) / times


def bootstrapping(price, coupon, T, spot_rate):
    if T >= 0.5:
        cummulate = price
        deduction = 0
        for i in range(len(spot_rate)):
            spot_rate_t = spot_rate[i]
            deduction += coupon * 100 * 0.5 * np.exp(
                -spot_rate_t * maturity_time[i])
        cummulate -= deduction
        spot_rate.append(
            -(np.log(cummulate / (100 + coupon * 100 * 0.5)) / T))
    else:
        spot_rate.append(
            zero_coupon_rate(price, T))
    return spot_rate


spots_date = {}
time_date = list(df.columns[-10:])
coupon = list(df['Coupon'])

n = 0
for date in time_date:
    daily_price = list(df[date])
    spot_rate = []

    for i in range(len(daily_price)):
        spot_rate = bootstrapping(daily_price[i],
                                  coupon[i],
                                  maturity_time[i],
                                  spot_rate).copy()
    spots_date[n] = spot_rate
    n += 1

# Plot spot rate curve
plt.figure()
for i in range(10):
    plt.plot(time_axis, spots_date[i], label=dates[i])

plt.title("Five year spot rate Curve")
plt.xlabel("Time to Maturity(year)")
plt.ylabel("Spot rate")
plt.legend()
plt.show()

# Calculate forward rate
forward_rate_date = {}
starting_time = 1
interval = [2, 3, 4, 5]
n = 0

for date in time_date:
    forward_rate = np.zeros(len(interval))
    spot_curve = spots_date[n]
    for i in range(len(interval)):
        forward_rate[i] = (spot_curve[interval[i] * 2 - 1] * interval[i] -
                           spot_curve[starting_time]) / (
                                      interval[i] - starting_time)
    forward_rate_date[n] = forward_rate
    n += 1

# Plot forward rate curve
# Plot forward rate
plt.figure()
year_interval = ['1yr-1yr', '1yr-2yr', '1yr-3yr', '1yr-4yr']

for i in range(10):
    plt.plot(year_interval, forward_rate_date[i], label=dates[i])

plt.title("Five year forward rate curve")
plt.xlabel("Forward time")
plt.ylabel("Forward rate")
plt.legend()
plt.show()

# Covariance matrices
ytm_matrix = np.empty([5, 9])

for i in range(5):
    for j in range(9):
        ytm_matrix[i, j] = np.log(
            ytm_by_bond[j + 1][2 * i + 1] / ytm_by_bond[j][
                2 * i + 1])

forward_matrix = np.empty([4, 9])
for i in range(4):
    for j in range(9):
        forward_matrix[i,j] = np.log(forward_rate_date[j+1][i]/forward_rate_date[j][i])

ytm_matrix_adjusted = (ytm_matrix.T - np.mean(ytm_matrix.T, axis = 0)).T
forward_matrix_adjusted = (forward_matrix.T - np.mean(forward_matrix.T, axis = 0)).T

ytm_cov = np.cov(ytm_matrix_adjusted)
forward_cov = np.cov(forward_matrix_adjusted)

ytm_eigenvalue, ytm_eigenvector = LA.eig(ytm_cov)
print(ytm_eigenvalue)
print(ytm_eigenvector)

forward_eigenvalue, forward_eigenvector = LA.eig(forward_cov)
print(forward_eigenvalue)
print(forward_eigenvector)

sum1 = 0
for i in ytm_eigenvalue:
    sum1 += i
ytm_eigenvalue[0]/sum1

sum2 = 0
for i in forward_eigenvalue:
    sum2 += i
forward_eigenvalue[0]/sum2

s
