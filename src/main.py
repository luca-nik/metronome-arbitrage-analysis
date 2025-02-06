import math
import numpy as np

from constants import *
from helpers import prettify_time

from functions import *
from plotter import *

# Label to simulate the arbitrage strategy
simulate_strategy_label = True
# Compute price function of DPA
t = np.linspace(0, T, T)
p_dpa = P_DPA(t)
# Compute price function of ACC
m = np.arange(0, MET_LIMIT, 0.0001)
p_acc = P_ACC(m)
#Plot
plot_prices(t, p_dpa, m, p_acc)
#
# Find the time t*, the associated profit and MET to buy
t_star, met_max = find_t_star(t)
print(f"The value of t* is: {prettify_time(t_star, include_milliseconds=True)} h with m*: {met_max}")
M = np.arange(met_max,MET_LIMIT)
profit = get_profit_value(M, t_star)
if max(profit) > FEE:
    m_d = np.argmax(profit)
    max_profit = max(profit)
    print(f"The value of the profit at this time is {max(profit)/GWEI} GWEI for {m_d} MET")
else:
    print(f"Trading at this exact time we have no profit")
    
# Get for all subsequent times the maximum value of the profit and plot it
t = np.linspace(math.ceil(t_star), T, 1000)
met_max_list = compute_met_limit(t)
max_profit_values = []
#
for met_max, t_lim in zip(met_max_list,t):
    M = np.linspace(0,met_max, 1000)
    profit = get_profit_function(M, t_lim)

    if max(profit) > FEE:
        m_d = np.argmax(profit)
        max_profit_values.append(max(profit))

plot_max_profit(t, max_profit_values)

# Plot the profit function on specific times
t_plot = np.linspace(t_star + 1, 5*t_star + 1, 5)
plot_profits(t_plot)

# Simulate the arbitrage startegy
if simulate_strategy_label is True:
    time_step = 0.5  # Set the desired time step
    results = simulate_strategy(time_step)
    plot_simulation_results(results)