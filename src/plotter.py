import numpy as np
from matplotlib import pyplot as plt
from functions import *
from helpers import prettify_time
from constants import *

def plot_prices(t_values, P_DPA_values, m_values, P_ACC_values):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns
    axes[0].plot(t_values, P_DPA_values, color = 'b', linewidth = 2)
    axes[0].set_title(r'${P}^{eth}_{D}(t) = {P}^{eth}_{D,0} \cdot (0.99)^{\frac{t}{60}}$')
    axes[0].set_xlabel('Time [h]')
    axes[0].set_ylabel(r'${P}^{eth}_{D}(t)$')
    num_ticks = 4
    tick_positions = np.linspace(t_values[0], t_values[-1], num_ticks)
    tick_labels = [prettify_time(t_values[np.abs(t_values - i).argmin()]) for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)

    axes[1].plot(m_values, P_ACC_values, color = 'r', linewidth = 2)
    axes[1].set_title(r'${P}^{eth}_{A}(m_d) = \frac{E_{A,0}}{M_{A,0} + 2m_d}$')
    axes[1].set_xlabel(r'$m_d$ [MET]')
    axes[1].set_ylabel(r'${P}^{eth}_{A}(m_d)$')
    # Set the grid
    for ax in axes:
        ax.grid(True)
    
    # Adjust layout for better fitting
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("images/prices.png", dpi=400)

def plot_profits(t):
    plt.figure(figsize=(10, 6))
    met_max = compute_met_limit(t)
    for time, m_max in zip(t[1:], met_max[1:]):
        m = np.linspace(0, m_max, 1000)
        p = get_profit_function(m, time)
        plt.plot(m, p, label=f"t = {prettify_time(time)} h", linestyle='-', marker=None, linewidth = 2)
    plt.xlabel(r'$m_d$ [MET]')
    plt.ylabel(r'$\eta$ [ETH]')
    plt.legend()
    plt.grid(True) 
    plt.tight_layout()
    plt.savefig("images/profits.png", dpi = 400)

def plot_simulation_results(results: list):
    """
    Plots the results of the simulation with two graphs: 
    (i) Time vs profit 
    (ii) Time vs e0 and m0.
    
    Args:
        results (list): The simulation results containing (time, profit, e0, m0, DAP_m0).
    """
    # Extract the data from the results
    times = [result[0] for result in results]
    profits = [(result[1] - FEE)/GWEI for result in results]
    e0_values = [result[2] for result in results]
    m0_values = [result[3] for result in results]
    
    # Create the 1x2 figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Time vs Profit 
    axes[0].plot(times, profits, color='r')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Profit (GWEI)")
    axes[0].set_title("Arbitrage Profit Over Time")
    axes[0].grid(True)
    # Choose 4 evenly spaced ticks for the x-axis in the first plot
    num_ticks = 4
    tick_positions = np.linspace(times[0], times[-1], num_ticks)
    tick_labels = [prettify_time(times[np.abs(times - i).argmin()]) for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
        # Plot 2: Time vs e0 and m0 using twinx
    ax1 = axes[1]
    ax2 = ax1.twinx()

    ax1.plot(times, e0_values, label='ACC ETH Reserve', color='blue')
    ax2.plot(times, m0_values, label='ACC MET Reserve', color='green')
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("ACC ETH Reserve", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylabel("ACC MET Reserve", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    ax1.set_title("ACC Resource Levels Over Time")
    ax1.grid(True)

    # Set the same x-ticks for both plots
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Display the plots
    plt.tight_layout()

    # Save the figute
    plt.savefig("images/strategy.png", dpi=400)


def plot_max_profit(t, max_profit):
    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns
    
    # Plot the original max_profit in the first subplot
    axes[0].plot(t, max_profit, color='r', linestyle='-', marker=None, linewidth = 2)
    axes[0].set_xlabel(r'time [h]')
    axes[0].set_ylabel(r'max $\eta$ [ETH]')
    axes[0].set_title('Max Profit')
    
    # Choose 4 evenly spaced ticks for the x-axis in the first plot
    num_ticks = 4
    tick_positions = np.linspace(t[0], t[-1], num_ticks)
    tick_labels = [prettify_time(t[np.abs(t - i).argmin()]) for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
    
    # Calculate the derivative using np.gradient in the second plot
    derivative = np.gradient(max_profit, t)  # Compute numerical derivative using np.gradient
    axes[1].plot(t, derivative, color='b', linestyle='-', marker=None, linewidth = 2)
    axes[1].set_xlabel(r'time [h]')
    axes[1].set_ylabel(r'$\frac{d}{dt}max \eta$ [ETH/s]')
    axes[1].set_title('Numerical Derivative of Max Profit')

    # Choose 4 evenly spaced ticks for the x-axis in the second plot
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    
    # Set the grid
    for ax in axes:
        ax.grid(True)
    
    # Adjust layout for better fitting
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("images/max_profits_and_derivative.png", dpi=400)
