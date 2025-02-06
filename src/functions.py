import numpy as np
from typing import Union

# Constants from constants.py are assumed to be in Decimal
from constants import *
from helpers import prettify_time, log_message

def P_DPA(t: Union[float, np.ndarray], P0 : float = P_DPA_0 ) -> Union[float, np.ndarray]:
    """
    Calculates P_DPA(t) = P_DPA_0 * (0.99)**(t / 60).
    Uses Decimal precision when t is a scalar.

    Args:
        t : Time, either as a scalar or array.

    Returns:
        Numeric: Value of P_DPA(t).
    """
    return P0 * (0.99 ** (t / 60))

def P_ACC(m: Union[float, np.ndarray],  E0: float = E_0, M0: float = M_0) -> Union[float, np.ndarray]:
    """
    Calculates P_ACC(m) = E_0 / (M_0 + 2 * m).
    Uses Decimal precision when m is a scalar.

    Args:
        m (Numeric): Mass, either as a scalar or array.

    Returns:
        Numeric: Value of P_ACC(m).
    """
    return E0 / (M0 + 2 * m)

def get_profit_value(m: float, t: float = 0, E0: float = E_0, M0: float = M_0, P0: float = P_DPA_0) -> float:
    """
    Calculates profit(m, t) = (P_ACC(m) - P_DPA(t)) * m for a single m and t.

    Args:
        m (float): MET.
        t (float): Time.
        E0 (float): Initial energy constant (default E_0).
        M0 (float): Initial mass constant (default M_0).
        P0 (float): Initial DPA power constant (default P_DPA_0).

    Returns:
        float: Value of profit(m, t).
    """
    P_ACC_val = E0 / (M0 + 2 * m)
    P_DPA_val = P0 * (0.99 ** (t / 60))
    return (P_ACC_val - P_DPA_val) * m


def get_profit_function(m_values, t_values, E0: float = E_0, M0: float = M_0, P0: float = P_DPA_0):
    """
    Calculates profit for all combinations of m and t efficiently.

    Args:
        m_values (Union[float, list, np.ndarray]): MET value(s).
        t_values (Union[float, list, np.ndarray]): Time value(s).
        E0 (float): Initial energy constant (default E_0).
        M0 (float): Initial mass constant (default M_0).
        P0 (float): Initial DPA power constant (default P_DPA_0).

    Returns:
        Union[list, np.ndarray]: 1D or 2D list of profit values depending on inputs.
    """
    # Check if m_values or t_values are scalars
    m_is_scalar = np.isscalar(m_values)
    t_is_scalar = np.isscalar(t_values)

    # Convert scalars to lists for iteration
    m_values = [m_values] if m_is_scalar else m_values
    t_values = [t_values] if t_is_scalar else t_values

    results = []
    for m in m_values:
        row = []
        for t in t_values:
            row.append(get_profit_value(m, t, E0, M0, P0))
        results.append(row)

    # Return appropriate shape
    if m_is_scalar:  # m is scalar, t is a vector
        return results[0]  # 1D list
    elif t_is_scalar:  # t is scalar, m is a vector
        return [row[0] for row in results]  # 1D list
    else:  # Both are vectors
        return results  # 2D list

def find_t_star(time_window: np.ndarray, E0: float = E_0, M0: float = M_0, P0: float = P_DPA_0) -> (float, float):
    """
    Finds t* such that the equation:
    (E_0 / P_DPA_0) * (0.99)**(t*/60) - M_0 > 0
    is satisfied using binary search for efficiency.

    Args:
        time_window (np.ndarray): The time window of the day, sorted in increasing order.
    
    Returns:
        float: The value of t* that satisfies the condition.
    """
    
    # Vectorized computation of left-hand side and MET values
    left_hand_side = (E0 / P0) * (0.99 ** (-time_window / 60))
    met_values = left_hand_side - M0
    
    # Find the first index where met_value is greater than 0
    idx = np.argmax(met_values > 0)
    
    if idx == 0:  # If the first element already satisfies the condition
        return time_window[0], met_values[0] / 2
    
    # Otherwise, return the first t and corresponding MET value where condition is satisfied
    return time_window[idx], met_values[idx] / 2

def find_t_max_profit(t_star: float, met_max: float, pct: float = PROFIT_PCT_STRATEGY, time_step: float = 1.0, E0: float = E_0, M0: float = M_0, P0: float = P_DPA_0) -> float:
    """
    Finds the lowest time t_max such that max(profit(t, m)) >= (1 + pct) * FEE.

    Args:
        t_star (float): Starting time to evaluate profits.
        met_max (float): Maximum MET value to consider.
        pct (float): Percentage threshold.
        time_step (float): Time step increment (default: 1.0).
        E0 (float): Initial energy constant (default: E_0).
        M0 (float): Initial mass constant (default: M_0).
        P0 (float): Initial DPA power constant (default: P_DPA_0).

    Returns:
        float: The smallest time t_max satisfying the condition.
    """
    # Threshold for the profit
    threshold = (1 + pct) * FEE

    # Initialize t
    t = t_star

    while True:
        # Initialize variables to track the maximum profit and corresponding m
        best_m = None

        # Iterate over m values to find max profit and corresponding m
        for m in np.arange(0, met_max, 0.00000001):
            profit = get_profit_value(m, t, E0, M0, P0)
            if profit >= threshold:
                best_m = m
                return t, best_m

        # Increment time
        t += time_step
        met_max = compute_met_limit(t)

def compute_met_limit_single(t: float, E0: float = E_0, M0: float = M_0, P0: float = P_DPA_0) -> float:
    """
    Computes m_d* for a single time.
    Args:
        t (float): Time value.

    Returns:
        float: The upper MET bound m_d* for the given time.
    """
    left_hand_side = (E0 / P0) * (0.99 ** (-t / 60))
    met_value = left_hand_side - M0
    return max(met_value / 2, 0)  # Ensure MET is non-negative

def compute_met_limit(time_window: Union[float, np.ndarray], E0: float = E_0, M0: float = M_0, P0: float = P_DPA_0) -> Union[float, np.ndarray]:
    """
    Finds m_d* (the upper bound for MET bought at the DPA) for a single time or an array of times.

    Args:
        time_window (Union[float, np.ndarray]): Single time or array of times to evaluate.
        E0 (float): Initial energy constant (default: E_0).
        M0 (float): Initial mass constant (default: M_0).
        P0 (float): Initial DPA power constant (default: P_DPA_0).

    Returns:
        Union[float, np.ndarray]: m_d* for the given time(s).
    """

    # Handle scalar input
    if np.isscalar(time_window):
        return compute_met_limit_single(time_window)

    # Handle array input
    return np.array([compute_met_limit_single(t) for t in time_window])

def simulate_strategy(time_step: float) -> list:
    """
    Simulates the arbitrage strategy over time, adjusting resources dynamically and computing the profit.
    
    Args:
        time_step (float): Time step increment for the simulation (default: 0.1).
    
    Returns:
        list: A list containing the profit at each time step where arbitrage occurred.
    """
    # Time array from 0 to T with T points
    time = np.arange(0, T, time_step)
    
    # Initial conditions: find starting time and MET limit
    t_star, met_max = find_t_star(time)
    t_lim, m_lim = find_t_max_profit(t_star, met_max, PROFIT_PCT_STRATEGY, time_step)
    
    # Initial resources
    DAP_m0, m0, e0 = MET_LIMIT, M_0, E_0
    
    # List to store the results
    results = []
    time = np.arange(t_star, t_star + 60, time_step)
    
    print(" ")
    print("Simulating the arbitrage strategy")
    counter = 0
    # Loop over each time step
    for t in time:
        if t >= t_lim and DAP_m0 > 0:  # Execute arbitrage if time exceeds limit
            if m_lim > DAP_m0: # We cannot profit enough from this trade, just exit
                return results
            counter +=1

            # Calculate the ETH obtained from the ACC with m_d = m_lim
            e_a = P_ACC(m_lim, e0, m0)*m_lim
            e0 -= e_a  # Update ACC ETH reservoir
            m0 += m_lim  # Update ACC MET reservoir
            DAP_m0 -= m_lim
            #profit = get_profit_value(m_lim, t_lim) # This should be the correct formula but I have numerical instabilities due to approximations of time and MET
            profit = (1 + PROFIT_PCT_STRATEGY) * FEE

            results.append((t, profit, e0, m0, DAP_m0))  # Store the time and profit

            log_message(
                f"t trade {prettify_time(t_lim, include_milliseconds=True)} with {m_lim} MET "
                f"for a net profit of {(profit-FEE)/GWEI} GWEI"
            )
            if counter % 60 == 0 or counter == 1:
                print(f"t trade {prettify_time(t_lim, include_milliseconds=True)} with {m_lim} MET for a net profit of {(profit-FEE)/GWEI} GWEI")

            # Recalculate next arbitrage opportunity based on updated resources
            time_window = time[np.where(time > t)[0][0]:-1]
            if time_window.size == 0:
                return results
            # 
            t_star, met_max = find_t_star(time_window, E0=e0, M0=m0)
            t_lim, m_lim = find_t_max_profit(t_star, met_max, PROFIT_PCT_STRATEGY, time_step, E0=e0, M0=m0)
        
        elif t < t_lim and DAP_m0 > 0:
            continue  # No arbitrage if time hasn't reached the limit
        else:
            return results

    return results
