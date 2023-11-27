from typing import Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def generate_random_number(n) -> int:
    return np.random.randint(0, n)


def generate_random_quantity(q1max, q2) -> tuple[int | bool | Any, int | Any]:
    q1 = np.random.randint(0, q1max)  # Randomly initialize quantities
    q2 = np.random.randint(0, q1max - q2)
    return q1, q2


def generate_optimized_quantity(q1max, a, c) -> tuple[int | bool | Any, int | Any]:
    q1 = np.random.randint(0, q1max)  # Randomly initialize quantities
    q2 = (a - q1 - c) // 2
    return q1, q2


def mc_sim(num_iterations, demand_curve_coefficient, c1, c2, q2, method, q1max=101) -> pd.DataFrame:
    results = {
        'Firm 1 Quantity': [],
        'Firm 2 Quantity': [],
        'Firm 1 Profit': [],
        'Firm 2 Profit': []
    }

    for _ in range(num_iterations):
        if method == 'random':
            firm1_quantity, firm2_quantity = generate_random_quantity(q1max, q2)
        elif method == 'optimize':
            firm1_quantity, firm2_quantity = generate_optimized_quantity(q1max, demand_curve_coefficient, c2)
        else:
            firm1_quantity, firm2_quantity = generate_random_quantity(q1max, q2)

        market_demand = demand_curve_coefficient - (firm1_quantity + firm2_quantity)

        firm1_profit = (market_demand - firm1_quantity) * firm1_quantity - c1 * firm1_quantity
        firm2_profit = (market_demand - firm2_quantity) * firm2_quantity - c2 * firm2_quantity

        results['Firm 1 Quantity'].append(firm1_quantity)
        results['Firm 2 Quantity'].append(firm2_quantity)
        results['Firm 1 Profit'].append(firm1_profit)
        results['Firm 2 Profit'].append(firm2_profit)

    return pd.DataFrame(results)


def get_profit(data: pd.DataFrame) -> dict:
    firm1_net_profit = data[(data['Firm 1 Profit'] > 0)]
    firm2_net_profit = data[(data['Firm 2 Profit'] > 0)]
    both_firm_total_profit = data[(data['Firm 1 Profit'] > 0) & (data['Firm 2 Profit'] > 0)]

    return {
        'firm1Profit': firm1_net_profit,
        'firm2Profit': firm2_net_profit,
        'bothProfit': both_firm_total_profit,
        'firm1ProfitCount': firm1_net_profit.shape[0],
        'firm2ProfitCount': firm2_net_profit.shape[0],
        'bothProfitCount': both_firm_total_profit.shape[0],
    }


def visualizer(df: pd.DataFrame) -> None:
    figure, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))

    ax1.scatter(df['Firm 1 Quantity'], df['Firm 2 Quantity'])
    ax1.set_title('Firm 1 Quantity vs Firm 2 Quantity')

    ax2.scatter(df['Firm 1 Profit'], df['Firm 2 Profit'])
    ax2.set_title('Firm 1 Profit vs Firm 2 Profit')

    ax3.scatter(df['Firm 1 Quantity'], df['Firm 1 Profit'])
    ax3.set_title('Firm 1 Quantity vs Firm 1 Profit')

    ax4.scatter(df['Firm 2 Quantity'], df['Firm 2 Profit'])
    ax4.set_title('Firm 2 Quantity vs Firm 2 Profit')

    plt.show()

    return None


def get_statistics(data: pd.DataFrame) -> None:
    firm1p_mean = data['Firm 1 Profit'].mean()
    firm2p_mean = data['Firm 2 Profit'].mean()
    print(f'Mean Profit for Firm 1: {firm1p_mean}')
    print(f'Mean Profit for Firm 2: {firm2p_mean}')

    firm1p_std = data['Firm 1 Profit'].std()
    firm2p_std = data['Firm 2 Profit'].std()

    print(f'Standard Deviation of Profit for Firm 1: {firm1p_std}')
    print(f'Standard Deviation of Profit for Firm 2: {firm2p_std}')

    data.boxplot(column=['Firm 1 Profit', 'Firm 2 Profit'])

    print('***** MANN-WHITNEY U TEST *****')
    statistic, p_value = mannwhitneyu(data['Firm 1 Profit'], data['Firm 2 Profit'], alternative='two-sided')

    print(f"Mann-Whitney U statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Check if the p-value is less than the significance level (e.g., 0.05)
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in the mean profits.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in the mean profits.")

    return None
