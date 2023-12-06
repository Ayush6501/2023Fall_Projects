from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# TODO: Unit tests


def generate_random_number(n: int, m: int = 0) -> int:
    """
    Function to generate a random integer between 0 and 'n'.
    :param m: max integer value of random number
    :param n: max integer value of random number
    :return: random integer between 0 and n
    """
    return np.random.randint(m, n)


def generate_random_quantity(q1: int, q2: int) -> tuple[int | bool | Any, int | Any]:
    """
    Function generate random quantity of production for simulation
    :param q1: Maximum quantity of production
    :param q2: Parameter to tweak quantity of production for Firm 2
    :return: Firm 1 and Firm 2 quantities
    """
    q1 = np.random.randint(10, q1)  # Randomly initialize quantities
    q2 = np.random.randint(10, q2)
    return q1, q2


def generate_optimized_quantity(q1: int, a: int, c: int | float) -> tuple[int | bool | Any, int | Any]:
    """
    Function to return a random quantity of production for Firm 1 and an optimized quantity for Firm 2 which
    depends on Firm 1 production
    :param q1: Maximum quantity of production
    :param a: The demand curve coefficient
    :param c: Firm 2 cost of production
    :return: Firm 1 and Firm 2 quantities
    """
    q1 = np.random.randint(0, q1)  # Randomly initialize quantities
    q2 = (a - q1 - c) // 2
    return q1, q2


def generate_nash_quantity(a, c1, c2) -> tuple[int | bool | Any, int | Any]:
    """
    Function to return Firm 1 and Firm 2 quantity of Production under the Nash Equilibrium optimization
    which ensure both the firms profit under any given circumstance
    :param a: The demand curve coefficient
    :param c1: Firm 1 cost of production
    :param c2: Firm 2 cost of production
    :return: Firm 1 and Firm 2 quantities
    """
    q1 = (a + c2 - (2 * c1)) // 3
    q2 = (a + c1 - (2 * c2)) // 3
    return q1, q2


def generate_elastic_quantity(alpha, beta, q_a, q_b, elasticity, c1, c2):
    q1 = (alpha - c1 - beta * q_b) / (2 * (1 - elasticity))
    q2 = (alpha - c2 - beta * q_a) / (2 * (1 - elasticity))
    return q1, q2


def generate_elastic_demand(q1, q2, elasticity, alpha, beta):
    return (alpha - (beta * (q1 + q2))) + (elasticity * (q1 + q2))


def plot_profits(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    plt.figure(figsize=(20, 15))
    plt.hist([df['Firm 1 Profit'], df['Firm 2 Profit']], stacked=True, color=['orange', 'blue'])
    plt.xlabel('Profit', fontsize=20)
    plt.ylabel('No. of occurrences', fontsize=20)
    plt.legend(['Firm 1 Profit', 'Firm 2 Profit'], fontsize=20)
    plt.show()
    return None


def cournot_model(
        num_iterations: int,
        demand_curve_coefficient_mu: int,
        demand_curve_coefficient_sigma: int,
        c1: int | float,
        c2: int | float,
        method: str,
        alpha: int | float | None = None,
        beta: int | float | None = None,
        q1: int = 101,
        q2: int = 0,
        elasticity: int | float | None = None,
) -> pd.DataFrame:
    """

    :param beta:
    :param alpha:
    :param elasticity:
    :param num_iterations:
    :param demand_curve_coefficient_mu:
    :param demand_curve_coefficient_sigma:
    :param c1:
    :param c2:
    :param method:
    :param q1:
    :param q2:
    :return:
    """
    results = {
        'Firm 1 Quantity': [],
        'Firm 2 Quantity': [],
        'Firm 1 Profit': [],
        'Firm 2 Profit': []
    }

    for _ in range(num_iterations):
        new_c2 = np.random.randint(10, c2)
        new_c1 = np.random.randint(10, c1)
        a = np.random.normal(demand_curve_coefficient_mu, demand_curve_coefficient_sigma, 1)
        demand_curve_coefficient = a[0].astype('int')
        if method == 'random':
            firm1_quantity, firm2_quantity = generate_random_quantity(q1, q2)
        elif method == 'optimize':
            firm1_quantity, firm2_quantity = generate_optimized_quantity(q1, demand_curve_coefficient, c2)
        elif method == 'elastic':
            new_quantity_1, new_quantity_2 = generate_random_quantity(q1, q2)
            firm1_quantity, firm2_quantity = generate_elastic_quantity(alpha, beta, new_quantity_1, new_quantity_2,
                                                                       elasticity, c1, c2)
            firm1_quantity, firm2_quantity = max(0, firm1_quantity), max(0, firm2_quantity)
        elif method == 'nash':
            firm1_quantity, firm2_quantity = generate_nash_quantity(demand_curve_coefficient, new_c1, new_c2)
        else:
            firm1_quantity, firm2_quantity = generate_random_quantity(q1, q2)

        if elasticity and method == 'elastic':
            demand_curve_coefficient = generate_elastic_demand(firm1_quantity, firm2_quantity, elasticity, alpha, beta)

        market_demand = demand_curve_coefficient - (firm1_quantity + firm2_quantity)

        firm1_profit = (market_demand * firm1_quantity) - (new_c1 * firm1_quantity)
        firm2_profit = (market_demand * firm2_quantity) - (new_c2 * firm2_quantity)

        results['Firm 1 Quantity'].append(firm1_quantity)
        results['Firm 2 Quantity'].append(firm2_quantity)
        results['Firm 1 Profit'].append(firm1_profit)
        results['Firm 2 Profit'].append(firm2_profit)

    return pd.DataFrame(results)


def get_profit(data: pd.DataFrame) -> dict:
    """

    :param data:
    :return:
    """
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
    """

    :param df:
    :return:
    """
    figure, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))

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
    """

    :param data:
    :return:
    """
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
    alpha = 0.03
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in the mean profits.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in the mean profits.")

    return None
