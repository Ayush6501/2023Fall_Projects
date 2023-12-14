from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def generate_random_number(n: int, m: int = 0) -> int:
    """
    Function to generate a random integer between 'm' and 'n'.
    :param m: max integer value of random number
    :param n: max integer value of random number
    :return: random integer between 0 and n
    >>> num = generate_random_number(10)
    >>> num < 10
    True
    >>> num > 10
    False
    >>> type(num)
    <class 'int'>
    >>> nums = generate_random_number(35, 10)
    >>> 10 < nums < 35
    True
    """
    return np.random.randint(m, n)


def generate_random_quantity(q1: int, q2: int) -> tuple[int | bool | Any, int | Any]:
    """
    Function generate random quantity of production for simulation
    :param q1: Maximum quantity of production
    :param q2: Parameter to tweak quantity of production for Firm 2
    :return: Firm 1 and Firm 2 quantities
    >>> f1q, f2q = generate_random_quantity(51, 51)
    >>> 10 <= f1q < 51
    True
    >>> 10 <= f2q < 51
    True
    >>> q = generate_random_quantity(51, 51)
    >>> type(q[0])
    <class 'int'>
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
    >>> q = generate_optimized_quantity(51, 60, 20)
    >>> 0 <= q[0] < 51
    True
    >>> q[1] == (60 - q[0] - 20) // 2
    True
    """
    q1 = np.random.randint(0, q1)  # Randomly initialize quantities
    q2 = (a - q1 - c) // 2
    return q1, q2


def generate_nash_quantity(a: int, c1: int | float, c2: int | float) -> tuple[int | bool | Any, int | Any]:
    """
    Function to return Firm 1 and Firm 2 quantity of production under the Nash Equilibrium optimization
    which ensure both the firms profit under any given circumstance
    :param a: The demand curve coefficient
    :param c1: Firm 1 cost of production
    :param c2: Firm 2 cost of production
    :return: Firm 1 and Firm 2 quantities
    >>> dcc = 50
    >>> q = generate_nash_quantity(dcc, 25, 20)
    >>> q[0] == (dcc + 20 - (2 * 25)) // 3
    True
    >>> q[1] == (dcc + 25 - (2 * 20)) // 3
    True
    >>> q[0] < dcc and q[1] < dcc
    True
    """
    q1 = (a + c2 - (2 * c1)) // 3
    q2 = (a + c1 - (2 * c2)) // 3
    return q1, q2


def generate_elastic_quantity(
        alpha: int | float,
        beta: float,
        q_a: int,
        q_b: int,
        elasticity: float,
        c1: int | float,
        c2: int | float
) -> tuple[int | bool | Any, int | Any]:
    """
    Function to generate elastic quantity based on demand supply curve
    :param alpha: Intercept on the demand supply curve
    :param beta: Slope of the demand supply curve
    :param q_a: Quantity produced by firm 1
    :param q_b: Quantity produced by firm 2
    :param elasticity: Numerical value for elasticity, value > 1 indicates an elastic market and < 1 denotes
    inelastic markets
    :param c1: Cost of production of firm 1
    :param c2: Cost of production of firm 2
    :return: Firm 1 and Firm 2 quantities
    >>> q = generate_elastic_quantity(1, 0.1, 50, 50, 1.5, 20, 25)
    >>> q[0] == (1 - 20 - 0.1 * 50) / (2 * (1 - 1.5))
    True
    >>> type(q[1])
    <class 'float'>
    >>> len(q)
    2
    """
    q1 = (alpha - c1 - beta * q_b) / (2 * (1 - elasticity))
    q2 = (alpha - c2 - beta * q_a) / (2 * (1 - elasticity))
    return q1, q2


def generate_elastic_demand(q1: int, q2: int, elasticity: float, alpha: int | float, beta: float) -> int | float:
    """
    Function to generate the demand curve coefficient under elastic or inelastic demands.
    :param q1: Firm 1 quantity of production
    :param q2: Firm 2 quantity of production
    :param elasticity: Numerical value for elasticity
    :param alpha: Intercept on the demand supply curve
    :param beta: Slope of the demand supply curve
    :return: demand curve coefficient under elastic or inelastic demand
    >>> dcc = generate_elastic_demand(50, 50, 1.5, 1, 0.1)
    >>> dcc == (1 - (0.1 * (50 + 50))) + (1.5 * (50 + 50))
    True
    >>> dcc >= 0
    True
    """
    return (alpha - (beta * (q1 + q2))) + (elasticity * (q1 + q2))


def plot_profits(df: pd.DataFrame):
    """
    Function to plot the profit values for both the firms after simulation.
    :param df: Dataframe consisting of simulation values
    :return: None
    >>> import pandas as pnd
    >>> import matplotlib.pyplot as pyplt
    >>> sample_data = {'Firm 1 Profit': [100, 150, 200, 250, 300],
    ...                'Firm 2 Profit': [50, 100, 150, 200, 250]}
    >>> d = pnd.DataFrame(sample_data)
    >>> plot_profits(d)
    """
    plt.figure(figsize=(20, 15))
    plt.hist([df['Firm 1 Profit'], df['Firm 2 Profit']], stacked=True, color=['#023047', '#0583D2'])
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
        q2: int = 101,
        elasticity: int | float | None = None,
) -> pd.DataFrame:
    """
    Function to simulate an Oligopoly Market using the Cournot Model. Simulation can be classified into two types -
    Real-World and Theoretical. Theoretical Models can be further broken down into 4 subtypes: Random Simulation,
    Optimized Simulation, Nash Equilibrium & Elastic Markets.
    :param beta: Slope of the demand supply curve
    :param alpha: Intercept of the demand supply curve
    :param elasticity: Numerical value for elasticity
    :param num_iterations: Number of iterations in the simulation
    :param demand_curve_coefficient_mu: Mean value of the demand curve coefficient to estimate market demand
    :param demand_curve_coefficient_sigma: standard deviation of the demand curve coefficient to estimate market demand
    :param c1: Firm 1 cost of production
    :param c2: Firm 2 cost of production
    :param method: The type of simulation, available options are: random, optimize, nash, elastic & real-world
    :param q1: Maximum quantity of production for Firm 1
    :param q2: Maximum quantity of production for Firm 2
    :return: Dataframe consisting of attributes - F1 Quantity, F2 Quantity, F1 Profit & F2 Profit
    >>> n = 10000
    >>> df = cournot_model(n, 100, 10, 20, 25, method='random', q1=101, q2=101)
    >>> df.shape[0] == n
    True
    >>> df.shape[1] == 4
    True
    >>> cournot_model(n, 100, 10, 20, 25, method='elastic', q1=101, q2=101)
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
    >>> df = cournot_model(n, 100, 10, 20, 25, method='nash', q1=101, q2=101)
    >>> df[df['Firm 1 Profit'] > 0].shape == df.shape
    True
    """
    results = {
        'Firm 1 Quantity': [],
        'Firm 2 Quantity': [],
        'Firm 1 Profit': [],
        'Firm 2 Profit': []
    }
    warehouse1 = 0
    warehouse2 = 0

    for _ in range(num_iterations):
        if method == 'real-world':
            demand_shock = np.random.normal(1, 0.2)
            supply_shock = np.random.normal(1, 0.1)

            inflation = np.random.choice([1, 1 + np.random.random(1)[0]], p=[0.7, 0.3])
            disease_outbreak = np.random.choice([1, np.random.random(1)[0]], p=[0.9, 0.1])

            new_c2 = np.random.randint(10, c2) * supply_shock * inflation
            new_c1 = np.random.randint(10, c1) * supply_shock * inflation

            firm1_quantity, firm2_quantity = generate_random_quantity(q1, q2)
            firm1_quantity -= warehouse1
            firm2_quantity -= warehouse2

            warehouse1, warehouse2 = 0, 0
            demand_curve_coefficient = np.random.normal(demand_curve_coefficient_mu, demand_curve_coefficient_sigma,
                                                        1)[0].astype('int')
            market_demand = int(demand_curve_coefficient * disease_outbreak * demand_shock)

            if (firm1_quantity + firm2_quantity) > market_demand:
                surplus = np.random.choice(['q1', 'q2', 'both'])
                if surplus == 'q1':
                    diff = market_demand - firm2_quantity
                    warehouse1 = firm1_quantity - diff
                    firm1_quantity -= warehouse1
                if surplus == 'q2':
                    diff = market_demand - firm1_quantity
                    warehouse2 = firm2_quantity - diff
                    firm2_quantity -= warehouse2
                if surplus == 'both':
                    diff = market_demand - (firm1_quantity + firm2_quantity)
                    warehouse1 = diff // 2
                    warehouse2 = diff // 2
                    firm1_quantity -= warehouse1
                    firm2_quantity -= warehouse2
        else:
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
                demand_curve_coefficient = generate_elastic_demand(firm1_quantity, firm2_quantity, elasticity, alpha,
                                                                   beta)

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
    Function to extract the net profits from the firms as well as the percentage of total profits.
    :param data: Dataframe consisting of attributes - F1 Quantity, F2 Quantity, F1 Profit & F2 Profit
    :return: Dict containing net profits of both the firm and percentage of total profits
    >>> sample_data = {'Firm 1 Quantity': [10, 15, 20, 25, 30],
    ...                'Firm 2 Quantity': [5, 10, 15, 20, 25],
    ...                'Firm 1 Profit': [100, 150, 200, 250, 300],
    ...                'Firm 2 Profit': [50, 100, 150, 200, 250]}
    >>> dt = pd.DataFrame(sample_data)
    >>> d = get_profit(dt)
    >>> type(d['firm1ProfitCount'])
    <class 'int'>
    >>> d['bothProfitCount'] > 0
    True
    >>> d['firm1Profit'].shape == dt[(dt['Firm 1 Profit'] > 0)].shape
    True
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
    Function to visualize the dataset generated after Monte Carlo simulation.
    :param df: Dataframe consisting of attributes - F1 Quantity, F2 Quantity, F1 Profit & F2 Profit
    :return: None
    >>> sample_data = {'Firm 1 Quantity': [10, 15, 20, 25, 30],
    ...                'Firm 2 Quantity': [5, 10, 15, 20, 25],
    ...                'Firm 1 Profit': [100, 150, 200, 250, 300],
    ...                'Firm 2 Profit': [50, 100, 150, 200, 250]}
    >>> d = pd.DataFrame(sample_data)
    >>> visualizer(d)  # doctest: +ELLIPSIS
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
    Function to find the statistical values of the dataset including the mean and standard deviation of the profits.
    Also, the function performs the Mann-Whitney U Test to statistically verify if there is a significant difference
    in mean profits.
    :param data: Dataframe consisting of attributes - F1 Quantity, F2 Quantity, F1 Profit & F2 Profit
    :return: None
    >>> sample_data = {'Firm 1 Profit': [100, 150, 200, 250, 300],
    ...                'Firm 2 Profit': [50, 100, 150, 200, 250]}
    >>> df = pd.DataFrame(sample_data)
    >>> get_statistics(df)
    Mean Profit for Firm 1: 200.0
    Mean Profit for Firm 2: 150.0
    Standard Deviation of Profit for Firm 1: 79.05694150420949
    Standard Deviation of Profit for Firm 2: 79.05694150420949
    ***** MANN-WHITNEY U TEST *****
    Mann-Whitney U statistic: 17.0
    P-value: 0.39761475195653073
    Fail to reject the null hypothesis: There is no significant difference in the mean profits.
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
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in the mean profits.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in the mean profits.")

    return None
