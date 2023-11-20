import numpy as np
import pandas as pd


def mc_sim(num_iterations, demand_curve_coefficient, c1, c2, q1max, q2):
    results = {
        'Firm 1 Quantity': [],
        'Firm 2 Quantity': [],
        'Firm 1 Profit': [],
        'Firm 2 Profit': []
    }

    for _ in range(num_iterations):
        firm1_quantity = np.random.randint(0, q1max)  # Randomly initialize quantities
        firm2_quantity = np.random.randint(0, q1max-q2)

        market_demand = demand_curve_coefficient - (firm1_quantity + firm2_quantity)

        firm1_profit = (market_demand - firm1_quantity) * firm1_quantity - c1 * firm1_quantity
        firm2_profit = (market_demand - firm2_quantity) * firm2_quantity - c2 * firm2_quantity

        results['Firm 1 Quantity'].append(firm1_quantity)
        results['Firm 2 Quantity'].append(firm2_quantity)
        results['Firm 1 Profit'].append(firm1_profit)
        results['Firm 2 Profit'].append(firm2_profit)

    return pd.DataFrame(results)


def get_profit(data: pd.DataFrame):
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
