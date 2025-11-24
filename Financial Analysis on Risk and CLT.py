import numpy as np
import pandas as pd
from scipy.stats import norm
days = 252 * 10
n_assets_list = [10, 30, 100]
mu = 0.0005
sigma = 0.01
simulations = 10000
np.random.seed(42)
sampling_means = {}
for n_assets in n_assets_list:

    asset_returns = np.random.normal(mu, sigma, size=(days, n_assets))

    portfolio_returns = asset_returns.mean(axis=1)

    resampled_means = [
        portfolio_returns[np.random.choice(len(portfolio_returns), n_assets)].mean()
        for _ in range(simulations)
    ]
    sampling_means[n_assets] = np.array(resampled_means)
largest_n = n_assets_list[-1]
port_mean = np.mean(sampling_means[largest_n])
port_stdev = np.std(sampling_means[largest_n])
norm_var_95 = norm.ppf(0.05, loc=port_mean, scale=port_stdev)
nonparam_var_95 = np.percentile(sampling_means[largest_n], 5)
summary = pd.DataFrame({
    'Portfolio_Size': n_assets_list,
    'Sampling_Mean': [np.mean(sampling_means[n]) for n in n_assets_list],
    'Sampling_Std': [np.std(sampling_means[n]) for n in n_assets_list]
})
summary['Norm_VaR_95'] = np.nan
summary['Empirical_VaR_95'] = np.nan
summary.loc[summary['Portfolio_Size'] == largest_n, 'Norm_VaR_95'] = norm_var_95
summary.loc[summary['Portfolio_Size'] == largest_n, 'Empirical_VaR_95'] = nonparam_var_95
summary.to_csv("portfolio_clt_summary.csv", index=False)
print(summary)