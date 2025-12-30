# 投资组合优化框架
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('default')
sns.set_palette("husl")


class PortfolioOptimizer:
    """投资组合优化器"""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.03):
        """
        初始化优化器

        Args:
            returns: 资产收益率数据 (日期 x 资产)
            risk_free_rate: 无风险利率
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)

        # 计算年度化指标
        self.annual_returns = self.mean_returns * 252
        self.annual_cov = self.cov_matrix * 252

    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """计算组合的收益率、波动率和夏普比率"""
        portfolio_return = np.dot(weights, self.annual_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return portfolio_return, portfolio_volatility, sharpe_ratio

    def minimize_volatility(self, target_return: Optional[float] = None) -> Dict:
        """最小化波动率 (或最小化波动率约束目标收益率)"""
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.annual_cov, weights)))

        def constraint_return(weights):
            return np.dot(weights, self.annual_returns) - target_return

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq', 'fun': constraint_return
            })

        # 边界条件 (权重在0-1之间)
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # 初始权重
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # 优化
        result = minimize(
            portfolio_volatility,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)

            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}

    def maximize_sharpe_ratio(self) -> Dict:
        """最大化夏普比率"""
        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, self.annual_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # 最小化负夏普比率

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 边界条件
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # 初始权重
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # 优化
        result = minimize(
            negative_sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)

            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}

    def efficient_frontier(self, n_portfolios: int = 100) -> pd.DataFrame:
        """生成有效前沿"""
        results = []

        for _ in range(n_portfolios):
            # 随机生成权重
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)

            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)

            results.append({
                'return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'weights': weights
            })

        return pd.DataFrame(results)

    def risk_parity_portfolio(self) -> Dict:
        """风险平价组合"""
        def risk_parity_objective(weights):
            # 计算每个资产的波动率贡献
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.annual_cov, weights)))
            asset_vol_contributions = weights * (np.dot(self.annual_cov, weights)) / portfolio_vol

            # 风险平价目标：最小化波动率贡献的方差
            return np.var(asset_vol_contributions)

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 边界条件
        bounds = tuple((0.01, 0.3) for _ in range(self.n_assets))  # 每个资产权重1%-30%

        # 初始权重
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # 优化
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)

            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}

    def black_litterman_model(self, market_caps: pd.Series, tau: float = 0.025,
                            P: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                            omega: Optional[np.ndarray] = None) -> Dict:
        """Black-Litterman模型"""
        # 简化版Black-Litterman实现

        # 市场权重 (市值加权)
        market_weights = market_caps / market_caps.sum()

        # 市场组合的预期超额收益率
        market_return = np.dot(market_weights, self.annual_returns)
        market_vol = np.sqrt(np.dot(market_weights.T, np.dot(self.annual_cov, market_weights)))

        # 市场风险厌恶系数
        risk_aversion = market_return / (market_vol ** 2)

        # 先验预期收益率
        pi = risk_aversion * np.dot(self.annual_cov, market_weights)

        # 如果没有观点，直接使用先验
        if P is None or Q is None:
            posterior_returns = pi
        else:
            # Black-Litterman公式
            if omega is None:
                omega = np.diag(np.diag(P @ (tau * self.annual_cov) @ P.T))

            # 后验预期收益率
            temp = np.linalg.inv(tau * self.annual_cov)
            posterior_returns = np.linalg.inv(temp + P.T @ np.linalg.inv(omega) @ P) @ \
                              (temp @ pi + P.T @ np.linalg.inv(omega) @ Q)

        # 使用后验预期收益率重新优化
        bl_optimizer = PortfolioOptimizer(self.returns, self.risk_free_rate)
        bl_optimizer.annual_returns = posterior_returns

        return bl_optimizer.maximize_sharpe_ratio()

    def robust_optimization(self, uncertainty_level: float = 0.1) -> Dict:
        """鲁棒优化 (考虑参数不确定性)"""
        def robust_objective(weights):
            # 考虑预期收益率的不确定性
            return -np.dot(weights, self.annual_returns) + \
                   uncertainty_level * np.sqrt(np.dot(weights.T, np.dot(self.annual_cov, weights)))

        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]

        # 边界条件
        bounds = tuple((0, 1) for _ in range(self.n_assets))

        # 初始权重
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # 优化
        result = minimize(
            robust_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)

            return {
                'weights': weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': result.message}

    def plot_efficient_frontier(self, n_portfolios: int = 1000):
        """绘制有效前沿"""
        # 生成随机组合
        frontier_data = self.efficient_frontier(n_portfolios)

        # 计算最优组合
        min_vol_portfolio = self.minimize_volatility()
        max_sharpe_portfolio = self.maximize_sharpe_ratio()

        plt.figure(figsize=(12, 8))

        # 绘制随机组合
        plt.scatter(frontier_data['volatility'], frontier_data['return'],
                   c=frontier_data['sharpe_ratio'], cmap='viridis', alpha=0.6, s=10)

        # 绘制最优组合
        if min_vol_portfolio['success']:
            plt.scatter(min_vol_portfolio['volatility'], min_vol_portfolio['expected_return'],
                       color='red', s=100, marker='*', label='Minimum Volatility')

        if max_sharpe_portfolio['success']:
            plt.scatter(max_sharpe_portfolio['volatility'], max_sharpe_portfolio['expected_return'],
                       color='green', s=100, marker='*', label='Maximum Sharpe Ratio')

        # 添加颜色条
        plt.colorbar(label='Sharpe Ratio')

        plt.xlabel('Expected Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_asset_allocation(self, weights: np.ndarray, asset_names: List[str]):
        """绘制资产配置图"""
        plt.figure(figsize=(10, 8))

        # 饼图
        plt.subplot(1, 2, 1)
        plt.pie(weights, labels=asset_names, autopct='%1.1f%%', startangle=90)
        plt.title('Asset Allocation')

        # 柱状图
        plt.subplot(1, 2, 2)
        bars = plt.bar(asset_names, weights * 100)
        plt.ylabel('Weight (%)')
        plt.title('Asset Weights')
        plt.xticks(rotation=45)

        # 添加数值标签
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight*100:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def backtest_portfolio(self, weights: np.ndarray, initial_capital: float = 1000000) -> pd.DataFrame:
        """回测组合表现"""
        # 计算组合收益率
        portfolio_returns = self.returns.dot(weights)

        # 计算组合价值
        portfolio_values = initial_capital * (1 + portfolio_returns).cumprod()

        # 计算绩效指标
        total_return = portfolio_values.iloc[-1] / initial_capital - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol

        # 计算最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        backtest_results = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'portfolio_return': portfolio_returns,
            'cumulative_return': (1 + portfolio_returns).cumprod() - 1,
            'drawdown': drawdown
        })

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

        return backtest_results, metrics


class AdvancedPortfolioAnalyzer:
    """高级组合分析器"""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.optimizer = PortfolioOptimizer(returns)

    def scenario_analysis(self, weights: np.ndarray, scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """情景分析"""
        results = []

        for scenario_name, scenario_params in scenarios.items():
            # 调整预期收益率和波动率
            adjusted_returns = self.optimizer.annual_returns * scenario_params.get('return_multiplier', 1.0)
            adjusted_cov = self.optimizer.annual_cov * scenario_params.get('vol_multiplier', 1.0)

            # 计算组合表现
            portfolio_return = np.dot(weights, adjusted_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))

            results.append({
                'scenario': scenario_name,
                'expected_return': portfolio_return,
                'expected_volatility': portfolio_vol,
                'sharpe_ratio': (portfolio_return - self.optimizer.risk_free_rate) / portfolio_vol
            })

        return pd.DataFrame(results)

    def stress_testing(self, weights: np.ndarray, stress_scenarios: Dict[str, pd.Series]) -> pd.DataFrame:
        """压力测试"""
        results = []

        for scenario_name, stress_returns in stress_scenarios.items():
            # 计算组合在压力情景下的表现
            portfolio_return = np.dot(weights, stress_returns)
            portfolio_value_change = portfolio_return * 100  # 百分比变化

            results.append({
                'scenario': scenario_name,
                'portfolio_return': portfolio_return,
                'value_change_pct': portfolio_value_change
            })

        return pd.DataFrame(results)

    def factor_attribution(self, weights: np.ndarray, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """因子归因分析"""
        # 使用因子模型进行归因
        from sklearn.linear_model import LinearRegression

        portfolio_returns = self.returns.dot(weights)

        # 对每个因子进行回归
        attributions = {}
        total_r_squared = 0

        for factor in factor_returns.columns:
            X = factor_returns[factor].values.reshape(-1, 1)
            y = portfolio_returns.values

            model = LinearRegression()
            model.fit(X, y)

            r_squared = model.score(X, y)
            attributions[factor] = {
                'coefficient': model.coef_[0],
                'r_squared': r_squared,
                'contribution': model.coef_[0] * factor_returns[factor].mean()
            }

            total_r_squared += r_squared

        # 计算未解释部分
        attributions['unexplained'] = {
            'coefficient': None,
            'r_squared': 1 - total_r_squared,
            'contribution': portfolio_returns.mean() - sum(attr['contribution'] for attr in attributions.values() if attr['contribution'] is not None)
        }

        return pd.DataFrame(attributions).T


# === 使用示例 ===

def demo_portfolio_optimization():
    """组合优化演示"""
    # 生成模拟数据
    np.random.seed(42)
    n_assets = 10
    n_days = 252 * 2  # 2年数据

    # 生成随机收益率数据
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (n_days, n_assets)),
        columns=[f'Asset_{i+1}' for i in range(n_assets)],
        index=pd.date_range('2020-01-01', periods=n_days, freq='D')
    )

    print("=== 投资组合优化演示 ===")
    print(f"资产数量: {n_assets}")
    print(f"数据天数: {n_days}")

    # 初始化优化器
    optimizer = PortfolioOptimizer(returns)

    # 1. 最小波动率组合
    print("\n1. 最小波动率组合")
    min_vol_result = optimizer.minimize_volatility()
    if min_vol_result['success']:
        print(".2%")
        print(".2%")
        print(".2f")
        print(f"权重分布: {dict(zip(returns.columns, min_vol_result['weights'].round(3)))}")

    # 2. 最大夏普比率组合
    print("\n2. 最大夏普比率组合")
    max_sharpe_result = optimizer.maximize_sharpe_ratio()
    if max_sharpe_result['success']:
        print(".2%")
        print(".2%")
        print(".2f")
        print(f"权重分布: {dict(zip(returns.columns, max_sharpe_result['weights'].round(3)))}")

    # 3. 风险平价组合
    print("\n3. 风险平价组合")
    risk_parity_result = optimizer.risk_parity_portfolio()
    if risk_parity_result['success']:
        print(".2%")
        print(".2%")
        print(".2f")
        print(f"权重分布: {dict(zip(returns.columns, risk_parity_result['weights'].round(3)))}")

    # 4. 鲁棒优化
    print("\n4. 鲁棒优化")
    robust_result = optimizer.robust_optimization(uncertainty_level=0.1)
    if robust_result['success']:
        print(".2%")
        print(".2%")
        print(".2f")

    # 5. 绘制有效前沿
    print("\n5. 生成有效前沿图...")
    try:
        optimizer.plot_efficient_frontier(n_portfolios=500)
    except:
        print("绘图失败，请检查matplotlib配置")

    # 6. 回测最优组合
    print("\n6. 回测最大夏普比率组合")
    if max_sharpe_result['success']:
        backtest_results, metrics = optimizer.backtest_portfolio(
            max_sharpe_result['weights'], initial_capital=1000000
        )

        print(".2%")
        print(".2%")
        print(".2f")
        print(".2%")

    return optimizer, max_sharpe_result


if __name__ == "__main__":
    demo_portfolio_optimization()
