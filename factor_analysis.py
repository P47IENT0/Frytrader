# 量化因子分析框架
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FactorAnalyzer:
    """因子分析器"""

    def __init__(self):
        self.price_data = {}
        self.factor_data = {}
        self.factor_returns = {}

    def load_price_data(self, price_data: Dict[str, pd.DataFrame]):
        """加载价格数据"""
        self.price_data = price_data
        print(f"加载了{len(price_data)}只股票的价格数据")

    # === 技术因子 ===

    def calculate_momentum_factors(self, window: int = 20) -> pd.DataFrame:
        """计算动量因子"""
        momentum_data = {}

        for symbol, df in self.price_data.items():
            # 价格动量
            df['momentum'] = df['close'] / df['close'].shift(window) - 1

            # 成交量动量
            df['volume_momentum'] = df['volume'] / df['volume'].shift(window) - 1

            # 波动率动量 (近期波动率相对长期波动率)
            short_vol = df['close'].pct_change().rolling(5).std()
            long_vol = df['close'].pct_change().rolling(window).std()
            df['volatility_momentum'] = short_vol / long_vol - 1

            momentum_data[symbol] = df[['momentum', 'volume_momentum', 'volatility_momentum']]

        # 合并为宽表格式
        momentum_df = self._merge_factor_data(momentum_data)
        return momentum_df

    def calculate_mean_reversion_factors(self) -> pd.DataFrame:
        """计算均值回归因子"""
        mr_data = {}

        for symbol, df in self.price_data.items():
            # 相对于移动平均线的偏离度
            ma20 = df['close'].rolling(20).mean()
            df['ma_deviation'] = (df['close'] - ma20) / ma20

            # RSI偏离
            df['rsi_deviation'] = df['RSI'] - 50  # 相对于50的偏离

            # 布林带位置
            df['bb_position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) - 0.5

            mr_data[symbol] = df[['ma_deviation', 'rsi_deviation', 'bb_position']]

        mr_df = self._merge_factor_data(mr_data)
        return mr_df

    def calculate_volatility_factors(self) -> pd.DataFrame:
        """计算波动率因子"""
        vol_data = {}

        for symbol, df in self.price_data.items():
            # 历史波动率
            df['hist_vol_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            df['hist_vol_60'] = df['close'].pct_change().rolling(60).std() * np.sqrt(252)

            # 波动率偏度
            returns = df['close'].pct_change().dropna()
            df['vol_skewness'] = returns.rolling(60).skew()

            # 波动率变化率
            df['vol_change'] = df['hist_vol_20'] / df['hist_vol_60'] - 1

            vol_data[symbol] = df[['hist_vol_20', 'hist_vol_60', 'vol_skewness', 'vol_change']]

        vol_df = self._merge_factor_data(vol_data)
        return vol_df

    # === 价值因子 ===

    def calculate_value_factors(self, financial_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算价值因子"""
        value_data = {}

        for symbol in self.price_data.keys():
            if symbol in financial_data:
                fin_df = financial_data[symbol]

                # 计算最新财务指标
                latest_fin = fin_df.iloc[-1] if not fin_df.empty else pd.Series()

                if not latest_fin.empty:
                    # 市盈率 (需要股价数据)
                    pe_ratio = None  # 需要补充计算

                    # 市净率 (需要净资产数据)
                    pb_ratio = None  # 需要补充计算

                    # 股息率
                    dividend_yield = None  # 需要股息数据

                    value_data[symbol] = pd.Series({
                        'pe_ratio': pe_ratio,
                        'pb_ratio': pb_ratio,
                        'dividend_yield': dividend_yield
                    })

        value_df = pd.DataFrame.from_dict(value_data, orient='index')
        return value_df

    # === 质量因子 ===

    def calculate_quality_factors(self, financial_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算质量因子"""
        quality_data = {}

        for symbol in self.price_data.keys():
            if symbol in financial_data:
                fin_df = financial_data[symbol]

                if not fin_df.empty:
                    # 盈利能力指标
                    latest = fin_df.iloc[-1]

                    # 净利率
                    net_margin = latest.get('net_profit_margin', np.nan)

                    # ROE
                    roe = latest.get('roe', np.nan)

                    # ROA
                    roa = latest.get('roa', np.nan)

                    quality_data[symbol] = pd.Series({
                        'net_margin': net_margin,
                        'roe': roe,
                        'roa': roa
                    })

        quality_df = pd.DataFrame.from_dict(quality_data, orient='index')
        return quality_df

    # === 因子处理和分析 ===

    def _merge_factor_data(self, factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并因子数据为宽表格式"""
        merged_data = []

        for symbol, df in factor_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            merged_data.append(df_copy)

        if merged_data:
            result = pd.concat(merged_data)
            # 重塑为宽表格式 (日期 x 因子_股票)
            result = result.pivot(columns='symbol')
            result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
            return result

        return pd.DataFrame()

    def normalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """标准化因子数据"""
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(factor_df)
        normalized_df = pd.DataFrame(normalized_data, index=factor_df.index, columns=factor_df.columns)
        return normalized_df

    def calculate_factor_correlation(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """计算因子相关性"""
        return factor_df.corr()

    def perform_pca_analysis(self, factor_df: pd.DataFrame, n_components: int = 5) -> Tuple[pd.DataFrame, PCA]:
        """主成分分析"""
        # 去除缺失值
        factor_df_clean = factor_df.dropna()

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(factor_df_clean)

        pc_df = pd.DataFrame(
            principal_components,
            index=factor_df_clean.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        return pc_df, pca

    def calculate_ic_analysis(self, factor_df: pd.DataFrame, returns_df: pd.DataFrame,
                            window: int = 20) -> Dict[str, pd.Series]:
        """IC分析 (信息系数)"""
        ic_results = {}

        for factor_col in factor_df.columns:
            factor_name = factor_col.split('_')[0]  # 提取因子名称

            if factor_name not in ic_results:
                ic_results[factor_name] = []

            # 计算滚动IC
            for i in range(window, len(factor_df)):
                factor_slice = factor_df[factor_col].iloc[i-window:i]
                returns_slice = returns_df.iloc[i-window:i]

                # 确保数据对齐
                common_index = factor_slice.index.intersection(returns_slice.index)
                if len(common_index) > 5:  # 至少需要5个观测值
                    ic = factor_slice.loc[common_index].corr(returns_slice.loc[common_index])
                    ic_results[factor_name].append(ic)

        # 转换为Series
        for factor_name in ic_results:
            ic_results[factor_name] = pd.Series(ic_results[factor_name])

        return ic_results

    def factor_portfolio_construction(self, factor_df: pd.DataFrame, method: str = 'equal_weight') -> pd.DataFrame:
        """基于因子的投资组合构建"""
        if method == 'equal_weight':
            # 等权重组合
            weights = np.ones(len(factor_df.columns)) / len(factor_df.columns)
            portfolio = factor_df.dot(weights)

        elif method == 'factor_mimicking':
            # 因子模仿组合 (简化版)
            # 这里可以实现更复杂的因子模型
            portfolio = factor_df.mean(axis=1)

        portfolio_df = pd.DataFrame(portfolio, columns=['portfolio_return'])
        return portfolio_df

    # === 因子数据存储 ===

    def save_factors(self, factors_dict: Dict[str, pd.DataFrame], path: str = './factors/'):
        """保存因子数据"""
        import os
        os.makedirs(path, exist_ok=True)

        for factor_name, df in factors_dict.items():
            filepath = f"{path}{factor_name}_factors.csv"
            df.to_csv(filepath)
            print(f"保存{factor_name}因子到{filepath}")

    def load_factors(self, factor_names: List[str], path: str = './factors/') -> Dict[str, pd.DataFrame]:
        """加载因子数据"""
        factors = {}

        for factor_name in factor_names:
            filepath = f"{path}{factor_name}_factors.csv"
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                factors[factor_name] = df
            except FileNotFoundError:
                print(f"因子文件{filepath}不存在")

        return factors


# === 使用示例 ===

def demo_factor_analysis():
    """因子分析演示"""
    from data_sources import QuantDataManager

    # 初始化
    fa = FactorAnalyzer()
    dm = QuantDataManager()

    # 获取数据
    symbols = ['000001', '600000', '000002', '600036']  # 示例股票
    start_date = '20240101'
    end_date = '20241201'

    print("=== 获取价格数据 ===")
    price_data = dm.get_daily_prices(symbols, start_date, end_date)
    fa.load_price_data(price_data)

    # 计算技术因子
    print("\n=== 计算技术因子 ===")

    # 为价格数据添加技术指标
    for symbol in price_data:
        price_data[symbol] = dm.calculate_technical_indicators(price_data[symbol])

    # 重新加载数据
    fa.load_price_data(price_data)

    # 计算各类因子
    momentum_factors = fa.calculate_momentum_factors()
    mr_factors = fa.calculate_mean_reversion_factors()
    vol_factors = fa.calculate_volatility_factors()

    print(f"动量因子形状: {momentum_factors.shape}")
    print(f"均值回归因子形状: {mr_factors.shape}")
    print(f"波动率因子形状: {vol_factors.shape}")

    # 合并所有因子
    all_factors = pd.concat([momentum_factors, mr_factors, vol_factors], axis=1)
    print(f"\n所有因子形状: {all_factors.shape}")

    # 因子标准化
    print("\n=== 因子标准化 ===")
    normalized_factors = fa.normalize_factors(all_factors.dropna())
    print(f"标准化后因子形状: {normalized_factors.shape}")

    # 主成分分析
    print("\n=== 主成分分析 ===")
    pc_df, pca = fa.perform_pca_analysis(normalized_factors, n_components=3)
    print(f"主成分形状: {pc_df.shape}")
    print(f"解释方差比例: {pca.explained_variance_ratio_}")

    # 因子相关性分析
    print("\n=== 因子相关性分析 ===")
    correlation_matrix = fa.calculate_factor_correlation(normalized_factors)
    print("相关性矩阵形状:", correlation_matrix.shape)

    # 保存因子数据
    factors_dict = {
        'momentum': momentum_factors,
        'mean_reversion': mr_factors,
        'volatility': vol_factors,
        'all_factors': all_factors
    }

    fa.save_factors(factors_dict)

    return factors_dict


if __name__ == "__main__":
    demo_factor_analysis()
