# Test the fixed print statements
metrics = {'annual_return': 0.1234, 'sharpe_ratio': 1.567, 'max_drawdown': 0.089, 'win_rate': 0.654, 'total_trades': 25}
portfolio_values = [100000, 105000, 110000, 108000, 115000]

print('Testing fixed print statements:')
print('   绩效指标:')
print(f'   - 年化收益率: {metrics["annual_return"]:.2%}')
print(f'   - 夏普比率: {metrics["sharpe_ratio"]:.3f}')
print(f'   - 最大回撤: {metrics["max_drawdown"]:.2%}')
print(f'   - 胜率: {metrics["win_rate"]:.1%}')
print(f'   - 总交易次数: {metrics["total_trades"]}')
print(f'   - 最终权益: {portfolio_values[-1]:,.0f}元')

print('All print statements work correctly!')
