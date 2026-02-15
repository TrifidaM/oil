import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 页面配置 ---
st.set_page_config(page_title="OilAlpha 原油量化终端", layout="wide")

# --- 核心数学逻辑 ---
def generate_data(volatility, trend, days=365):
    """模拟生成原油期货数据 (几何布朗运动)"""
    np.random.seed(42)
    start_price = 75.0
    dates = pd.date_range(end=datetime.today(), periods=days)
    returns = np.random.normal(loc=trend/days, scale=volatility/np.sqrt(days), size=days)
    price = start_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({'Date': dates, 'Close': price})
    return df

def backtest_strategy(df, short_window, long_window, initial_capital):
    """计算双均线策略"""
    # 1. 计算指标
    df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
    df['Long_MA'] = df['Close'].rolling(window=long_window).mean()
    
    # 2. 生成信号 (1=买入, 0=空仓)
    df['Signal'] = 0.0
    df.loc[short_window:, 'Signal'] = np.where(
        df['Short_MA'][short_window:] > df['Long_MA'][short_window:], 1.0, 0.0
    )
    
    # 3. 计算持仓 (Signal shift 1天，因为今天的信号指导明天的交易)
    df['Position'] = df['Signal'].shift(1)
    
    # 4. 计算收益
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Position']
    
    # 5. 资金曲线
    df['Equity_Curve'] = initial_capital * (1 + df['Strategy_Return'].fillna(0)).cumprod()
    
    return df

# --- 侧边栏：参数控制 ---
st.sidebar.header("⚙️ 策略参数配置")
initial_capital = st.sidebar.number_input("初始资金 ($)", value=100000)
short_ma = st.sidebar.slider("短期均线 (天)", 5, 50, 10)
long_ma = st.sidebar.slider("长期均线 (天)", 20, 100, 30)

st.sidebar.markdown("---")
st.sidebar.header("📊 市场模拟环境")
volatility = st.sidebar.slider("市场波动率", 0.1, 0.5, 0.3)
trend = st.sidebar.slider("年度趋势", -0.2, 0.2, 0.05)

# --- 主界面 ---
st.title("🛢️ OilAlpha 原油量化策略系统")
st.markdown("基于 **双均线交叉 (Golden Cross)** 算法的实时回测终端")

# 1. 运行引擎
if short_ma >= long_ma:
    st.error("❌ 错误：短期均线必须小于长期均线！")
else:
    # 生成数据 & 运行回测
    df = generate_data(volatility, trend)
    result_df = backtest_strategy(df, short_window=short_ma, long_window=long_ma, initial_capital=initial_capital)
    
    # 计算核心指标
    total_return = (result_df['Equity_Curve'].iloc[-1] / initial_capital) - 1
    annual_return = result_df['Strategy_Return'].mean() * 252
    vol = result_df['Strategy_Return'].std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / vol if vol != 0 else 0 # 假设无风险利率2%
    
    # 2. 展示关键指标 (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最终权益", f"${result_df['Equity_Curve'].iloc[-1]:,.2f}")
    col2.metric("总收益率", f"{total_return*100:.2f}%", delta_color="normal")
    col3.metric("年化夏普比率", f"{sharpe:.2f}")
    
    current_signal = "🟢 买入持有" if result_df['Signal'].iloc[-1] == 1 else "🔴 空仓观望"
    col4.metric("当前交易信号", current_signal)

    # 3. 交互式图表 (Plotly)
    st.subheader("📈 策略表现 vs 市场走势")
    
    fig = go.Figure()
    
    # K线/收盘价
    fig.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Close'], mode='lines', name='原油价格', line=dict(color='gray', width=1)))
    
    # 均线
    fig.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Short_MA'], mode='lines', name=f'MA{short_ma}', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Long_MA'], mode='lines', name=f'MA{long_ma}', line=dict(color='blue', width=1.5)))
    
    # 买卖信号标记
    buy_signals = result_df[(result_df['Signal'] == 1) & (result_df['Signal'].shift(1) == 0)]
    sell_signals = result_df[(result_df['Signal'] == 0) & (result_df['Signal'].shift(1) == 1)]
    
    fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='买入信号', marker=dict(color='green', symbol='triangle-up', size=10)))
    fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='卖出信号', marker=dict(color='red', symbol='triangle-down', size=10)))

    fig.update_layout(height=500, xaxis_title="日期", yaxis_title="价格 ($)")
    st.plotly_chart(fig, use_container_width=True)

    # 4. 资金曲线图
    st.subheader("💰 资金曲线 (Equity Curve)")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Equity_Curve'], mode='lines', name='策略净值', line=dict(color='green', width=2)))
    fig_equity.add_trace(go.Scatter(x=result_df['Date'], y=result_df['Close'] / result_df['Close'].iloc[0] * initial_capital, mode='lines', name='基准(持有不动)', line=dict(color='gray', dash='dash')))
    st.plotly_chart(fig_equity, use_container_width=True)

    # 5. 数据表格
    with st.expander("查看详细交易数据"):
        st.dataframe(result_df[['Date', 'Close', 'Short_MA', 'Long_MA', 'Signal', 'Equity_Curve']].tail(20))
