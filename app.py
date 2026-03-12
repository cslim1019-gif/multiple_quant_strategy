import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import quantstats as qs
from datetime import datetime, timedelta
from scipy.optimize import minimize

# 투자 전략 클래스 (기존 임포트 유지)
from strategies.classic_6040 import Classic6040
from strategies.permanent_portfolio import PermanentPortfolio
from strategies.all_seasons_portfolio import AllSeasonsPortfolio
from strategies.all_weather_portfolio import AllWeatherPortfolio
from strategies.gtaa_5 import GTAA5
from strategies.dual_momentum import DualMomentum
from strategies.composite_dual_momentum import CompositeDualMomentum
from strategies.paa import PAA
from strategies.vaa_agressive import VAAAggressive
from strategies.vaa_balanced import VAABalanced
from strategies.daa import DAA
from strategies.laa import LAA
from strategies.baa import BAA
from strategies.baa_aggressive import BAAAggressive

# --- 1. 초기 데이터 엔진 (Pre-loading & Caching) ---

@st.cache_data(show_spinner=False)
def load_all_data(tickers):
    """모든 ETF 데이터를 한 번에 다운로드하고 동기화"""
    df_list = []
    start_date = "1990-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    p_bar = st.progress(0)
    p_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        p_text.text(f"📥 데이터 수집 중: {ticker} ({i+1}/{len(tickers)})")
        try:
            df = fdr.DataReader(ticker, start_date, end_date)['Adj Close']
            df.name = ticker
            df_list.append(df)
        except Exception as e:
            st.error(f"데이터 로드 실패 ({ticker}): {e}")
        p_bar.progress((i + 1) / len(tickers))
        
    data = pd.concat(df_list, axis=1).ffill()
    
    p_text.text("📊 실업률(UNRATE) 데이터 동기화 중...")
    try:
        unrate = fdr.DataReader('FRED:UNRATE', start_date, end_date)
        unrate.columns = ['UNRATE']
        data = data.join(unrate, how='left').ffill()
    except:
        data['UNRATE'] = np.nan
        
    final_data = data.dropna()
    p_bar.empty()
    p_text.empty()
    return final_data

# --- 2. 전략 수익률 수집 함수 ---
def get_individual_strategy_returns(data, strategy_name):
    strats = {
        "60/40 Portfolio": Classic6040,
        "Permanent Portfolio": PermanentPortfolio,
        "All Seasons Portfolio": AllSeasonsPortfolio,
        "All Weather Portfolio": AllWeatherPortfolio,
        "GTAA5 Strategy": GTAA5,
        "Dual Momentum": DualMomentum,
        "Composite Dual Momentum": CompositeDualMomentum,
        "PAA Strategy": PAA,
        "VAA Aggressive": VAAAggressive,
        "VAA Balanced": VAABalanced,
        "DAA Strategy": DAA,
        "LAA Strategy": LAA,
        "BAA Strategy": BAA,
        "BAA Aggressive": BAAAggressive
    }
    if strategy_name in strats:
        return strats[strategy_name]().calculate(data)
    return pd.Series(0, index=data.index)

# --- 3. 전체 전략 수익률 캐싱 함수 (성능 최적화의 핵심) ---
# 이 함수 덕분에 사이드바 위젯을 조작해도 스피너가 돌지 않음.
@st.cache_data(show_spinner="⚙️ 모든 전략의 수익률을 계산하고 있습니다 (최초 1회)...")
def get_all_strat_returns_cached(data, strategy_list):
    all_rets = {}
    for s_name in strategy_list:
        all_rets[s_name] = get_individual_strategy_returns(data, s_name)
    return pd.DataFrame(all_rets)

# --- 4. 수학적 최적화 함수 (Max Sharpe) ---
def calculate_optimal_weights(returns_df):
    num_assets = len(returns_df.columns)
    ann_ret = returns_df.mean() * 252
    ann_cov = returns_df.cov() * 252

    def objective(weights):
        p_ret = np.dot(weights, ann_ret)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(ann_cov, weights)))
        return -(p_ret / (p_vol + 1e-9))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    opt_results = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt_results.x

# --- 5. 메인 UI 및 로직 ---
def main():
    st.set_page_config(layout="wide", page_title="Strategy Mixer Pro")
    st.title("📊 Multi-Strategy Mixer Dashboard")
    
    all_tickers = ["SPY", "QQQ", "IWM", "VGK", "EWJ", "EEM", "VNQ", "GLD", "DBC", "HYG", "BIL", "VEA",
                   "LQD", "TLT", "IEF", "SHY", "AGG", "EFA", "REM", "VWO", "BND", "IWD", "TIP", "^GSPC"]
    
    # 데이터 로드
    raw_data = load_all_data(all_tickers)
    master_start_dt = raw_data.index.min().date()

    # 가용 전략 리스트
    available_strategies = [
        "60/40 Portfolio", "Permanent Portfolio", "All Seasons Portfolio", 
        "GTAA5 Strategy", "Dual Momentum", "Composite Dual Momentum", 
        "PAA Strategy", "VAA Aggressive", "VAA Balanced", "DAA Strategy", 
        "LAA Strategy", "BAA Strategy", "BAA Aggressive"
    ]

    # [핵심] 모든 전략의 결과를 미리 계산하여 캐싱함
    # 모든 로직은 이 DataFrame에서 필요한 열만 뽑아서 사용 (운영 효율화를 위해)
    all_returns_df_full = get_all_strat_returns_cached(raw_data, available_strategies)

    # 사이드바 1: 전역 설정
    st.sidebar.header("1. Global Settings")
    ui_start_date = st.sidebar.date_input("Start Date", datetime(2010, 1, 1))
    ui_end_date = st.sidebar.date_input("End Date", datetime.today())
    initial_seed = st.sidebar.number_input("Initial Capital ($)", value=10000)

    if ui_start_date < master_start_dt:
        st.error(f"⚠️ {master_start_dt} 이후로 설정해주세요.")
        return

    # 사이드바 2: 전략 선택
    st.sidebar.markdown("---")
    st.sidebar.header("2. Strategy Selection")

    selected_strats = st.sidebar.multiselect(
        "믹스할 전략 선택", 
        available_strategies, 
        default=["Permanent Portfolio"]
    )

    strat_weights = {}
    
    if selected_strats:
        st.sidebar.subheader("2-1. 비중 설정 (%)")
        
        # 균등 배분을 초기값으로 설정
        default_weight = 100 // len(selected_strats)
        
        for strat in selected_strats:
            # 직접 숫자 입력 (0~100 범위)
            weight_val = st.sidebar.number_input(
                f"{strat} 비중", 
                min_value=0, 
                max_value=100, 
                value=default_weight,
                step=1
            )
            strat_weights[strat] = weight_val / 100

        # 비중 합계 체크
        total_weight_pct = sum(strat_weights.values()) * 100
        st.sidebar.write(f"📊 현재 비중 합계: **{total_weight_pct:.1f}%**")
        
        if abs(sum(strat_weights.values()) - 1.0) > 0.001:
            st.sidebar.error("⚠️ 비중 합계를 100%에 맞춰주세요.")
            return
    else:
        st.sidebar.warning("전략을 먼저 선택해주세요.")
        return
    
    
    # 사이드바 3: 최적화 도구 (계산만 수행)
    st.sidebar.markdown("---")
    st.sidebar.header("3. Optimization Tool")
    st.sidebar.subheader("💡 사용 가이드")
    st.sidebar.info("""
    1️⃣ **전략 선택**             
    2번에서 전략을 고르세요.

    2️⃣ **최적 비율 계산**  
    아래 버튼을 눌러 Sharpe Ratio 최대 포트폴리오를 찾습니다.
    """)

    if st.sidebar.button("🚀 포트폴리오 최적 비중 계산하기"):
        # 이미 계산된 all_returns_df_full에서 선택된 전략만 뽑아서 날짜 필터링
        mask = (all_returns_df_full.index >= pd.Timestamp(ui_start_date)) & (all_returns_df_full.index <= pd.Timestamp(ui_end_date))
        selected_rets = all_returns_df_full[selected_strats].loc[mask]
        
        opt_w = calculate_optimal_weights(selected_rets)
        
        st.sidebar.success("✅ 추천 최적 비중 (Max Sharpe)")
        opt_df = pd.DataFrame({'Strategy': selected_strats, 'Optimal (%)': opt_w * 100})
        st.sidebar.table(opt_df.style.format({'Optimal (%)': '{:.1f}%'}))
        st.sidebar.caption("위 수치를 참고하여 슬라이더를 조정하세요.")

    # --- 6. 백테스트 계산 (캐시된 데이터 활용) ---
    # 선택된 비중으로 결합 수익률 계산
    # get_individual_strategy_returns를 반복 호출하지 않고 행렬 연산으로 처리하여 매우 빠름
    combined_returns_all = (all_returns_df_full[selected_strats] * pd.Series(strat_weights)).sum(axis=1)

    mask = (combined_returns_all.index >= pd.Timestamp(ui_start_date)) & \
           (combined_returns_all.index <= pd.Timestamp(ui_end_date))
    combined_returns = combined_returns_all.loc[mask]

    portfolio_value = (1 + combined_returns).cumprod() * initial_seed

    # --- 7. 대시보드 출력 ---
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Final Value", f"${portfolio_value.iloc[-1]:,.0f}")
    col2.metric("CAGR", f"{qs.stats.cagr(combined_returns)*100:.2f}%")
    col3.metric("Std Dev", f"{qs.stats.volatility(combined_returns)*100:.2f}%")
    col4.metric("Sharpe", f"{qs.stats.sharpe(combined_returns):.2f}")
    col5.metric("MDD", f"{qs.stats.max_drawdown(combined_returns)*100:.2f}%")
    win_months = (qs.stats.monthly_returns(combined_returns) > 0).sum().sum()
    col6.metric("Win Months", f"{win_months} mon")

    st.markdown("---")
    st.plotly_chart(go.Figure(data=[go.Scatter(x=portfolio_value.index, y=portfolio_value, name="Mixed Portfolio", line=dict(color='royalblue', width=2.5))],
                             layout=go.Layout(title="Mixed Portfolio Growth ($)", template="plotly_white", height=450)), use_container_width=True)

    yearly_ret = combined_returns.groupby(combined_returns.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    dd_series = qs.stats.to_drawdown_series(combined_returns) * 100
    fig_sub = make_subplots(rows=1, cols=2, subplot_titles=("Yearly Returns (%)", "Drawdown (%)"))
    fig_sub.add_trace(go.Bar(x=yearly_ret.index, y=yearly_ret, marker_color='#ef553b'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='gray')), row=1, col=2)
    st.plotly_chart(fig_sub, use_container_width=True)

    # --- 8. 전략 구성 참고를 위한 상관관계 히트맵 (전체 전략 대상) ---
    st.markdown("---")
    st.subheader("📊 전략 구성 참고를 위한 상관관계 히트맵")
    
    st.info("""
        💡 **전체 전략 분석 가이드**
        - 아래 히트맵은 시스템에 등록된 **모든 가용 전략** 간의 상관관계를 보여줍니다.
        - **파란색(0에 가깝거나 마이너스)** 영역에 있는 전략들을 조합할수록 리스크 분산 효과가 극대화됩니다.
    """)

    # 캐시된 전체 데이터를 날짜로 필터링하여 히트맵 생성
    mask_all = (all_returns_df_full.index >= pd.Timestamp(ui_start_date)) & \
               (all_returns_df_full.index <= pd.Timestamp(ui_end_date))
    full_corr_matrix = all_returns_df_full.loc[mask_all].corr()

    fig_full_corr = px.imshow(
        full_corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r', 
        range_color=[-1, 1],
        labels=dict(color="상관계수"),
        x=full_corr_matrix.columns,
        y=full_corr_matrix.columns
    )
    fig_full_corr.update_layout(
        template="plotly_white",
        height=700,
        margin=dict(t=50, l=50, r=50, b=50),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_full_corr, use_container_width=True)

if __name__ == "__main__":
    main()
