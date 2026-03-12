import pandas as pd
import numpy as np

class BAAAggressive:
    def __init__(self):
        """
        BAA (Bold Asset Allocation) Aggressive 전략
        공격 자산: QQQ, VWO, VEA, BND (4개)
        수비 자산: TIP, DBC, BIL, IEF, TLT, LQD, BND (7개)
        카나리아 자산: SPY, VWO, VEA, BND (4개)
        """
        self.offensive_tickers = ['QQQ', 'VWO', 'VEA', 'BND']
        self.safe_tickers = ['TIP', 'DBC', 'BIL', 'IEF', 'TLT', 'LQD', 'BND']
        self.canary_tickers = ['SPY', 'VWO', 'VEA', 'BND']
        self.all_tickers = list(set(self.offensive_tickers + self.safe_tickers + self.canary_tickers))

    def calculate_canary_score(self, prices_row, p1, p3, p6, p12):
        """
        카나리아용 가중 모멘텀 스코어: (12 * 1m) + (4 * 3m) + (2 * 6m) + (1 * 12m)
        """
        r1 = (prices_row / p1) - 1
        r3 = (prices_row / p3) - 1
        r6 = (prices_row / p6) - 1
        r12 = (prices_row / p12) - 1
        return (12 * r1) + (4 * r3) + (2 * r6) + (1 * r12)

    def calculate(self, data):
        # 1. 데이터 준비 및 미래 참조 방지 (어제 종가 기준 판단)
        prices = data[self.all_tickers].copy().dropna()
        logic_prices = prices.shift(1)
        
        # 월간 데이터 리샘플링 (SMA12 계산용)
        monthly_logic_prices = logic_prices.resample('ME').last()
        
        # 카나리아 판단용 일별 룩백 데이터 (logic_prices 기준)
        p1 = logic_prices.shift(21)
        p3 = logic_prices.shift(63)
        p6 = logic_prices.shift(126)
        p12 = logic_prices.shift(252)
        
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태: 현금 100%
        current_weights = pd.Series(0.0, index=self.all_tickers)
        current_weights['BIL'] = 1.0
        
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직] SMA12 계산을 위해 13개월치 월말 데이터가 필요함
            if pd.isna(p12.loc[current_date]).any() or \
               len(monthly_logic_prices.loc[:current_date]) < 13:
                port_values.append(total_val)
                continue
            
            # [리밸런싱 로직] 월말 리밸런싱
            if last_month is not None and current_date.month != last_month:
                try:
                    curr_p_logic = logic_prices.loc[current_date]
                    
                    # 1. 카나리아 모멘텀 체크 (하나라도 0 미만이면 수비 모드)
                    canary_scores = self.calculate_canary_score(
                        curr_p_logic[self.canary_tickers],
                        p1.loc[current_date, self.canary_tickers],
                        p3.loc[current_date, self.canary_tickers],
                        p6.loc[current_date, self.canary_tickers],
                        p12.loc[current_date, self.canary_tickers]
                    )
                    
                    is_offensive = (canary_scores >= 0).all()
                    
                    # 2. SMA12 계산을 위한 월말 데이터 추출 (현재 포함 최근 13개월)
                    recent_monthly = monthly_logic_prices.loc[:current_date].iloc[-13:]
                    # 공식: 현재가 / (현재 포함 13개월 평균) - 1
                    sma12_values = (recent_monthly.iloc[-1] / recent_monthly.mean()) - 1
                    
                    new_weights = pd.Series(0.0, index=self.all_tickers)
                    
                    if is_offensive:
                        # [공격 모드] SMA12 기준 Top 1 집중 투자
                        off_sma = sma12_values[self.offensive_tickers]
                        top_1_off = off_sma.idxmax()
                        new_weights[top_1_off] = 1.0
                    else:
                        # [수비 모드] SMA12 기준 Top 3 + BIL 보호 로직
                        safe_sma = sma12_values[self.safe_tickers]
                        top_3_safe = safe_sma.nlargest(3).index
                        bil_sma = sma12_values['BIL']
                        
                        for ticker in top_3_safe:
                            if safe_sma[ticker] < bil_sma:
                                new_weights['BIL'] += 1.0 / 3.0
                            else:
                                new_weights[ticker] += 1.0 / 3.0
                                
                    current_weights = new_weights
                except Exception:
                    pass
            
            # 일일 수익률 반영 (어제 결정한 비중으로 오늘 수익 확정)
            daily_port_ret = (daily_returns.loc[current_date] * current_weights).sum()
            total_val *= (1 + daily_port_ret)
            port_values.append(total_val)
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)