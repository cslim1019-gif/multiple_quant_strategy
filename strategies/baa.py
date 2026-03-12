import pandas as pd
import numpy as np

class BAA:
    def __init__(self):
        """
        BAA (Bold Asset Allocation) Balanced 전략
        공격 자산: SPY, QQQ, IWM, VGK, EWJ, VWO, VNQ, DBC, GLD, TLT, HYG, LQD (12개)
        안전 자산: TIP, DBC, BIL, IEF, TLT, LQD, BND (7개)
        카나리아 자산: SPY, VWO, VEA, BND (4개)
        """
        self.offensive_tickers = ['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'VWO', 'VNQ', 'DBC', 'GLD', 'TLT', 'HYG', 'LQD']
        self.safe_tickers = ['TIP', 'DBC', 'BIL', 'IEF', 'TLT', 'LQD', 'BND']
        self.canary_tickers = ['SPY', 'VWO', 'VEA', 'BND']
        self.all_tickers = list(set(self.offensive_tickers + self.safe_tickers + self.canary_tickers))

    def calculate_canary_score(self, prices_row, p1, p3, p6, p12):
        """
        카나리아 자산군용 가중 모멘텀 스코어: (12 * 1m) + (4 * 3m) + (2 * 6m) + (1 * 12m)
        """
        r1 = (prices_row / p1) - 1
        r3 = (prices_row / p3) - 1
        r6 = (prices_row / p6) - 1
        r12 = (prices_row / p12) - 1
        return (12 * r1) + (4 * r3) + (2 * r6) + (1 * r12)

    def calculate_sma12(self, monthly_prices):
        """
        SMA12 = 현재가 / (현재가 + 지난 12개월 가격의 평균) - 1
        총 13개의 월말 가격 데이터가 필요합니다.
        """
        # 현재 행(t)을 포함한 최근 13개월 데이터의 평균 계산
        rolling_mean_13 = monthly_prices.rolling(window=13).mean()
        return (monthly_prices / rolling_mean_13) - 1

    def calculate(self, data):
        # 1. 데이터 준비 및 미래 참조 방지
        prices = data[self.all_tickers].copy().dropna()
        logic_prices = prices.shift(1) # 어제 종가 기준 판단
        
        # 월간 데이터 리샘플링 (SMA12 및 월간 모멘텀용)
        monthly_logic_prices = logic_prices.resample('ME').last()
        
        # 카나리아 판단용 룩백 데이터 (daily logic_prices 기준)
        p1 = logic_prices.shift(21)
        p3 = logic_prices.shift(63)
        p6 = logic_prices.shift(126)
        p12 = logic_prices.shift(252)
        
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        current_weights = pd.Series(0.0, index=self.all_tickers)
        current_weights['BIL'] = 1.0 # 초기값은 현금
        
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직] 12개월 웜업 (SMA12를 위해 13개월 데이터가 쌓일 때까지 대기)
            # monthly_logic_prices가 최소 13개 이상 확보되어야 함
            if pd.isna(p12.loc[current_date]).any() or \
               len(monthly_logic_prices.loc[:current_date]) < 13:
                port_values.append(total_val)
                continue
            
            # [리밸런싱 로직] 월말 리밸런싱
            if last_month is not None and current_date.month != last_month:
                try:
                    curr_p_logic = logic_prices.loc[current_date]
                    
                    # 1. 카나리아 자산군 모멘텀 체크
                    canary_scores = self.calculate_canary_score(
                        curr_p_logic[self.canary_tickers],
                        p1.loc[current_date, self.canary_tickers],
                        p3.loc[current_date, self.canary_tickers],
                        p6.loc[current_date, self.canary_tickers],
                        p12.loc[current_date, self.canary_tickers]
                    )
                    
                    # 2. 투자 모드 결정: 하나라도 마이너스면 수비 모드
                    is_offensive = (canary_scores >= 0).all()
                    
                    new_weights = pd.Series(0.0, index=self.all_tickers)
                    
                    # 최근 13개월 월말 가격 데이터 추출 (SMA12용)
                    recent_monthly = monthly_logic_prices.loc[:current_date].iloc[-13:]
                    # SMA12 공식: 현재가 / 13개월 평균 - 1
                    sma12_values = (recent_monthly.iloc[-1] / recent_monthly.mean()) - 1
                    
                    if is_offensive:
                        # [공격 모드] SMA12 기준 상위 6개 동일 비중
                        off_sma = sma12_values[self.offensive_tickers]
                        top_6_off = off_sma.nlargest(6).index
                        new_weights[top_6_off] = 1.0 / 6.0
                    else:
                        # [수비 모드] SMA12 기준 상위 3개 선정 + BIL 보호 로직
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
            
            # 일일 수익률 반영
            daily_port_ret = (daily_returns.loc[current_date] * current_weights).sum()
            total_val *= (1 + daily_port_ret)
            port_values.append(total_val)
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)