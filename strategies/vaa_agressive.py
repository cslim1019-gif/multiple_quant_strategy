import pandas as pd
import numpy as np

class VAAAggressive:
    def __init__(self):
        """
        VAA (Vigilant Asset Allocation) Aggressive 전략
        공격 자산: SPY, EFA, EEM, AGG
        안전 자산: LQD, IEF, SHY
        """
        self.offensive_tickers = ['SPY', 'EFA', 'EEM', 'AGG']
        self.safe_tickers = ['LQD', 'IEF', 'SHY']
        self.all_tickers = self.offensive_tickers + self.safe_tickers

    def calculate_momentum_score(self, prices_row, prices_1m, prices_3m, prices_6m, prices_12m):
        """
        가중 모멘텀 스코어 계산: (12 * 1개월 수익률) + (4 * 3개월) + (2 * 6개월) + (1 * 12개월)
        """
        ret_1m = (prices_row / prices_1m) - 1
        ret_3m = (prices_row / prices_3m) - 1
        ret_6m = (prices_row / prices_6m) - 1
        ret_12m = (prices_row / prices_12m) - 1
        
        return (12 * ret_1m) + (4 * ret_3m) + (2 * ret_6m) + (1 * ret_12m)
    
    def calculate(self, data):
        # 1. 데이터 준비
        prices = data[self.all_tickers].copy().dropna()
        
        # [핵심 수정] 미래 참조 방지: 판단에 사용하는 모든 가격 데이터를 하루씩 뒤로 밀기
        # 즉, T일의 의사결정은 T-1일까지의 가격 데이터만을 사용
        logic_prices = prices.shift(1)
        
        # 가중 모멘텀 계산을 위한 과거 가격들 (판단 기준일로부터 각각 1, 3, 6, 12개월 전)
        prices_1m = logic_prices.shift(21)
        prices_3m = logic_prices.shift(63)
        prices_6m = logic_prices.shift(126)
        prices_12m = logic_prices.shift(252)
        
        # 실제 수익률 계산은 오늘(T) 가격을 사용합니다. (거래 집행)
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        current_asset = 'SHY' # 초기값
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # 웜업 구간 방어: 12개월 룩백 데이터(p12)가 준비될 때까지 대기
            # logic_prices 기반이므로 상장 후 약 1년 + 1일 뒤부터 투자가 시작
            if pd.isna(prices_12m.loc[current_date]).any():
                port_values.append(total_val)
                continue

            # [리밸런싱 로직] 월말 리밸런싱 수행
            if last_month is not None and current_date.month != last_month:
                try:
                    # [수정] 어제까지의 가격(logic_prices)과 과거 가격들로 스코어 계산
                    curr_p_logic = logic_prices.loc[current_date]
                    p1 = prices_1m.loc[current_date]
                    p3 = prices_3m.loc[current_date]
                    p6 = prices_6m.loc[current_date]
                    p12 = prices_12m.loc[current_date]
                    
                    scores = self.calculate_momentum_score(curr_p_logic, p1, p3, p6, p12)
                    
                    if scores.isna().any():
                        current_asset = 'SHY'
                    else:
                        off_scores = scores[self.offensive_tickers]
                        safe_scores = scores[self.safe_tickers]
                        
                        # 공격 자산 4개 모두 모멘텀 스코어 0 이상일 때만 공격
                        if (off_scores >= 0).all():
                            current_asset = off_scores.idxmax()
                        else:
                            # 하나라도 0 미만이면 가장 강한 안전 자산으로 대피
                            current_asset = safe_scores.idxmax()
                            
                except Exception:
                    current_asset = 'SHY'
            
            # [수익률 반영] 어제 내린 결정(current_asset)으로 오늘(daily_returns) 수익을 얻음
            total_val *= (1 + daily_returns.loc[current_date, current_asset])
            port_values.append(total_val)
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)

    