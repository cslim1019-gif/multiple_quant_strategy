import pandas as pd
import numpy as np

class VAABalanced:
    def __init__(self):
        """
        VAA (Vigilant Asset Allocation) Balanced 전략
        공격형 자산: 12개 (SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, GLD, DBC, HYG, LQD, TLT)
        안전 자산: 3개 (LQD, IEF, SHY)
        """
        self.offensive_tickers = [
            'SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 
            'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT'
        ]
        self.safe_tickers = ['LQD', 'IEF', 'SHY']
        self.all_tickers = list(set(self.offensive_tickers + self.safe_tickers))

    def calculate_momentum_score(self, prices_row, p1, p3, p6, p12):
        """
        가중 모멘텀 스코어: (12 * 1m) + (4 * 3m) + (2 * 6m) + (1 * 12m)
        """
        r1 = (prices_row / p1) - 1
        r3 = (prices_row / p3) - 1
        r6 = (prices_row / p6) - 1
        r12 = (prices_row / p12) - 1
        return (12 * r1) + (4 * r3) + (2 * r6) + (1 * r12)

    def calculate(self, data):
        # 1. 데이터 준비 및 필터링
        prices = data[self.all_tickers].copy().dropna()
        
        # 미래 참조 방지: 판단용 가격 데이터를 하루 뒤로 밀기
        # T일의 매매 결정은 T-1일까지 확정된 종가를 기준으로 판단
        logic_prices = prices.shift(1)
        
        # 룩백 기간 데이터 생성 (logic_prices를 기준으로 shift)
        p1 = logic_prices.shift(21)
        p3 = logic_prices.shift(63)
        p6 = logic_prices.shift(126)
        p12 = logic_prices.shift(252)
        
        # 실제 수익률 계산은 당일(T) 가격을 사용 (거래 집행)
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태 설정
        current_weights = pd.Series(0.0, index=self.all_tickers)
        current_weights['SHY'] = 1.0
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직] 12개월 웜업 구간 (T-1일 기준 252일 데이터가 있는지 확인)
            if pd.isna(p12.loc[current_date]).any():
                port_values.append(total_val)
                continue
            
            # [리밸런싱 로직] 월말 리밸런싱
            if last_month is not None and current_date.month != last_month:
                try:
                    # 판단 근거는 어제까지의 가격(logic_prices)
                    curr_p_logic = logic_prices.loc[current_date]
                    
                    scores = self.calculate_momentum_score(
                        curr_p_logic, 
                        p1.loc[current_date], 
                        p3.loc[current_date], 
                        p6.loc[current_date], 
                        p12.loc[current_date]
                    )
                    
                    off_scores = scores[self.offensive_tickers]
                    safe_scores = scores[self.safe_tickers]
                    
                    # 1. 하락 추세(스코어 < 0) 자산 수 측정
                    num_down = (off_scores < 0).sum()
                    
                    # 2. 안전 자산 비중 결정 (0~4개 단계적 대응)
                    if num_down == 0: safe_w = 0.0
                    elif num_down == 1: safe_w = 0.25
                    elif num_down == 2: safe_w = 0.50
                    elif num_down == 3: safe_w = 0.75
                    else: safe_w = 1.0
                    
                    off_w = 1.0 - safe_w
                    new_weights = pd.Series(0.0, index=self.all_tickers)
                    
                    # 3. 안전 자산 투자 (가장 높은 스코어 1개)
                    if safe_w > 0:
                        top_safe = safe_scores.idxmax()
                        new_weights[top_safe] = safe_w
                    
                    # 4. 공격 자산 투자 (상위 5개 균등 배분)
                    if off_w > 0:
                        top_5_off = off_scores.nlargest(5).index
                        new_weights[top_5_off] = off_w / 5.0
                    
                    current_weights = new_weights
                    
                except Exception:
                    pass
            
            # [수익률 반영] 어제 결정한 비중(current_weights)으로 오늘(daily_returns) 수익 창출
            daily_port_ret = (daily_returns.loc[current_date] * current_weights).sum()
            total_val *= (1 + daily_port_ret)
            port_values.append(total_val)
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)