import pandas as pd
import numpy as np

class DAA:
    def __init__(self):
        """
        DAA (Dynamic Asset Allocation) 전략
        공격 자산: 12개 (SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, GLD, DBC, HYG, LQD, TLT)
        안전 자산: 3개 (LQD, IEF, SHY)
        카나리아 자산: 2개 (VWO, BND)
        """
        self.offensive_tickers = [
            'SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 
            'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT'
        ]
        self.safe_tickers = ['LQD', 'IEF', 'SHY']
        self.canary_tickers = ['VWO', 'BND']
        self.all_tickers = list(set(self.offensive_tickers + self.safe_tickers + self.canary_tickers))

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
        # 1. 데이터 준비 (해당 전략에 필요한 티커만 추출)
        prices = data[self.all_tickers].copy().dropna()
        
        # [핵심 수정] 미래 참조 방지: 판단용 가격 데이터를 하루 뒤로 밀기
        # T일의 의사결정은 T-1일까지 확정된 가격(어제 종가)을 기준으로 판단
        logic_prices = prices.shift(1)
        
        # 룩백 기간 데이터 생성 (logic_prices 기준)
        p1 = logic_prices.shift(21)
        p3 = logic_prices.shift(63)
        p6 = logic_prices.shift(126)
        p12 = logic_prices.shift(252)
        
        # 실제 거래 수익률은 당일(T) 가격 변동을 사용
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태 설정: 100% 안전 자산(SHY)으로 시작
        current_weights = pd.Series(0.0, index=self.all_tickers)
        current_weights['SHY'] = 1.0
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직] 12개월(T-1일 기준) 데이터가 쌓이기 전까지는 판단 보류
            if pd.isna(p12.loc[current_date]).any():
                port_values.append(total_val)
                last_month = current_date.month
                continue
            
            # [리밸런싱 로직] 월말 리밸런싱
            if last_month is not None and current_date.month != last_month:
                try:
                    # 모든 신호(Signal)는 어제까지의 가격(logic_prices)으로 계산
                    curr_p_logic = logic_prices.loc[current_date]
                    
                    scores = self.calculate_momentum_score(
                        curr_p_logic, 
                        p1.loc[current_date], p3.loc[current_date], 
                        p6.loc[current_date], p12.loc[current_date]
                    )
                    
                    # 1. 카나리아 자산군 스코어 확인 (VWO, BND)
                    vwo_score = scores['VWO']
                    bnd_score = scores['BND']
                    
                    # 2. 공격형 자산 비중(Offensive Weight) 결정
                    if vwo_score >= 0 and bnd_score >= 0:
                        off_w = 1.0
                    elif vwo_score < 0 and bnd_score < 0:
                        off_w = 0.0
                    else:
                        off_w = 0.5
                    
                    safe_w = 1.0 - off_w
                    new_weights = pd.Series(0.0, index=self.all_tickers)
                    
                    # 3. 안전 자산 배분 (가장 모멘텀 스코어가 높은 1개)
                    if safe_w > 0:
                        top_safe = scores[self.safe_tickers].idxmax()
                        new_weights[top_safe] = safe_w
                    
                    # 4. 공격 자산 배분 (가중 모멘텀 상위 6개 자산에 균등 투자)
                    if off_w > 0:
                        top_6_off = scores[self.offensive_tickers].nlargest(6).index
                        new_weights[top_6_off] = off_w / 6.0
                    
                    current_weights = new_weights
                    
                except Exception:
                    pass
            
            # [수익률 반영] 어제 내린 결정으로 오늘 장의 수익률을 입힘
            daily_port_ret = (daily_returns.loc[current_date] * current_weights).sum()
            total_val *= (1 + daily_port_ret)
            port_values.append(total_val)
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)