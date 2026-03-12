import pandas as pd
import numpy as np

class PAA:
    def __init__(self):
        """
        PAA (Protecting Asset Allocation) 전략
        공격 자산: SPY, QQQ, IWM, VGK, EWJ, EEM, VNQ, GLD, DBC, HYG, LQD, TLT (12개)
        안전 자산: IEF
        """
        self.offensive_tickers = [
            'SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM', 
            'VNQ', 'GLD', 'DBC', 'HYG', 'LQD', 'TLT'
        ]
        self.safe_ticker = 'IEF'
        self.tickers = list(set(self.offensive_tickers + [self.safe_ticker]))

    def calculate(self, data):
        # 1. 데이터 준비
        prices = data[self.tickers].copy().dropna()
        
        # 12개월(약 252거래일) 단순 이동평균(SMA) 계산
        # 이 시점에서 첫 251행은 NaN이 됩니다.
        sma_12m = prices[self.offensive_tickers].rolling(window=252).mean().shift(1)
        
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태: 100% 현금(안전자산)으로 시작
        current_weights = pd.Series(0.0, index=self.tickers)
        current_weights[self.safe_ticker] = 1.0
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직 추가] 12개월 SMA 데이터가 준비되지 않은 웜업 구간은 계산 제외
            # 이 구간에서는 투자 결정을 내리지 않고 초기 자산 가치를 유지
            if pd.isna(sma_12m.loc[current_date]).any():
                port_values.append(total_val)
                last_month = current_date.month # 월 변경 감지를 위해 날짜는 기록
                continue
            
            # [리밸런싱 로직] 월말 리밸런싱 수행
            if last_month is not None and current_date.month != last_month:
                try:
                    # 리밸런싱 시점의 가격 및 SMA 데이터
                    curr_prices = prices.loc[current_date]
                    curr_sma = sma_12m.loc[current_date]
                    
                    # 1. 하락 추세 자산 수 측정 (현재가 < 12개월 이평선)
                    is_down = curr_prices[self.offensive_tickers] < curr_sma
                    num_down = is_down.sum()
                    
                    # 2. 안전 자산(IEF) 비중 결정
                    safe_weight = min(num_down / 6.0, 1.0)
                    offensive_total_weight = 1.0 - safe_weight
                    
                    # 비중 초기화
                    new_weights = pd.Series(0.0, index=self.tickers)
                    new_weights[self.safe_ticker] = safe_weight
                    
                    # 3. 공격 자산 배분 (상위 6개 자산 선정)
                    if offensive_total_weight > 0:
                        # 모멘텀 스코어: (현재가 / 12개월 이평선) - 1
                        mom_scores = (curr_prices[self.offensive_tickers] / curr_sma) - 1
                        top_6 = mom_scores.nlargest(6).index
                        new_weights[top_6] = offensive_total_weight / 6.0
                    
                    current_weights = new_weights
                except Exception:
                    pass
            
            # 일일 포트폴리오 수익률 계산
            daily_port_return = (daily_returns.loc[current_date] * current_weights).sum()
            total_val *= (1 + daily_port_return)
            port_values.append(total_val)
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)