import pandas as pd
import numpy as np

class DualMomentum:
    def __init__(self):
        """
        듀얼 모멘텀 전략: 상대적 모멘텀 + 절대적 모멘텀 결합
        대상: SPY(미국 주식), EFA(선진국 주식), BIL(현금), AGG(채권)
        """
        self.tickers = ['SPY', 'EFA', 'BIL', 'AGG']

    def calculate(self, data):
        # 1. 데이터 준비
        prices = data[self.tickers].copy().dropna()
        
        # [핵심 수정] 미래 참조 방지: 모든 판단용 지표는 어제 종가(T-1)를 기준으로 판단
        logic_prices = prices.shift(1)
        
        # 최근 12개월(252 거래일) 수익률 계산 (판단용 지표)
        lookback = 252
        returns_12m = logic_prices.pct_change(lookback)
        
        # 월말 시점의 모멘텀 데이터만 리샘플링 (판단용)
        monthly_returns_12m = returns_12m.resample('ME').last()
        
        # 실제 매매 수익률은 당일(T)의 가격 변동을 사용
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태 설정
        current_asset = 'AGG' 
        total_val = 1.0
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            current_month = current_date.month
            
            # [방어 로직] 12개월 수익률 데이터(T-1일 기준)가 없는 웜업 구간은 자산 가치 유지
            if pd.isna(returns_12m.loc[current_date]).any():
                port_values.append(total_val)
                last_month = current_month
                continue
            
            # [리밸런싱 로직] 월이 바뀌면 모멘텀 측정 후 자산 교체
            if last_month is not None and current_month != last_month:
                try:
                    # 어제까지 확정된 가장 최신의 월간 모멘텀 데이터를 가져오기
                    # 월초 첫 거래일(T)에 이 코드가 실행되면, iloc[-1]은 자동으로 '지난달 말(T-1 근방)' 데이터가 됨
                    mom_data = monthly_returns_12m.loc[:current_date].iloc[-1]
                    
                    spy_12m = mom_data['SPY']
                    efa_12m = mom_data['EFA']
                    bil_12m = mom_data['BIL']
                    
                    # 1. 절대적 모멘텀 확인: 주식(SPY)이 현금(BIL)보다 나은가?
                    if spy_12m > bil_12m:
                        # 2. 상대적 모멘텀 확인: 미국(SPY) vs 선진국(EFA) 중 승자 선택
                        current_asset = 'SPY' if spy_12m > efa_12m else 'EFA'
                    else:
                        # 주식 시장이 좋지 않으면 안전자산(AGG) 보유
                        current_asset = 'AGG'
                except (IndexError, KeyError):
                    current_asset = 'AGG'
            
            # [수익률 반영] 어제 내린 결정(current_asset)으로 오늘(daily_returns) 수익률을 얻음
            asset_return = daily_returns.loc[current_date, current_asset]
            total_val *= (1 + asset_return)
            port_values.append(total_val)
            last_month = current_month
            
        # 결과 시계열 반환
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)