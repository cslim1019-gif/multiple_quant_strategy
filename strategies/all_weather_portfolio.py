import pandas as pd
import numpy as np

class AllWeatherPortfolio:
    def __init__(self, weights=None):
        """
        올웨더 포트폴리오 전략 (9개 자산군 구성)
        """
        if weights is None:
            self.weights = {
                'SPY': 0.12, 'EFA': 0.12, 'EEM': 0.12,  # 주식 (36%)
                'DBC': 0.07, 'GLD': 0.07,              # 원자재/금 (14%)
                'EDV': 0.18, 'LTPZ': 0.18,             # 장기국채/물가연동채 (36%)
                'LQD': 0.07, 'EMLC': 0.07              # 회사채/신흥국채권 (14%)
            }
        else:
            self.weights = weights
            
        self.tickers = list(self.weights.keys())

    def calculate(self, data):
        """
        연 1회 리밸런싱을 적용한 올웨더 포트폴리오 수익률 계산
        """
        # 1. 데이터 추출 및 결측치 처리 (FDR로 받아온 전체 데이터 중 필요한 티커만)
        # 9개 자산이 모두 존재하는 시점부터 계산하기 위해 dropna() 적용
        prices = data[self.tickers].copy().dropna()
        daily_returns = prices.pct_change().fillna(0)
        
        port_values = []
        # 초기 자산 가치 설정 (비중 합계 = 1.0)
        current_asset_values = np.array([self.weights[t] for t in self.tickers])
        
        last_year = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [리밸런싱 로직] 연도가 바뀌면 타겟 비중으로 리셋
            if last_year is not None and current_date.year != last_year:
                total_portfolio_value = np.sum(current_asset_values)
                current_asset_values = total_portfolio_value * np.array([self.weights[t] for t in self.tickers])
            
            # 각 자산별 수익률 반영
            current_asset_values = current_asset_values * (1 + daily_returns.iloc[i].values)
            
            # 전체 포트폴리오 가치 합산 기록
            port_values.append(np.sum(current_asset_values))
            last_year = current_date.year
            
        # 2. 결과 시계열 반환
        portfolio_series = pd.Series(port_values, index=daily_returns.index)
        portfolio_returns = portfolio_series.pct_change().fillna(0)
        
        return portfolio_returns