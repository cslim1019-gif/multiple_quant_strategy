import pandas as pd
import numpy as np

class AllSeasonsPortfolio:
    def __init__(self, weights={'SPY': 0.30, 'IEF': 0.15, 'TLT': 0.40, 'GLD': 0.075, 'DBC': 0.075}):
        """
        사계절 포트폴리오 전략: 
        주식 30%, 중기채 15%, 장기채 40%, 금 7.5%, 원자재 7.5%
        """
        self.weights = weights
        self.tickers = list(weights.keys())

    def calculate(self, data):
        """
        연 1회 리밸런싱을 적용한 사계절 포트폴리오 수익률 계산
        """
        # 1. 데이터 추출 및 결측치 처리
        prices = data[self.tickers].copy().dropna()
        daily_returns = prices.pct_change().fillna(0)
        
        port_values = []
        # 초기 자산 가치를 비중대로 설정
        current_asset_values = np.array([self.weights[t] for t in self.tickers])
        
        last_year = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [리밸런싱 로직] 연도가 바뀌면 타겟 비중으로 자산 재배분
            if last_year is not None and current_date.year != last_year:
                total_portfolio_value = np.sum(current_asset_values)
                current_asset_values = total_portfolio_value * np.array([self.weights[t] for t in self.tickers])
            
            # 각 자산의 일일 수익률 반영
            current_asset_values = current_asset_values * (1 + daily_returns.iloc[i].values)
            
            # 포트폴리오 총 가치 기록
            port_values.append(np.sum(current_asset_values))
            last_year = current_date.year
            
        # 2. 결과 시계열 반환
        portfolio_series = pd.Series(port_values, index=daily_returns.index)
        portfolio_returns = portfolio_series.pct_change().fillna(0)
        
        return portfolio_returns