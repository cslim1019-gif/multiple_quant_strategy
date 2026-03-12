import pandas as pd
import numpy as np

class Classic6040:
    def __init__(self, weights={'SPY': 0.6, 'IEF': 0.4}):
        self.weights = weights
        self.tickers = list(weights.keys())

    def calculate(self, data):
        """
        연 1회 리밸런싱을 적용한 60/40 포트폴리오 수익률 계산
        """
        # 1. 필요한 자산만 추출 및 결측치 처리
        prices = data[self.tickers].copy().dropna()
        daily_returns = prices.pct_change().fillna(0)
        
        # 2. 리밸런싱 로직 시뮬레이션
        # 포트폴리오의 일별 가치를 저장할 리스트
        port_values = []
        # 각 자산의 현재 가치 (초기 비중으로 시작)
        current_asset_values = np.array(list(self.weights.values()))
        
        last_year = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [리밸런싱 체크] 연도가 바뀌면 비중을 다시 초기화
            if last_year is not None and current_date.year != last_year:
                total_portfolio_value = np.sum(current_asset_values)
                current_asset_values = total_portfolio_value * np.array(list(self.weights.values()))
            
            # 자산별 수익률 반영 (가치 업데이트)
            current_asset_values = current_asset_values * (1 + daily_returns.iloc[i].values)
            
            # 전체 포트폴리오 가치 기록
            port_values.append(np.sum(current_asset_values))
            last_year = current_date.year
            
        # 3. 가치 변화를 다시 수익률 시계열로 변환
        portfolio_series = pd.Series(port_values, index=daily_returns.index)
        portfolio_returns = portfolio_series.pct_change().fillna(0)
        
        return portfolio_returns