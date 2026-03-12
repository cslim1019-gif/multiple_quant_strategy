import pandas as pd
import numpy as np

class PermanentPortfolio:
    def __init__(self, weights={'SPY': 0.25, 'TLT': 0.25, 'GLD': 0.25, 'BIL': 0.25}):
        """
        영구 포트폴리오 전략: 주식, 채권, 금, 현금에 25%씩 배분
        """
        self.weights = weights
        self.tickers = list(weights.keys())

    def calculate(self, data):
        """
        연 1회 리밸런싱을 적용한 영구 포트폴리오 수익률 계산
        """
        # 1. 필요한 자산 데이터만 추출 및 결측치 처리
        # 데이터가 2007-04-11부터 있는지 확인하기 위해 dropna() 사용
        prices = data[self.tickers].copy().dropna()
        daily_returns = prices.pct_change().fillna(0)
        
        port_values = []
        # 초기 자산을 1로 설정하고 비중대로 나눔 (0.25씩)
        current_asset_values = np.array([self.weights[t] for t in self.tickers])
        
        last_year = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [리밸런싱 로직] 연도가 바뀌면 (매년 첫 거래일) 비중을 다시 25%씩으로 리셋
            if last_year is not None and current_date.year != last_year:
                total_portfolio_value = np.sum(current_asset_values)
                # 현재 총 자산을 다시 4등분함
                current_asset_values = total_portfolio_value * np.array([self.weights[t] for t in self.tickers])
            
            # 자산별 일일 수익률 반영
            current_asset_values = current_asset_values * (1 + daily_returns.iloc[i].values)
            
            # 전체 포트폴리오 가치 합계 기록
            port_values.append(np.sum(current_asset_values))
            last_year = current_date.year
            
        # 2. 가치 변화를 다시 수익률 시계열(Series)로 변환
        portfolio_series = pd.Series(port_values, index=daily_returns.index)
        portfolio_returns = portfolio_series.pct_change().fillna(0)
        
        return portfolio_returns