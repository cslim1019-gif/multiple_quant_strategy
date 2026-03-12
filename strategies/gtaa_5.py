import pandas as pd
import numpy as np

class GTAA5:
    def __init__(self, weights={'SPY': 0.2, 'EFA': 0.2, 'IEF': 0.2, 'DBC': 0.2, 'VNQ': 0.2}):
        """
        GTAA5 전략: 5개 자산군에 대해 10개월 이평선 돌파 시 20% 투자, 미달 시 현금화
        """
        self.weights = weights
        self.tickers = list(weights.keys())
        self.cash_ticker = 'BIL' # 현금 대용 자산

    def calculate(self, data):
        """
        연 1회 리밸런싱 및 추세 추종 로직 적용
        """
        # 1. 데이터 준비 (5개 자산 + 현금 자산)
        all_required = self.tickers + [self.cash_ticker]
        prices = data[all_required].copy().dropna()
        
        # 10개월 이동평균 계산 (약 200거래일)
        # 웜업 기간(초반 199행)은 NaN이 발생합니다.
        sma_10m = prices[self.tickers].rolling(window=200).mean().shift(1)
        
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태 설정
        current_asset_values = np.zeros(len(all_required))
        total_val = 1.0
        last_year = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직] 200일 이평선 데이터가 준비되지 않은 웜업 구간은 계산 제외
            if pd.isna(sma_10m.loc[current_date]).any():
                port_values.append(total_val)
                last_year = current_date.year # 연도 기록은 유지
                continue

            # [리밸런싱 로직] 연도가 바뀌면 추세 확인 후 비중 재배분
            # 혹은 웜업이 끝나고 처음으로 투자를 시작하는 시점(last_year 설정 이후 첫 구동)
            if last_year is not None and current_date.year != last_year:
                total_val = np.sum(current_asset_values)
                new_values = np.zeros(len(all_required))
                
                for idx, ticker in enumerate(self.tickers):
                    current_price = prices.loc[current_date, ticker]
                    current_sma = sma_10m.loc[current_date, ticker]
                    
                    # 추세 확인: 가격 > 10개월 이평선
                    if current_price > current_sma:
                        new_values[idx] = total_val * self.weights[ticker]
                    else:
                        # 추세 미달 시 해당 비중만큼 현금(BIL)에 할당
                        new_values[-1] += total_val * self.weights[ticker]
                
                current_asset_values = new_values
            
            # [초기화] 웜업이 끝나고 처음으로 자산을 배분하는 시점 처리
            elif np.sum(current_asset_values) == 0:
                # 웜업이 끝난 직후, 일단 현금(BIL)으로 전액 보유 시작
                current_asset_values[-1] = total_val 
                
            # 일일 수익률 반영
            # current_asset_values 배열에 각 자산의 당일 수익률을 곱함
            current_asset_values = current_asset_values * (1 + daily_returns.iloc[i].values)
            
            total_val = np.sum(current_asset_values)
            port_values.append(total_val)
            last_year = current_date.year
            
        # 2. 결과 시계열 반환
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)