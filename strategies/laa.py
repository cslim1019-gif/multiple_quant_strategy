import pandas as pd
import numpy as np

class LAA:
    def __init__(self):
        """
        LAA (Lethargic Asset Allocation) 전략
        고정 자산: IWD, GLD, IEF (각 25%) - 연 1회 리밸런싱
        타이밍 자산: QQQ or SHY (25%) - 월 1회 리밸런싱
        """
        self.fixed_assets = ['IWD', 'GLD', 'IEF']
        self.timing_assets = ['QQQ', 'SHY']
        self.sp500_ticker = '^GSPC'
        self.unrate_col = 'UNRATE'

    def calculate(self, data):
        # 1. 데이터 준비
        # 주가 데이터와 UNRATE 컬럼을 포함한 데이터 필터링
        prices = data.copy().dropna(subset=self.fixed_assets + self.timing_assets + [self.sp500_ticker])
        
        # S&P 500 200일 이동평균 (약 200거래일)
        sma_200 = prices[self.sp500_ticker].rolling(window=200).mean().shift(1)
        # 실업률 12개월 이동평균 (약 252거래일 기준)
        unrate_sma_12m = prices[self.unrate_col].rolling(window=252).mean().shift(1)
        
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태 설정
        total_val = 1.0
        # 각 자산별 가치 (IWD, GLD, IEF, Timing Asset 각 25%)
        asset_values = np.array([0.25, 0.25, 0.25, 0.25])
        current_timing_asset = 'QQQ'
        
        last_year = None
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            
            # [방어 로직] 200일 주가 이평선 혹은 12개월 실업률 이평선이 준비되지 않은 구간은 스킵
            if pd.isna(sma_200.loc[current_date]) or pd.isna(unrate_sma_12m.loc[current_date]):
                port_values.append(total_val)
                last_month = current_date.month
                last_year = current_date.year
                continue

            # --- 리밸런싱 로직 ---
            # 1. 월간 리밸런싱 (타이밍 자산 결정)
            if last_month is not None and current_date.month != last_month:
                try:
                    curr_sp500 = prices.loc[current_date, self.sp500_ticker]
                    curr_sma200 = sma_200.loc[current_date]
                    curr_unrate = prices.loc[current_date, self.unrate_col]
                    curr_unrate_sma = unrate_sma_12m.loc[current_date]
                    
                    # 타이밍 조건: (S&P500 < 200일 이평) AND (실업률 > 12개월 이평) 일 때만 방어 자산(SHY)
                    if curr_sp500 < curr_sma200 and curr_unrate > curr_unrate_sma:
                        current_timing_asset = 'SHY'
                    else:
                        current_timing_asset = 'QQQ'
                except Exception:
                    pass
            
            # 2. 연간 리밸런싱 (고정 자산 포함 전체 비중 25%씩 재조정)
            if last_year is not None and current_date.year != last_year:
                total_val = np.sum(asset_values)
                asset_values = np.array([total_val * 0.25] * 4)
            
            # --- 수익률 반영 ---
            # 고정 자산 3개(IWD, GLD, IEF) 수익률 반영
            for idx, ticker in enumerate(self.fixed_assets):
                asset_values[idx] *= (1 + daily_returns.loc[current_date, ticker])
            
            # 타이밍 자산(QQQ 또는 SHY) 수익률 반영
            asset_values[3] *= (1 + daily_returns.loc[current_date, current_timing_asset])
            
            # 당일 총 포트폴리오 가치 합산
            total_val = np.sum(asset_values)
            port_values.append(total_val)
            
            last_year = current_date.year
            last_month = current_date.month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)