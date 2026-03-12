import pandas as pd
import numpy as np

class CompositeDualMomentum:
    def __init__(self):
        """
        혼합 듀얼 모멘텀 전략
        4개 파트(지수, 섹터, 채권 등)별로 각각 듀얼 모멘텀 적용
        """
        # 파트 구성 및 티커 정의 (실제 클래스 설정에 맞춰 사용)
        self.parts = {
            "Equities": ["SPY", "EFA"],
            "Fixed Income": ["HYG", "LQD"],
            "Real Estate": ["VNQ", "REM"],
            "Commodities": ["GLD", "DBC"]
        }
        self.cash_ticker = "BIL"
        self.all_tickers = list(set([t for tickers in self.parts.values() for t in tickers] + [self.cash_ticker]))

    def calculate(self, data):
        # 1. 데이터 준비
        prices = data[self.all_tickers].copy().dropna()
        
        # 미래 참조 방지: 판단용 가격 지표는 하루 뒤로 밀기
        logic_prices = prices.shift(1)
        
        # 최근 12개월(252 거래일) 수익률 계산 (판단용 지표)
        lookback = 252
        returns_12m = logic_prices.pct_change(lookback)
        
        # 월간 리샘플링 (판단용)
        monthly_returns_12m = returns_12m.resample('ME').last()
        
        # 실제 매매 수익률은 당일(T) 가격 변동 사용
        daily_returns = prices.pct_change().fillna(0)
        port_values = []
        
        # 초기 상태: 4개 파트 모두 현금(BIL)으로 시작
        current_assets = {part: self.cash_ticker for part in self.parts}
        total_val = 1.0
        # 각 파트별 초기 가치 배분 (25%씩)
        part_values = np.array([0.25, 0.25, 0.25, 0.25])
        
        last_month = None
        
        for i in range(len(daily_returns)):
            current_date = daily_returns.index[i]
            current_month = current_date.month
            
            # [방어 로직] 12개월 웜업 구간 (T-1일 기준 데이터가 있는지 확인)
            if pd.isna(returns_12m.loc[current_date]).any():
                port_values.append(total_val)
                last_month = current_month
                continue
            
            # [리밸런싱 로직] 월이 바뀌면 각 파트별 승자 결정
            if last_month is not None and current_month != last_month:
                try:
                    # [수정] 어제(T-1)까지 확정된 가장 최신 월간 모멘텀 데이터 추출
                    mom_data = monthly_returns_12m.loc[:current_date].iloc[-1]
                    bil_12m = mom_data[self.cash_ticker]
                    
                    # 현재 총 가치를 다시 4등분하여 리밸런싱 준비
                    current_total_val = np.sum(part_values)
                    part_values = np.array([current_total_val * 0.25] * 4)
                    
                    for idx, (part_name, tickers) in enumerate(self.parts.items()):
                        asset_a, asset_b = tickers
                        a_12m = mom_data[asset_a]
                        b_12m = mom_data[asset_b]
                        
                        # 1. 상대적 모멘텀: 어제까지의 성적 기준 둘 중 승자 선택
                        winner = asset_a if a_12m > b_12m else asset_b
                        winner_12m = max(a_12m, b_12m)
                        
                        # 2. 절대적 모멘텀: 승자가 현금(BIL)보다 수익률이 낮으면 현금 선택
                        if winner_12m > bil_12m:
                            current_assets[part_name] = winner
                        else:
                            current_assets[part_name] = self.cash_ticker
                            
                except (IndexError, KeyError):
                    pass
            
            # [수익률 반영] 어제 내린 결정으로 오늘 각 파트의 가치 업데이트
            for idx, (part_name, asset) in enumerate(current_assets.items()):
                part_values[idx] *= (1 + daily_returns.loc[current_date, asset])
            
            # 전체 파트의 가치 합산
            total_val = np.sum(part_values)
            port_values.append(total_val)
            last_month = current_month
            
        return pd.Series(port_values, index=daily_returns.index).pct_change().fillna(0)