# Strategy Mixer Pro

### Multi-Strategy Quant Backtesting Dashboard

여러 퀀트 전략을 조합하고 포트폴리오 비중을 최적화할 수 있는 백테스팅 대시보드입니다.
14개의 자산배분 전략을 지원하며, 전략 조합 및 샤프 비율 기반 포트폴리오 최적화를 테스트할 수 있습니다.

Streamlit 기반 인터페이스에서 전략 선택, 성과 비교, 최적 비중 계산 등을 한 번에 확인할 수 있습니다.

---

## Features

**Multiple Quant Strategies**
정적 자산배분과 모멘텀 기반 전략을 포함한 14개 전략 지원

**Look-ahead Bias Prevention**
모든 전략에 `shift(1)` 로직을 적용하여 미래 데이터를 참조하지 않도록 처리

**Unified Data Engine**
여러 ETF의 상장 시점을 고려하여 데이터 시작 시점을 자동 동기화

**Portfolio Optimization**
SciPy 기반 최적화(SLSQP)를 사용하여 선택한 전략들의 **Max Sharpe 비중 계산**

**Interactive Visualization**
Plotly 기반 인터랙티브 차트 제공

* 누적 수익률 곡선
* 연도별 수익률
* 전략 간 상관관계 히트맵

---

## Implemented Strategies

| Category            | Strategies                                               |
| ------------------- | -------------------------------------------------------- |
| Static Allocation   | 60/40, Permanent, All Seasons, All Weather               |
| Momentum            | Dual Momentum, Composite Dual Momentum, PAA              |
| Tactical Allocation | GTAA5, LAA, DAA                                          |
| Bold Allocation     | VAA (Aggressive / Balanced), BAA (Aggressive / Balanced) |

---

## Tech Stack

**Language**

* Python 3.12+

**Framework**

* Streamlit

**Data**

* FinanceDataReader
* FRED
* Pandas
* NumPy

**Optimization**

* SciPy (SLSQP)

**Visualization**

* Plotly
* QuantStats

---

## How to Run

```bash
# clone repository
git clone https://github.com/cslim1019-gif/multiple_quant_strategy.git

# install dependencies
pip install -r requirements.txt

# run dashboard
streamlit run app.py
```

---

## Portfolio Optimization

선택한 전략들의 비중을 조정하여 **Sharpe Ratio가 최대가 되는 포트폴리오**를 찾습니다.

### Optimization Problem

$$Maximize \quad S_p = \frac{E[R_p - R_f]}{\sigma_p}$$

### subject to
($$\sum w_i = 1$$)
($$0 \le w_i \le 1$$)

SciPy의 **SLSQP optimizer**를 사용하여 위 제약 조건을 만족하는 최적 비중을 계산합니다.
