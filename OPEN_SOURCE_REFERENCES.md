# Open-source reference register

STOCK RADAR does not copy proprietary screening logic or user-published trading
scripts.  Public mathematical definitions are independently implemented and
then tested as challengers against the existing model.

| Reference | Licence | What is referenced | Integration rule |
|---|---|---|---|
| [TA-Lib](https://github.com/TA-Lib/ta-lib) | BSD-3-Clause | Wilder RSI, ATR, ADX and standard indicator definitions | Formula behaviour only; independent Python implementation |
| [ta](https://github.com/bukosabino/ta) | MIT | Candidate catalogue: CMF, Donchian, ROC and Bollinger width | Formula behaviour only; no copied source |
| [QuantConnect LEAN](https://github.com/QuantConnect/Lean) | Apache-2.0 | Separation of indicators, execution assumptions and evaluation | Architectural reference only |
| [Microsoft Qlib](https://github.com/microsoft/qlib) | MIT | Multiple-factor challenger evaluation and explicit benchmark metrics | Evaluation concept only |

Excluded: GPL/AGPL code integration, code without a clear licence, proprietary
site reverse engineering, and TradingView community scripts without explicit
reuse permission.

Adoption gate: a challenger must improve after-cost mean return and profit
factor without relying on a tiny sample.  A higher win rate with negative
expectancy is rejected and still shown in the research report.
