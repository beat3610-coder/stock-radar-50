import unittest

from flow_warmup import run_warmup
from expanded_direction_audit import korean_flow_targets


class FlowWarmupTests(unittest.TestCase):
    def test_targets_are_unique_korean_common_stocks_only(self):
        universe={
            "KOSPI":[("005930","삼성전자","KR","전자"),
                     ("005930","삼성전자 중복","KR","전자"),
                     ("472150","ETF","KR","ETF"),
                     ("123456","샘플 보통주","KR","기타")],
            "KOSDAQ":[("247540","에코프로비엠","KQ","화학"),
                      ("0262B0","TIGER ETF","KQ","ETF"),
                      ("654321","Duplicate code","KQ","기타")],
            "US":[("AAPL","Apple Inc.","US","Technology")],
        }
        result=korean_flow_targets(universe)
        codes=[x["ticker"] for x in result]
        self.assertEqual(codes,["005930","123456","247540","654321"])
        self.assertEqual(len(codes),len(set(codes)))
        self.assertTrue(all(x["market"] in ("KR","KQ") for x in result))
        self.assertNotIn("AAPL",codes)
        self.assertNotIn("472150",codes)
        self.assertNotIn("0262B0",codes)

    def test_expired_budget_exits_with_uncompleted_tickers_recorded(self):
        universe={"KOSPI":[("005930","삼성전자","KR","전자")],
                  "KOSDAQ":[("247540","에코프로비엠","KQ","화학")],"US":[]}
        calls=[];prints=[]
        def fetch(code,**kwargs):
            calls.append(code)
            raise AssertionError("fetch must not start after the budget expires")
        result=run_warmup(time_budget_seconds=0,workers=1,universe=universe,
                          flow_fetcher=fetch,progress_printer=lambda *a,**k:prints.append(a[0]))
        self.assertEqual(calls,[])
        self.assertTrue(result["budget_exhausted"])
        self.assertEqual(result["attempted_symbols"],0)
        self.assertEqual(result["uncompleted_tickers"],["005930","247540"])
        self.assertEqual(result["available_symbols"],0)


if __name__=="__main__":
    unittest.main()
