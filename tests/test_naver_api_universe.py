import unittest
from unittest.mock import Mock
from naver_api_universe import naver_api_top100

def mocked_session(market="KOSPI", n=100):
    data={"items":[{"itemCode":f"{i:06d}","stockName":f"회사{i}",
                    "marketType":market} for i in range(1,n+1)]}
    response=Mock()
    response.json.return_value=data
    response.raise_for_status.return_value=None
    session=Mock()
    session.headers={}
    session.get.return_value=response
    return session

class NaverApiUniverseTests(unittest.TestCase):
    def test_100_kospi(self):
        rows,source=naver_api_top100("KR",session=mocked_session())
        self.assertEqual(len(rows),100)
        self.assertIn("KOSPI",source)
    def test_100_kosdaq(self):
        rows,source=naver_api_top100("KQ",session=mocked_session("KOSDAQ"))
        self.assertEqual(len(rows),100)
    def test_wrong_market_rejected(self):
        with self.assertRaises(RuntimeError):
            naver_api_top100("KR",session=mocked_session("KOSDAQ"))
    def test_short_list_fails_closed(self):
        with self.assertRaises(RuntimeError):
            naver_api_top100("KR",session=mocked_session(n=20))
    def test_unexpected_schema_fails_closed(self):
        session=mocked_session()
        session.get.return_value.json.return_value={"error":"invalid"}
        with self.assertRaises(ValueError):
            naver_api_top100("KR",session=session)

if __name__=="__main__":
    unittest.main()
