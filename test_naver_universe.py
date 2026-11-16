"""Offline fixtures for the Korean market-cap backup list."""
import unittest
from unittest.mock import patch
from naver_universe import naver_top100

class Response:
    status_code=200
    apparent_encoding="euc-kr"
    def __init__(self,page):
        start=(page-1)*60+1
        rows=[]
        for i in range(start,start+60):
            code=f"{i:06d}"
            rows.append(f'<tr><td>1</td><td><a href="/item/main.naver?code={code}">기업{i}</a></td><td>1000</td><td>100</td><td>50</td></tr>')
        self.text='<table class="type_2">'+''.join(rows)+'</table>'
    def raise_for_status(self):pass

class Session:
    headers={}
    def get(self,url,params,timeout):
        return Response(int(params["page"]))

class EmptySession(Session):
    def get(self,url,params,timeout):
        x=Response(1)
        x.text='<html>not valid listing</html>'
        return x

class TestKoreanFallback(unittest.TestCase):
    def test_two_pages_one_hundred(self):
        for m in ("KR","KQ"):
            rows,src=naver_top100(m,session=Session())
            self.assertEqual(len(rows),100)
            self.assertEqual(len(set(x[0] for x in rows)),100)
            self.assertIn("과거 구성종목 아님",src)
    def test_fail_closed_incomplete(self):
        with self.assertRaises(RuntimeError):
            naver_top100("KR",session=EmptySession())
    def test_market_validation(self):
        with self.assertRaises(ValueError):
            naver_top100("US",session=Session())

if __name__=="__main__":unittest.main()
