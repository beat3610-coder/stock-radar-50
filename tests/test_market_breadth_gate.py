import unittest
from datetime import date,timedelta
import pandas as pd
from market_breadth_gate import build_breadth,audit

def sample():
    dates=pd.bdate_range("2024-01-01",periods=400)
    frames={}
    for k in range(40):
        # A common market uptrend followed by a broad, shared sell-off.
        price=[100+i*.25 if i<280 else 170-(i-280)*.8 for i in range(400)]
        frames[str(k)]=pd.DataFrame({"Close":price,
             "High":[p*1.01 for p in price],
             "Low":[p*.99 for p in price]},index=dates)
    rows=[]
    for i in range(90,390,3):
        sig=dates[i].date().isoformat()
        out=dates[i+1].date().isoformat()
        rows.append({"market":"KR","ticker":str(i%40),"signal_date":sig,
          "exit_date":out,"net_pct":2 if i<280 else -2})
    return frames,rows

class MarketBreadthTests(unittest.TestCase):
    def test_market_selloff_detected(self):
        frames,_=sample()
        breadth=build_breadth(frames)
        dates=sorted(breadth)
        self.assertGreater(breadth[dates[80]]["market_breadth60_pct"],90)
        self.assertLess(breadth[dates[-1]]["market_breadth60_pct"],10)
    def test_no_future_prices_in_past_features(self):
        frames,_=sample()
        before=build_breadth(frames)
        target=sorted(before)[100]
        for frame in frames.values():
            frame.loc[frame.index.date>date.fromisoformat(target),"Close"]=10000
        after=build_breadth(frames)
        self.assertEqual(before[target]["market_breadth60_pct"],
                         after[target]["market_breadth60_pct"])
    def test_no_live_approval(self):
        frames,rows=sample()
        r=audit(rows,frames,market="KR")
        self.assertFalse(r["live_approved"])
        self.assertEqual(r["target_win_pct"],60)
    def test_inadequate_market_data_fails_closed(self):
        f,r=sample()
        self.assertEqual(audit(r,{"only":next(iter(f.values()))},market="KR")["usable"],0)
if __name__=="__main__":unittest.main()
