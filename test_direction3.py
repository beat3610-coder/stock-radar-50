import unittest
import pandas as pd
import numpy as np
from direction3 import direction_label,make_direction_rows,score_directions,OnlineDirectionModel,LABELS,HORIZON

def row(i,date,available,label,market="KR"):
    values={"signal_rsi":65 if label=="UP" else 35 if label=="DOWN" else 50,
      "signal_adx":30,"signal_atr_pct":2.5,
      "signal_ma20_ma60_pct":3 if label=="UP" else -3 if label=="DOWN" else 0,
      "signal_volume_ratio":1.6,"signal_5d_return_pct":3 if label=="UP" else -3 if label=="DOWN" else 0,
      "signal_macdh_atr":.3 if label=="UP" else -.3 if label=="DOWN" else 0,
      "signal_price_ma200_pct":3 if label=="UP" else -3 if label=="DOWN" else 0,
      "signal_bb_position":.6 if label=="UP" else -.6 if label=="DOWN" else 0}
    return dict(market=market,ticker=f"S{i:05d}",signal_date=date,label_available_date=available,
       direction_label=label,forward_return_pct=4 if label=="UP" else -4 if label=="DOWN" else 0,**values)

class DirectionTests(unittest.TestCase):
    def test_exact_boundaries_and_probabilities(self):
        self.assertEqual(direction_label(3),"SIDEWAYS")
        self.assertEqual(direction_label(-3),"SIDEWAYS")
        self.assertEqual(direction_label(3.0001),"UP")
        self.assertEqual(direction_label(-3.0001),"DOWN")
        model=OnlineDirectionModel()
        for i in range(120):model.update(row(i,"2025-01-01","2025-01-15",LABELS[i%3]))
        probs=model.predict(row(999,"2025-02-01","2025-02-15","UP"))
        self.assertEqual(len(probs),3)
        self.assertAlmostEqual(sum(probs),1.,places=12)
        self.assertTrue(all(0<=p<=1 for p in probs))
    def test_model_learns_three_distinct_patterns(self):
        model=OnlineDirectionModel()
        for i in range(900):
            model.update(row(i,"2025-01-01","2025-01-15",LABELS[i%3]))
        for label in LABELS:
            p=model.predict(row(999,"2025-02-01","2025-02-15",label))
            self.assertEqual(LABELS[p.index(max(p))],label)
    def test_future_outcomes_never_update_same_day_predictions(self):
        past=[row(i,"2025-01-01","2025-01-15",LABELS[i%3]) for i in range(150)]
        test=row(900,"2025-02-01","2025-02-15","UP")
        future=row(901,"2025-02-01","2025-02-09","DOWN")
        a=score_directions(past+[test])
        b=score_directions(past+[test,future])
        self.assertEqual(a["windows"]["train"]["confusion"]["UP"],
                         b["windows"]["train"]["confusion"]["UP"])
        self.assertEqual(a["windows"]["train"]["accuracy_pct"],
                         b["windows"]["train"]["accuracy_pct"])
    def test_all_three_classes_counted_including_sideways(self):
        past=[row(i,"2025-01-01","2025-01-15",LABELS[i%3]) for i in range(150)]
        future=[row(i+200,"2025-12-01","2025-12-15",LABELS[i%3]) for i in range(60)]
        report=score_directions(past+future)
        metrics=report["windows"]["validation"]
        self.assertEqual(report["tuning"]["candidate_count"],75)
        self.assertEqual(len(report["tuning"]["candidates"]),75)
        self.assertFalse(report["tuning"]["diagnostic_used_for_selection"])
        self.assertEqual(metrics["total"],60)
        self.assertEqual(sum(metrics["actual"].values()),60)
        self.assertEqual(sum(metrics["predicted"].values()),60)
        self.assertEqual(sum(sum(z.values()) for z in metrics["confusion"].values()),60)
        for cls in LABELS:
            self.assertEqual(metrics["actual"][cls],20)
    def test_no_hindsight_in_features_for_future_relabel(self):
        n=280
        idx=pd.bdate_range("2025-01-01",periods=n)
        x=pd.DataFrame({"Close":np.linspace(100,130,n),"Volume":np.full(n,10000.),
            "MA20":np.full(n,105.),"MA60":np.full(n,104.),"MA200":np.full(n,100.),
            "RSI":np.full(n,56.),"ADX":np.full(n,25.),"ATR":np.full(n,2.),
            "VOL20":np.full(n,9000.),"MACDH":np.full(n,.2),
            "BBLOW":np.full(n,90.),"BBUP":np.full(n,140.)},index=idx)
        a=make_direction_rows(x,"A","US",end_date="2027-01-01")
        self.assertGreater(len(a),0)
        y=x.copy()
        y.loc[y.index[250]:,"Close"]=y.loc[y.index[250]:,"Close"]*0.4
        b=make_direction_rows(y,"A","US",end_date="2027-01-01")
        self.assertEqual(a[0]["signal_date"],b[0]["signal_date"])
        self.assertEqual(a[0]["signal_rsi"],b[0]["signal_rsi"])
        self.assertNotEqual(a[0]["direction_label"],b[0]["direction_label"])

if __name__=="__main__":unittest.main()
