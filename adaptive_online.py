"""Constant-memory online logistic learner for continuously arriving completed trades.

Each feature weight is updated *only after exit*. This is genuine parameter
learning, not threshold screening. Predict and update are O(number of features).
Research-only; no promise of positive trading edge.
"""
import math

FEATURES=("signal_rsi","signal_adx","signal_atr_pct",
          "signal_ma20_ma60_pct","signal_volume_ratio",
          "signal_5d_return_pct","entry_gap_pct")
CENTER=(55.,25.,2.5,2.,1.3,2.,0.)
SCALE=(20.,20.,3.,4.,1.5,8.,2.5)
def signal_features(trade):
    try:
        values=[float(trade[k]) for k in FEATURES]
        if not all(math.isfinite(v) for v in values):return None
        return [1.]+[max(-3.,min(3.,(v-c)/s)) for v,c,s in zip(values,CENTER,SCALE)]
    except (ValueError,TypeError,KeyError,OverflowError):
        return None

class OnlineLogistic:
    """AdaGrad-regularized SGD with a small market-prior ensemble."""
    def __init__(self,learning_rate=0.16,regularization=0.05):
        self.w=[0.]*8
        self.grad_sq=[4.]*8
        self.lr=learning_rate
        self.l2=regularization
        self.count=0
        self.wins=0
    def prior(self):
        return (self.wins+2)/(self.count+4)
    def predict(self,t):
        v=signal_features(t)
        if v is None or self.count<35:return None
        score=sum(a*b for a,b in zip(self.w,v))
        p=1/(1+math.exp(-max(-30,min(30,score))))
        # Mix in the market-only prior to avoid extreme early predictions.
        return 0.80*p+0.20*self.prior()
    def update(self,t):
        v=signal_features(t)
        try:
            net=float(t["net_pct"])
            if v is None or not math.isfinite(net):return False
        except (TypeError,ValueError,KeyError):return False
        y=float(net>0)
        pred=1/(1+math.exp(-max(-30,min(30,sum(a*b for a,b in zip(self.w,v))))))
        for i,feat in enumerate(v):
            gradient=(pred-y)*feat+(0 if i==0 else self.l2*self.w[i])
            self.grad_sq[i]+=gradient*gradient
            self.w[i]-=self.lr*gradient/math.sqrt(self.grad_sq[i])
        self.count+=1
        self.wins+=int(y)
        return True
