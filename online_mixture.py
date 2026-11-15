"""Online, outcome-calibrated mixture of prequential probability experts.

Each mixture weight changes only when the corresponding trade has fully exited.
Unlike a fixed threshold, every eligible candidate receives a probability.
All inputs, including the cached expert forecasts, are available at signal time.
"""
import math

class OnlineExpertMixture:
    def __init__(self, decay=0.998, learning_rate=1.25, floor=0.08):
        self.decay=float(decay)
        self.learning_rate=float(learning_rate)
        self.floor=float(floor)
        self.log_weights=[0.0,0.0,0.0]  # online logistic, adaptive regime, market prior
        self.updated=0
    def weights(self):
        m=max(self.log_weights)
        unnorm=[math.exp(x-m) for x in self.log_weights]
        s=sum(unnorm)
        n=len(unnorm)
        return [self.floor+(1-n*self.floor)*x/s for x in unnorm]
    def predict(self, probabilities):
        if probabilities is None or len(probabilities)!=3:
            return None
        if not all(x is not None and math.isfinite(float(x)) and 0<=x<=1 for x in probabilities):
            return None
        weights=self.weights()
        return sum(float(p)*w for p,w in zip(probabilities,weights))
    def update(self, signal_probabilities, net_pct):
        """Use the *cached signal-time* predictions, never recompute at exit."""
        if self.predict(signal_probabilities) is None:return False
        try:
            net=float(net_pct)
            if not math.isfinite(net):return False
        except (TypeError,ValueError):return False
        y=1.0 if net>0 else 0.0
        for i,p in enumerate(signal_probabilities):
            clipped=min(1-1e-8,max(1e-8,float(p)))
            log_loss=-(y*math.log(clipped)+(1-y)*math.log(1-clipped))
            self.log_weights[i]=self.decay*self.log_weights[i]-self.learning_rate*log_loss
        # Shift is numerically safe and leaves the softmax invariant.
        mx=max(self.log_weights)
        self.log_weights=[x-mx for x in self.log_weights]
        self.updated+=1
        return True
