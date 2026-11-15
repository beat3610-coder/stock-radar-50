from adaptive_trade_plans import PlanLearner


plans={"good":(1.5,1.2,5),"bad":(1.5,.7,5)}
learner=PlanLearner(plans)
for i in range(400):
    score=i/400
    good_net=(1.0 if i%2==0 else -0.35) if i>=200 else -0.1
    bad_net=-0.2 if i>=200 else 0.05
    learner.history["good"].append((score,.60,good_net))
    learner.history["bad"].append((score,.70,bad_net))
learner.profile_dirty=100
predictions={
    "good":{"expected_net_pct":1.1,"p_success":.61},
    "bad":{"expected_net_pct":1.2,"p_success":.72},
}
chosen=learner.selective_choose(predictions,.50)
assert chosen is not None and chosen[0]=="good"
assert chosen[2]["mean_net_pct"]>0
assert chosen[2]["profit_factor"]>1.05

predictions["good"]["expected_net_pct"]=-1
assert learner.selective_choose(predictions,.50) is None
print("selective plan tests passed")
