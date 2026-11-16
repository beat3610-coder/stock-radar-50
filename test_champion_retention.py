from adaptive_trade_plans import make_plan_candidates
from buy_success import _buy_candidates
from direction3 import _direction_candidates


def run():
    direction={"learning_rate":.071,"l2":.044,"prior_mix":.27,
               "sideways_weight":1.8}
    buy={"learning_rate":.063,"l2":.051,"prior_mix":.31}
    plan={"risk_atr":2.1,"reward_rr":1.4,"hold_sessions":6}

    dc=_direction_candidates(9,direction)
    bc=_buy_candidates(9,buy)
    pc=make_plan_candidates(9,plan)

    assert any(x["name"]=="previous_champion" and
               x["learning_rate"]==direction["learning_rate"] for x in dc)
    assert any(x["name"]=="previous_champion" and
               x["learning_rate"]==buy["learning_rate"] for x in bc)
    assert pc["previous_champion"]==plan
    print("champion retention tests passed")


if __name__=="__main__":
    run()
