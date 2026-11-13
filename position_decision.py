"""Position-management arithmetic: user-owned shares, explicit break-even.

Amounts include an assumed round-trip percentage cost, not brokerage tax
advice. Gap risk, taxes and intraday execution are NOT included.
"""
import math

def position_review(*, entry,shares,current,stop,target,ma20=None,macdh=None,cost_pct=0.35):
    def good(v):
        try:return math.isfinite(float(v))
        except (ValueError,TypeError):return False
    if not all(map(good,(entry,shares,current,stop,target,cost_pct))):
        return {"valid":False,"reason":"필수 입력값 누락"}
    entry=float(entry);shares=int(shares);current=float(current)
    stop=float(stop);target=float(target);cost_pct=float(cost_pct)
    if not(0<entry and shares>0 and 0<current and 0<stop<target
           and 0<=cost_pct<=5):
        return {"valid":False,"reason":"가격·주수 또는 왕복거래비용 입력 범위 오류"}
    def net_at(price):
        return shares*(price-entry) - shares*entry*(cost_pct/100)
    net=net_at(current)
    stop_net=net_at(stop)
    target_net=net_at(target)
    if current<=stop:
        label="손절선 이탈·청산 우선 검토"
        why="현재가격이 설정 손절가 이하"
    elif current>=target:
        label="익절 목표 도달·이익실현 검토"
        why="현재가격이 목표가 이상"
    elif good(ma20) and current<float(ma20) and good(macdh) and float(macdh)<0:
        label="추세 약화·비중 축소 검토"
        why="20일 이동평균선 하회·MACD 약세"
    else:
        label="계획 유지·손절가 준수"
        why="설정 손절·익절선 미도달"
    return {"valid":True,"shares":shares,"cost_basis":round(entry*shares,2),
            "marked_value":round(current*shares,2),
            "liquidation_net":round(net,2),
            "liquidation_net_pct":round(100*net/(entry*shares),3),
            "stop_net":round(stop_net,2),
            "target_net":round(target_net,2),
            "stop_vs_now_pct":round(100*(stop/current-1),2),
            "target_vs_now_pct":round(100*(target/current-1),2),
            "decision":label,"why":why,
            "cost_note":"왕복 비용 추정치. 실제 세금·스프레드·갭 손실은 별도"}
