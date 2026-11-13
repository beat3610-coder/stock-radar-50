# STOCK RADAR — 상용 베타 (판매 전 준비 중)

Streamlit 기반 한국·미국 주식 **기술적 분석 보조 도구**. 본 프로그램은 공식 투자자문이나 수익 보장 서비스가 아닙니다.

## 현재 구현된 기능
- 코스피 시가총액 상위 100 + TIGER ETF 2종, 코스닥 시가총액 상위 100, Nasdaq-100 기업 약 100 + SOXL
- **시장 단일 선택**: 선택 시장의 분석 목록·상세 차트·국내 수급을 함께 전환
- 50종목씩 순차 분석, 결과 페이지 보관, 후보 필터·정렬, CSV 내보내기
- 20·60·120·200일 이동평균, MACD, RSI, ADX, 거래량, OBV, 52주 고저점
- 관측된 저항선 기반 통합 목표가와 손절·손익비 참고치. 주가 예측이 아닌 규칙
- 일치하는 거래일의 다른 무료 시세 공급처 종가를 **0.5% 범위 내 교차 비교**. 불일치/미검증시 매수 신호 보류
- 로그인/이메일 허용목록을 사용하는 **비공개 시범 운영 모드** 지원 (설정 필요)
- 실제 무료 시세의 출처와 진단 결과, 백테스트의 한계 명시

## 직접 실행
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

### 공개 시범 모드
설정이 없으면 `APP_ACCESS_MODE=demo` 입니다. 단, 데이터 재배포 권한이 확인되지 않은 상태에서는 **공개 판매/배포를 시작하면 안 됩니다.** 시범 모드라는 이름 자체가 재배포 권한을 부여하지 않습니다.

### 비공개 OIDC 회원 모드
서버 환경변수:
```text
APP_ACCESS_MODE=private
APP_ALLOWED_EMAILS=authorized.customer@example.com,another.customer@example.com
```
인증 공급자(예: Google)를 등록하고 Render의 비밀 환경설정 또는 안전한 Streamlit Secrets에 `[auth]`를 설정합니다. **실제 키와 비밀값은 GitHub에 커밋하지 마세요.**
```toml
[auth]
redirect_uri = "https://YOUR_DOMAIN/oauth2callback"
cookie_secret = "GENERATE_A_LONG_RANDOM_SECRET"
client_id = "YOUR_PROVIDER_CLIENT_ID"
client_secret = "YOUR_PROVIDER_CLIENT_SECRET"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```
OIDC 로그인 시 `email_verified=true`이고 `APP_ALLOWED_EMAILS`에 포함된 계정만 접근 가능합니다. 다른 계정은 차단합니다. **구독 결제, 자동 결제 상태 조회, 취소/환불 관리 기능은 아직 없습니다.** 개인별 유료 판매는 결제/권한 서버 구축 후 진행해야 합니다.

## 검증
- `python test_market_switch.py`: 코스피→코스닥→나스닥→코스피 전환 시 결과·상세 선택 변경
- `python test_quote_verify.py`: 보조 시세 날짜 불일치, 공급처 동일, 종가 차이, 데이터 없음
- `python test_access_control.py`: 승인되지 않은 계정 접근 차단
- `python accuracy_audit.py`: 7개 **일부 선정 종목**의 종가 비교 및 10거래일 수익률 진단
- GitHub Actions에서 테스트 실행, `data/` 폴더에 산출 결과 저장

## 유료 판매 전 필수 선행 조건
1. **시세 재배포·상업적 표시 계약**: KRX는 개인의 단순 조회와 앱/웹 재배포 수익사업을 구분하고 별도 계약을 요구합니다. FinanceDataReader의 MIT 소스코드 라이선스는 거래소 시세의 재배포 권한과 다릅니다. 정식 계약에 맞는 공급처로 전환해야 합니다.
2. **자본시장법 검토**: 불특정 다수에게 유료로 종목별 매매 신호/가격을 제공하는 사업은 유사투자자문업 등 해당 여부를 전문가·금융당국에 확인해야 합니다. 사용자별 1:1 추천/자동 주문 등은 별도 영역입니다.
3. **실제 시세 적정성**: 국내 4종목 교차 비교에서 공급처 간 종가 불일치 0.57~1.46%를 관측했습니다. KRX 확정 데이터로 정확성 검증 전 매매 추천을 신뢰하면 안 됩니다.
4. **전략 검증**: 7개 현재 유명 종목, 310개 관측의 제한적 실험에서는 '매수 유보' 신호의 우수성을 확인하지 못했습니다. 현 시총순 목록의 과거 생존편향·수급 누락·비용을 고려한 확장된 시점별 백테스트가 필요합니다.
5. **실서비스 인프라**: 결제사(PG) 정기결제/웹훅/구독 만료 처리, 사용자 약관·개인정보 처리방침, 서비스 로그, 장애 알림, 사용량 제한, 백업, 지원 채널.
6. **공무원 겸직·사업 운영**: 사업 주체·직접 운영 여부를 기준으로 겸직 및 영리업무 제한을 먼저 확인합니다.

## 참고 링크
- KRX 데이터사업: https://openapi.krx.co.kr/contents/OPP/DATA/OPPDATA003.jsp
- 자본시장법: https://law.go.kr/LSW/lsInfoP.do?lsId=010513
- Streamlit OIDC: https://docs.streamlit.io/develop/concepts/connections/authentication

**출시 분류:** 내부 기능 확인된 상용 베타. 실제 유료 대외 판매 개시 **불가** (상기 계약·규제·결제 검증 전).
