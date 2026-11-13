"""Optional production access policy. No shared passwords or hard-coded buyer records.

APP_ACCESS_MODE=private enables OIDC + authorized email allowlist.
Absent setup = public demonstration only, never a paid subscription.
"""
import os

def access_mode():
    return os.environ.get("APP_ACCESS_MODE","demo").strip().lower()

def is_allowed_identity(user, allowed_csv):
    allowed={x.strip().casefold() for x in allowed_csv.split(",") if x.strip()}
    if not allowed:return False
    try:
        email=str(user.get("email") or "").strip().casefold()
        logged_in=bool(user.get("is_logged_in"))
        verified=user.get("email_verified") is True
        return logged_in and verified and email in allowed
    except (AttributeError,TypeError):
        return False

def authorized_emails_configured():
    return bool(os.environ.get("APP_ALLOWED_EMAILS","").strip())

def auth_required():
    return access_mode()=="private"

def enforce_access(st):
    if not auth_required():
        return
    if not authorized_emails_configured():
        st.error("접근 제어 설정 오류: APP_ALLOWED_EMAILS가 등록되지 않았습니다. 비공개 서비스는 공개하지 않습니다.")
        st.stop()
    try:
        authenticated=bool(st.user.is_logged_in)
    except (AttributeError,RuntimeError,KeyError):
        st.error("로그인 설정이 없습니다. Streamlit [auth] OIDC 설정을 서버의 안전한 비밀값 저장소에 먼저 등록하세요.")
        st.stop()
    if not authenticated:
        st.title("STOCK RADAR | 회원 로그인")
        st.caption("유료 사용자를 위한 별도 로그인입니다. 회원 권한은 서버에서 검증합니다.")
        st.button("로그인",on_click=st.login,type="primary")
        st.stop()
    if not is_allowed_identity(st.user,os.environ.get("APP_ALLOWED_EMAILS","")):
        st.error("현재 계정에는 사용 권한이 없습니다. 운영자에게 구독 상태를 확인하세요.")
        st.button("다른 계정으로 로그인",on_click=st.logout)
        st.stop()
