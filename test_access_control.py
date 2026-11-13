"""Pure authorization regression tests for production access mode."""
from unittest.mock import patch
import access_control as a
verified={"email":"Buyer@example.com","is_logged_in":True,"email_verified":True}
unverified={"email":"Buyer@example.com","is_logged_in":True,"email_verified":False}
assert a.is_allowed_identity(verified," buyer@example.com, team@example.com ")
assert not a.is_allowed_identity(unverified,"buyer@example.com")
assert not a.is_allowed_identity({"email":"buyer@example.com","is_logged_in":False,"email_verified":True},"buyer@example.com")
assert not a.is_allowed_identity(verified,"")
assert not a.is_allowed_identity(verified,"another@example.com")
assert not a.is_allowed_identity(verified,"buyer@example.com.fake")
with patch.dict("os.environ",{"APP_ACCESS_MODE":"private","APP_ALLOWED_EMAILS":"buyer@example.com"}):
    assert a.auth_required() and a.authorized_emails_configured()
with patch.dict("os.environ",{"APP_ACCESS_MODE":"demo"}):
    assert not a.auth_required()
print("Production access authorization checks passed")
