"""Cache-backed OpenDART annual/quarterly financial data and past-only features.

The API key is read only from ``OPENDART_API_KEY``.  Neither request URLs nor
exception messages are written to diagnostics, because both can contain secrets.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import time
import zipfile
from datetime import date, datetime
from pathlib import Path

import requests
import xml.etree.ElementTree as ET


API_BASE = "https://opendart.fss.or.kr/api"
CORP_CODE_URL = f"{API_BASE}/corpCode.xml"
FINANCIAL_URL = f"{API_BASE}/fnlttSinglAcntAll.json"
DEFAULT_CACHE_ROOT = Path(".cache/stock-radar/opendart")
REPORT_CODES = ("11014", "11012", "11013", "11011")  # Q1, half-year, Q3, annual
REPORT_QUARTERS = {"11014": 1, "11012": 2, "11013": 3, "11011": 4}
EARNINGS_FEATURES = (
    "rev_yoy_pct", "op_yoy_pct", "ni_yoy_pct", "rev_qoq_pct", "op_qoq_pct",
    "opm_pct", "opm_yoy_delta_pp", "earnings_momentum",
)
_FINAL_REQUEST_STATUSES = {"000", "013"}


def request_interval_seconds() -> float:
    try:
        value = float(os.environ.get("OPENDART_REQUEST_INTERVAL_SECONDS", "0.15"))
    except (TypeError, ValueError):
        value = 0.15
    return max(0.15, value)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def parse_corp_code_zip(payload: bytes) -> dict[str, str]:
    """Return six-digit stock code to DART corp_code mapping from corpCode.xml."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xml_name = next((name for name in archive.namelist() if name.lower().endswith(".xml")), None)
        if not xml_name:
            raise ValueError("corp-code archive has no XML member")
        root = ET.fromstring(archive.read(xml_name))
    mapping = {}
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] != "list":
            continue
        values = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in item}
        stock_code = values.get("stock_code", "").strip()
        corp_code = values.get("corp_code", "").strip()
        if re.fullmatch(r"\d{6}", stock_code) and re.fullmatch(r"\d{8}", corp_code):
            mapping[stock_code] = corp_code
    return mapping


def _number(value) -> float:
    text = str(value if value is not None else "").replace(",", "").replace(" ", "").strip()
    if not text or text in {"-", "--", "N/A", "nan", "None"}:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _date_from_receipt(rcept_no) -> str | None:
    match = re.match(r"^(\d{8})", str(rcept_no or ""))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return None


def _account_value(rows, account_ids, name_tokens) -> float:
    for account_id in account_ids:
        for row in rows:
            if str(row.get("account_id", "")).strip() == account_id:
                value = _number(row.get("thstrm_amount"))
                if math.isfinite(value):
                    return value
    for row in rows:
        name = str(row.get("account_nm", "")).replace(" ", "")
        if any(token.replace(" ", "") in name for token in name_tokens):
            value = _number(row.get("thstrm_amount"))
            if math.isfinite(value):
                return value
    return math.nan


def parse_financial_response(payload, *, stock_code: str, corp_code: str,
                             bsns_year: int, reprt_code: str, fs_div: str):
    """Parse one fnlttSinglAcntAll response, considering IS/CIS rows only."""
    if isinstance(payload, (bytes, bytearray, str)):
        payload = json.loads(payload)
    status = str(payload.get("status", ""))
    if status != "000":
        return status, [], str(payload.get("message", ""))
    rows = [row for row in payload.get("list", [])
            if str(row.get("sj_div", "")).upper() in {"IS", "CIS"}]
    if not rows:
        return "013", [], ""
    record = {
        "stock_code": str(stock_code).zfill(6),
        "corp_code": str(corp_code),
        "bsns_year": int(bsns_year),
        "reprt_code": str(reprt_code),
        "fs_div": str(fs_div),
        "rcept_no": next((str(row.get("rcept_no")) for row in rows if row.get("rcept_no")), None),
        "revenue": _account_value(rows, ("ifrs-full_Revenue",), ("매출액", "수익(매출액)", "영업수익")),
        "operating_income": _account_value(rows,
            ("dart_OperatingIncomeLoss", "ifrs-full_OperatingIncomeLoss"), ("영업이익",)),
        "net_income": _account_value(rows, ("ifrs-full_ProfitLoss",), ("당기순이익",)),
        "currency": next((str(row.get("currency")) for row in rows if row.get("currency")), None),
    }
    record["rcept_date"] = _date_from_receipt(record["rcept_no"])
    for field in ("revenue", "operating_income", "net_income"):
        if not math.isfinite(record[field]):
            record[field] = None
    return "000", [record], ""


def _empty_collection(stock_code, corp_code, *, status, key_present, reason=None):
    return {"records": [], "diagnostics": {
        "stock_code": str(stock_code).zfill(6), "corp_code": corp_code,
        "status": status, "available": False, "attempted": False,
        "key_present": bool(key_present), "requests_n": 0, "failures": [],
        "reason": reason,
    }}


class OpenDartClient:
    """Small synchronous client with a shared per-instance throttle and cache."""
    def __init__(self, cache_root=DEFAULT_CACHE_ROOT, *, http_get=None,
                 clock=None, sleeper=None):
        self.cache_root = Path(cache_root)
        self.corp_cache_path = self.cache_root / "corp-codes.json"
        # No alternate key source is accepted; this is the only secret lookup.
        self._api_key = os.environ.get("OPENDART_API_KEY", "").strip()
        self._http_get = requests.get if http_get is None else http_get
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleeper is None else sleeper
        self._last_request_at = None
        self._deadline = None
        self._requests_n = 0

    @property
    def key_present(self) -> bool:
        return bool(self._api_key)

    def set_time_budget(self, seconds: float | None):
        self._deadline = None if seconds is None else self._clock() + max(0.0, float(seconds))

    def _budget_available(self) -> bool:
        return self._deadline is None or self._clock() < self._deadline

    def _throttle(self):
        interval = request_interval_seconds()
        now = self._clock()
        if self._last_request_at is not None:
            delay = interval - (now - self._last_request_at)
            if delay > 0:
                self._sleep(delay)
        self._last_request_at = self._clock()

    def _request(self, url, params):
        if not self.key_present:
            return None, "ABSTAIN_NO_API_KEY"
        if not self._budget_available():
            return None, "SKIPPED_BUDGET"
        self._throttle()
        self._requests_n += 1
        try:
            response = self._http_get(url, params=params, timeout=30)
            response.raise_for_status()
            return response, None
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return None, f"HTTP_{status}" if status else "HTTP_ERROR"
        except requests.RequestException as exc:
            # Never serialize exception text: Requests may include the full URL and API key.
            return None, type(exc).__name__
        except Exception as exc:
            return None, type(exc).__name__

    def get_corp_codes(self) -> dict[str, str]:
        if self.corp_cache_path.exists():
            try:
                payload = json.loads(self.corp_cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return {str(code).zfill(6): str(corp) for code, corp in payload.items()}
            except (OSError, ValueError, TypeError):
                return {}
        if not self.key_present:
            return {}
        response, error = self._request(CORP_CODE_URL, {"crtfc_key": self._api_key})
        if error or response is None:
            return {}
        try:
            mapping = parse_corp_code_zip(response.content)
        except (ValueError, zipfile.BadZipFile, ET.ParseError):
            return {}
        _atomic_json(self.corp_cache_path, mapping)
        return mapping

    def company_financials(self, stock_code: str, *, corp_code: str | None = None,
                           start_year: int = 2015, end_year: int = 2026,
                           reprt_codes=REPORT_CODES):
        ticker = str(stock_code).strip().zfill(6)
        if not self.key_present:
            return _empty_collection(ticker, corp_code, status="ABSTAIN_NO_API_KEY",
                key_present=False, reason="OPENDART_API_KEY is absent; no network request was made.")
        if corp_code is None:
            corp_code = self.get_corp_codes().get(ticker)
        if not corp_code:
            return _empty_collection(ticker, None, status="NO_CORP_CODE",
                key_present=True, reason="No DART corp_code mapping for this ticker.")

        cache_path = self.cache_root / "financials" / f"{corp_code}.json"
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        except (OSError, ValueError, TypeError):
            cached = {}
        request_status = dict(cached.get("request_status", {}))
        records_by_key = {}
        for record in cached.get("records", []):
            key = (int(record.get("bsns_year", 0)), str(record.get("reprt_code")), str(record.get("fs_div")))
            records_by_key[key] = record

        failures = []
        attempted = 0
        budget_exhausted = False

        def request_fs(year, report_code, fs_div):
            nonlocal attempted, budget_exhausted
            combo = f"{year}:{report_code}:{fs_div}"
            prior = request_status.get(combo, {})
            if isinstance(prior, dict) and prior.get("status") in _FINAL_REQUEST_STATUSES:
                return str(prior.get("status")), records_by_key.get((year, report_code, fs_div))
            if not self._budget_available():
                budget_exhausted = True
                return "SKIPPED_BUDGET", None
            params = {"crtfc_key": self._api_key, "corp_code": corp_code,
                      "bsns_year": str(year), "reprt_code": report_code, "fs_div": fs_div}
            response, error = self._request(FINANCIAL_URL, params)
            attempted += 1
            if error:
                if error == "SKIPPED_BUDGET":
                    budget_exhausted = True
                else:
                    failures.append(f"{combo}:{error}")
                    request_status[combo] = {"status": error, "retryable": True}
                return error, None
            try:
                payload = response.json()
                status, parsed, message = parse_financial_response(payload,
                    stock_code=ticker, corp_code=corp_code, bsns_year=year,
                    reprt_code=report_code, fs_div=fs_div)
            except Exception as exc:
                status, parsed, message = type(exc).__name__, [], ""
            request_status[combo] = {"status": status, "retryable": status not in _FINAL_REQUEST_STATUSES}
            if status == "000" and parsed:
                record = parsed[0]
                records_by_key[(year, report_code, fs_div)] = record
                return status, record
            if status not in _FINAL_REQUEST_STATUSES:
                failures.append(f"{combo}:{status}")
            return status, None

        for year in range(int(start_year), int(end_year) + 1):
            for report_code in reprt_codes:
                status, record = request_fs(year, report_code, "CFS")
                if status == "013":
                    request_fs(year, report_code, "OFS")
                if budget_exhausted:
                    break
            if budget_exhausted:
                break

        _atomic_json(cache_path, {
            "stock_code": ticker, "corp_code": str(corp_code),
            "request_status": request_status,
            "records": sorted(records_by_key.values(), key=lambda row: (
                int(row.get("bsns_year", 0)), REPORT_QUARTERS.get(str(row.get("reprt_code")), 0),
                0 if row.get("fs_div") == "CFS" else 1)),
        })
        records = sorted(records_by_key.values(), key=lambda row: (
            int(row.get("bsns_year", 0)), REPORT_QUARTERS.get(str(row.get("reprt_code")), 0),
            0 if row.get("fs_div") == "CFS" else 1))
        return {"records": records, "diagnostics": {
            "stock_code": ticker, "corp_code": str(corp_code),
            "status": "PARTIAL_BUDGET" if budget_exhausted else ("OK" if records else "NO_DATA"),
            "available": bool(records), "attempted": attempted > 0,
            "key_present": True, "requests_n": attempted,
            "first_rcept_date": next((r.get("rcept_date") for r in records if r.get("rcept_date")), None),
            "last_rcept_date": next((r.get("rcept_date") for r in reversed(records) if r.get("rcept_date")), None),
            "records_n": len(records), "budget_exhausted": budget_exhausted,
            "failures": failures,
        }}


def get_corp_code_map(cache_root=DEFAULT_CACHE_ROOT) -> dict[str, str]:
    return OpenDartClient(cache_root=cache_root).get_corp_codes()


def get_company_financials(stock_code: str, *, corp_code: str | None = None,
                           start_year: int = 2015, end_year: int = 2026,
                           cache_root=DEFAULT_CACHE_ROOT, reprt_codes=REPORT_CODES):
    return OpenDartClient(cache_root=cache_root).company_financials(stock_code,
        corp_code=corp_code, start_year=start_year, end_year=end_year,
        reprt_codes=reprt_codes)


def _pct_change(current, previous, *, require_positive=False):
    if not (math.isfinite(current) and math.isfinite(previous)):
        return math.nan
    if previous == 0 or (require_positive and (current <= 0 or previous <= 0)):
        return math.nan
    return (current / previous - 1.0) * 100.0


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _quarter_standalones(records, field):
    by_year = {}
    for row in records:
        year = int(row.get("bsns_year", 0))
        quarter = REPORT_QUARTERS.get(str(row.get("reprt_code")))
        if quarter is None:
            continue
        by_year.setdefault(year, {})[quarter] = _finite(row.get(field))
    standalone = {}
    for year, cumulative in by_year.items():
        q1 = cumulative.get(1, math.nan)
        q2y = cumulative.get(2, math.nan)
        q3y = cumulative.get(3, math.nan)
        q4y = cumulative.get(4, math.nan)
        if math.isfinite(q1): standalone[(year, 1)] = q1
        if math.isfinite(q2y) and math.isfinite(q1): standalone[(year, 2)] = q2y - q1
        if math.isfinite(q3y) and math.isfinite(q2y): standalone[(year, 3)] = q3y - q2y
        if math.isfinite(q4y) and math.isfinite(q3y): standalone[(year, 4)] = q4y - q3y
    return standalone


def _nan_features(*, status, last_rcept_date=None, days_since=None):
    return {**{name: math.nan for name in EARNINGS_FEATURES},
        "earnings_data_available": 0, "earnings_last_rcept_date": last_rcept_date,
        "earnings_days_since_rcept": days_since, "earnings_status": status}


def earnings_features_for_signal(records, signal_date, *, corp_code,
                                  not_applicable=False, abstain=False):
    """Build eight fixed financial features using receipts strictly before signal_date.

    Quarterly standalone operating-income YoY is calculated for each quarter as
    ``(current quarter - same quarter prior year) / prior-year quarter``; earnings
    momentum is the mean of the latest two finite quarterly YoY values.
    """
    if not_applicable or not corp_code:
        return _nan_features(status="not_applicable")
    if abstain:
        return _nan_features(status="abstain_no_api_key")
    try:
        signal_day = date.fromisoformat(str(signal_date)[:10])
    except (TypeError, ValueError):
        return _nan_features(status="invalid_signal_date")
    eligible = []
    for row in records or []:
        try:
            receipt = date.fromisoformat(str(row.get("rcept_date"))[:10])
        except (TypeError, ValueError):
            continue
        if receipt < signal_day:
            eligible.append({**row, "rcept_date": receipt.isoformat()})
    if not eligible:
        return _nan_features(status="no_eligible_reports")

    # When both statements exist for a period, keep consolidated CFS first.
    by_period = {}
    for row in sorted(eligible, key=lambda r: (r["rcept_date"], str(r.get("fs_div")) == "CFS")):
        key = (int(row.get("bsns_year", 0)), str(row.get("reprt_code")))
        old = by_period.get(key)
        if old is None or (row.get("fs_div") == "CFS" and old.get("fs_div") != "CFS") or (
                row.get("fs_div") == old.get("fs_div") and row.get("rcept_date", "") > old.get("rcept_date", "")):
            by_period[key] = row
    eligible = list(by_period.values())
    latest = max(eligible, key=lambda r: (int(r.get("bsns_year", 0)),
        REPORT_QUARTERS.get(str(r.get("reprt_code")), 0), r.get("rcept_date", "")))
    ytd_key = (int(latest.get("bsns_year", 0)), str(latest.get("reprt_code")))
    prior = by_period.get((ytd_key[0] - 1, ytd_key[1]), {})
    current_revenue = _finite(latest.get("revenue")); prior_revenue = _finite(prior.get("revenue"))
    current_op = _finite(latest.get("operating_income")); prior_op = _finite(prior.get("operating_income"))
    current_ni = _finite(latest.get("net_income")); prior_ni = _finite(prior.get("net_income"))

    revenue_q = _quarter_standalones(eligible, "revenue")
    op_q = _quarter_standalones(eligible, "operating_income")
    latest_quarter = (int(latest.get("bsns_year", 0)), REPORT_QUARTERS.get(str(latest.get("reprt_code")), 0))
    previous_quarter = (latest_quarter[0] - 1, 4) if latest_quarter[1] == 1 else (latest_quarter[0], latest_quarter[1] - 1)
    rev_qoq = _pct_change(revenue_q.get(latest_quarter, math.nan), revenue_q.get(previous_quarter, math.nan), require_positive=True)
    op_qoq = _pct_change(op_q.get(latest_quarter, math.nan), op_q.get(previous_quarter, math.nan))

    opm = current_op / current_revenue * 100.0 if math.isfinite(current_op) and math.isfinite(current_revenue) and current_revenue > 0 else math.nan
    prior_opm = prior_op / prior_revenue * 100.0 if math.isfinite(prior_op) and math.isfinite(prior_revenue) and prior_revenue > 0 else math.nan

    quarterly_op_yoy = []
    for (year, quarter), current_value in op_q.items():
        previous_value = op_q.get((year - 1, quarter), math.nan)
        growth = _pct_change(current_value, previous_value)
        if math.isfinite(growth):
            quarterly_op_yoy.append(((year, quarter), growth))
    momentum_values = [value for _, value in sorted(quarterly_op_yoy)[-2:]]
    momentum = float(sum(momentum_values) / len(momentum_values)) if len(momentum_values) == 2 else math.nan

    latest_receipt = max(row["rcept_date"] for row in eligible)
    days_since = (signal_day - date.fromisoformat(latest_receipt)).days
    features = {
        "rev_yoy_pct": _pct_change(current_revenue, prior_revenue, require_positive=True),
        "op_yoy_pct": _pct_change(current_op, prior_op),
        "ni_yoy_pct": _pct_change(current_ni, prior_ni),
        "rev_qoq_pct": rev_qoq,
        "op_qoq_pct": op_qoq,
        "opm_pct": opm,
        "opm_yoy_delta_pp": opm - prior_opm if math.isfinite(opm) and math.isfinite(prior_opm) else math.nan,
        "earnings_momentum": momentum,
    }
    available = any(math.isfinite(value) for value in features.values())
    if not available:
        return _nan_features(status="no_usable_metrics", last_rcept_date=latest_receipt, days_since=days_since)
    return {**features, "earnings_data_available": 1,
        "earnings_last_rcept_date": latest_receipt,
        "earnings_days_since_rcept": days_since, "earnings_status": "ok"}


def earnings_gate_status(features, *, earnings_momentum_min=None,
                         op_yoy_min=None, opm_min=None):
    """Return PASS/REJECT/ABSTAIN; unavailable data never rejects a trade."""
    if not features or not features.get("earnings_data_available"):
        return "ABSTAIN"
    checks = (("earnings_momentum", earnings_momentum_min),
              ("op_yoy_pct", op_yoy_min), ("opm_pct", opm_min))
    for name, threshold in checks:
        if threshold is None:
            continue
        value = _finite(features.get(name))
        if not math.isfinite(value):
            return "ABSTAIN"
        if value < float(threshold):
            return "REJECT"
    return "PASS"
