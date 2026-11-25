"""Recent public securities-lending cache (not a historical backtest source)."""
from __future__ import annotations

import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


SLB_BASE_URL = "https://apis.data.go.kr/B552481/SlbSvc"
DEALING_URL = f"{SLB_BASE_URL}/getSlbDealingByIsin"
RANK_URL = f"{SLB_BASE_URL}/getSlbStockRank"
DEFAULT_CACHE_ROOT = Path(".cache/stock-radar/slb")
DAILY_CALL_LIMIT = 100
MIN_REQUEST_INTERVAL_SECONDS = 0.3
SLB_FEATURES = ("slb_rem_qty_change_1d_pct", "slb_rem_qty_change_5d_pct",
                "slb_matc_to_rem_pct", "slb_history_days")


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _number(value):
    text = str(value if value is not None else "").replace(",", "").replace(" ", "").strip()
    if not text or text in {"-", "--", "N/A", "nan", "None"}:
        return math.nan
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _digits(value: str) -> str:
    return "".join(str(int(char)) for char in value)


def isin_for_stock_code(stock_code: str) -> str:
    """Create Korean common-stock ISIN, including the verified Luhn digit."""
    code = str(stock_code).strip().zfill(6)
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("stock_code must be a six-digit numeric code")
    prefix = f"KR7{code}00"
    expanded = "".join(str(ord(char) - ord("A") + 10) if char.isalpha() else char for char in prefix)
    digits = _digits(expanded)
    total = 0
    for offset, char in enumerate(reversed(digits)):
        value = int(char) * (2 if offset % 2 == 0 else 1)
        total += value // 10 + value % 10
    return prefix + str((-total) % 10)


def parse_slb_xml(payload: bytes | str):
    """Parse the service's XML response into header status and item mappings."""
    root = ET.fromstring(payload)
    values = {}
    items = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "item":
            item = {}
            for child in element:
                item[child.tag.rsplit("}", 1)[-1]] = (child.text or "").strip()
            items.append(item)
        elif tag in {"resultCode", "resultMsg", "resultMessage"}:
            values[tag] = (element.text or "").strip()
    return {"result_code": values.get("resultCode"),
            "result_message": values.get("resultMsg", values.get("resultMessage", "")),
            "items": items}


def _normal_date(value=None):
    if value is None:
        return datetime.now(ZoneInfo("Asia/Seoul")).date()
    if isinstance(value, date):
        return value
    text = str(value).strip().replace("-", "")
    return datetime.strptime(text[:8], "%Y%m%d").date()


class SlbClient:
    def __init__(self, cache_root=DEFAULT_CACHE_ROOT, *, http_get=None, clock=None, sleeper=None):
        self.cache_root = Path(cache_root)
        # No alternative key source is accepted; this is the only secret lookup.
        self._api_key = os.environ.get("DATA_GO_KR_API_KEY", "").strip()
        self._http_get = requests.get if http_get is None else http_get
        self._clock = time.monotonic if clock is None else clock
        self._sleep = time.sleep if sleeper is None else sleeper
        self._last_request_at = None
        self._requests_n = 0

    @property
    def key_present(self):
        return bool(self._api_key)

    def _today(self):
        return datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()

    def _traffic_path(self, day):
        return self.cache_root / "traffic" / f"{day}.json"

    def _reserve_traffic(self, day, service):
        path = self._traffic_path(day)
        try:
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"calls": 0, "by_service": {}}
        except (OSError, ValueError, TypeError):
            # A corrupt ledger must never reset the daily limit to zero.
            return False
        if int(state.get("calls", 0)) >= DAILY_CALL_LIMIT:
            return False
        state["calls"] = int(state.get("calls", 0)) + 1
        by_service = state.setdefault("by_service", {})
        by_service[service] = int(by_service.get(service, 0)) + 1
        _atomic_json(path, state)
        return True

    def _throttle(self):
        try:
            configured = float(os.environ.get("SLB_REQUEST_INTERVAL_SECONDS", str(MIN_REQUEST_INTERVAL_SECONDS)))
        except (TypeError, ValueError):
            configured = MIN_REQUEST_INTERVAL_SECONDS
        interval = max(MIN_REQUEST_INTERVAL_SECONDS, configured)
        now = self._clock()
        if self._last_request_at is not None:
            delay = interval - (now - self._last_request_at)
            if delay > 0:
                self._sleep(delay)
        self._last_request_at = self._clock()

    def _get_xml(self, url, params, service):
        if not self.key_present:
            return {"status": "ABSTAIN_NO_API_KEY", "items": [], "error": "DATA_GO_KR_API_KEY is absent."}
        day = self._today()
        if not self._reserve_traffic(day, service):
            return {"status": "DAILY_CALL_LIMIT", "items": [], "error": "Daily 100-call limit reached."}
        self._requests_n += 1
        self._throttle()
        try:
            response = self._http_get(url, params={**params, "serviceKey": self._api_key}, timeout=30)
            response.raise_for_status()
            parsed = parse_slb_xml(response.content)
            if parsed.get("result_code") not in {None, "00", "0", "000"}:
                return {"status": str(parsed["result_code"]), "items": parsed["items"],
                        "error": str(parsed.get("result_message", ""))[:160]}
            return {"status": "OK" if parsed["items"] else "NO_DATA", "items": parsed["items"],
                    "result_message": parsed.get("result_message", "")}
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            return {"status": f"HTTP_{status}" if status else "HTTP_ERROR", "items": [],
                    "error": "HTTP request failed; response text omitted."}
        except requests.RequestException as exc:
            return {"status": type(exc).__name__, "items": [],
                    "error": "Network request failed; URL and service key omitted."}
        except Exception as exc:
            return {"status": type(exc).__name__, "items": [],
                    "error": "Response could not be parsed; service key omitted."}

    def get_dealing_by_isin(self, isin: str, *, refresh=False):
        path = self.cache_root / "dealing" / f"{isin}.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError, TypeError):
            cached = {}
        if cached and not refresh and cached.get("fetched_date") == self._today():
            return cached
        result = self._get_xml(DEALING_URL, {"isin": isin}, "dealing")
        if result["status"] in {"ABSTAIN_NO_API_KEY", "DAILY_CALL_LIMIT"}:
            return {"isin": isin, "records": cached.get("records", []),
                    "fetched_date": cached.get("fetched_date"), "status": result["status"],
                    "error": result.get("error")}
        merged = {str(row.get("stdDt")): row for row in cached.get("records", []) if row.get("stdDt")}
        for row in result.get("items", []):
            if row.get("stdDt"):
                merged[str(row["stdDt"])] = row
        records = [merged[key] for key in sorted(merged)[-5:]]
        payload = {"isin": isin, "records": records, "fetched_date": self._today(),
                   "status": result["status"], "error": result.get("error")}
        _atomic_json(path, payload)
        return payload

    def get_dealing_for_ticker(self, ticker: str, *, refresh=False):
        return self.get_dealing_by_isin(isin_for_stock_code(ticker), refresh=refresh)

    def get_stock_rank(self, std_dt, *, refresh=False):
        day = _normal_date(std_dt).strftime("%Y%m%d")
        path = self.cache_root / "rank" / f"{day}.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError, TypeError):
            cached = {}
        if cached and not refresh:
            return cached
        result = self._get_xml(RANK_URL, {"stdDt": day}, "rank")
        if result["status"] in {"ABSTAIN_NO_API_KEY", "DAILY_CALL_LIMIT"}:
            return {"std_dt": day, "records": cached.get("records", []),
                    "status": result["status"], "error": result.get("error")}
        payload = {"std_dt": day, "records": result.get("items", [])[:100],
                   "status": result["status"], "error": result.get("error")}
        _atomic_json(path, payload)
        return payload


def get_slb_dealing_by_isin(isin: str, *, cache_root=DEFAULT_CACHE_ROOT, refresh=False):
    return SlbClient(cache_root=cache_root).get_dealing_by_isin(isin, refresh=refresh)


def get_slb_stock_rank(std_dt, *, cache_root=DEFAULT_CACHE_ROOT, refresh=False):
    return SlbClient(cache_root=cache_root).get_stock_rank(std_dt, refresh=refresh)


def _nan_features(status):
    return {**{name: math.nan for name in SLB_FEATURES},
            "slb_data_available": 0, "slb_status": status}


def slb_features_for_signal(ticker: str, signal_date, *, cache_root=DEFAULT_CACHE_ROOT,
                            dealing_records=None):
    """Return values only if the cached dealing series contains this signal day."""
    try:
        day = _normal_date(signal_date).strftime("%Y%m%d")
    except (TypeError, ValueError):
        return _nan_features("invalid_signal_date")
    isin = isin_for_stock_code(ticker)
    if dealing_records is None:
        path = Path(cache_root) / "dealing" / f"{isin}.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            dealing_records = cached.get("records", [])
        except (OSError, ValueError, TypeError):
            dealing_records = []
    ordered = sorted((row for row in dealing_records if str(row.get("stdDt", "")).replace("-", "") <= day),
                     key=lambda row: str(row.get("stdDt", "")).replace("-", ""))
    current = next((row for row in reversed(ordered)
                    if str(row.get("stdDt", "")).replace("-", "") == day), None)
    if current is None:
        return _nan_features("signal_date_not_covered")
    rem = _number(current.get("slbTrRemQty"))
    previous = _number(ordered[-2].get("slbTrRemQty")) if len(ordered) >= 2 else math.nan
    oldest = _number(ordered[0].get("slbTrRemQty")) if len(ordered) >= 2 else math.nan
    matched = _number(current.get("matcQty"))
    one_day = (rem / previous - 1.0) * 100.0 if math.isfinite(rem) and math.isfinite(previous) and previous > 0 else math.nan
    # This API exposes five recent rows; compare the latest with the oldest row in that window.
    five_day = (rem / oldest - 1.0) * 100.0 if math.isfinite(rem) and math.isfinite(oldest) and oldest > 0 else math.nan
    matched_to_rem = matched / rem * 100.0 if math.isfinite(matched) and math.isfinite(rem) and rem > 0 else math.nan
    return {"slb_rem_qty_change_1d_pct": one_day,
            "slb_rem_qty_change_5d_pct": five_day,
            "slb_matc_to_rem_pct": matched_to_rem,
            "slb_history_days": len(ordered), "slb_data_available": 1,
            "slb_status": "ok"}


def collect_slb_snapshot(tickers, std_dt=None, *, cache_root=DEFAULT_CACHE_ROOT, refresh=False):
    """Fetch up to 60 Korean benchmark dealing series and one daily rank snapshot."""
    client = SlbClient(cache_root=cache_root)
    if not client.key_present:
        return {"status": "ABSTAIN_NO_API_KEY", "dealing": {}, "rank": None,
                "requests_n": 0, "traffic_limit": DAILY_CALL_LIMIT}
    unique = list(dict.fromkeys(str(ticker).zfill(6) for ticker in tickers))[:60]
    dealing = {ticker: client.get_dealing_for_ticker(ticker, refresh=refresh) for ticker in unique}
    day = _normal_date(std_dt)
    rank = client.get_stock_rank(day, refresh=refresh)
    return {"status": "OK", "dealing": dealing, "rank": rank,
            "requests_n": client._requests_n, "traffic_limit": DAILY_CALL_LIMIT,
            "source": "data.go.kr B552481/SlbSvc"}
