import math
import unittest

from audit_63_fast import (_collect_earnings_history, _earnings_policy_stage,
    _feature_coverage, _legacy_policy_stage, _paired_earnings_ablation,
    _three_way_report)
from opendart import EARNINGS_FEATURES


class EarningsAuditTests(unittest.TestCase):
    def test_missing_api_key_abstains_without_network_or_collection(self):
        class Client:
            key_present = False

            def get_corp_codes(self):
                raise AssertionError("keyless mode must not make a request")

        records, summary = _collect_earnings_history(
            [("005930", "Samsung")], client_factory=Client, budget_seconds=1,
            clock=lambda: 0, progress=lambda *args, **kwargs: None)
        self.assertFalse(summary["key_present"])
        self.assertEqual(summary["attempted_symbols_n"], 0)
        self.assertEqual(records["005930"]["diagnostics"]["status"], "ABSTAIN_NO_API_KEY")

    def test_earnings_features_are_covered_and_nonconstant_values_remain_visible(self):
        rows = [
            {key: float(i) for i, key in enumerate(EARNINGS_FEATURES, 1)},
            {key: float(i + 1) for i, key in enumerate(EARNINGS_FEATURES, 1)},
        ]
        report, excluded = _feature_coverage(rows)
        for key in EARNINGS_FEATURES:
            self.assertEqual(report[key]["non_missing_n"], 2)
            self.assertFalse(report[key]["auto_excluded_constant"])
            self.assertNotIn(key, excluded)

    def test_missing_earnings_abstains_and_keeps_upstream_legacy_trade(self):
        row = {"ticker": "005930", "signal_date": "2025-01-02",
            "entry_date": "2025-01-03", "exit_date": "2025-01-06",
            "net_pct": 1.0, "market_regime": "NEUTRAL",
            "disclosure_bad_60d": 0, "disclosure_periodic_days_since": 100,
            "earnings_data_available": 0}
        accepted = _legacy_policy_stage([row], "legacy_risk_off_disclosure_earnings",
            {"disclosure_bad_60d_max": 0, "disclosure_periodic_days_since_floor": 0,
             "earnings_momentum_min": 0, "op_yoy_min": 0, "opm_min": 0})
        self.assertEqual(len(accepted), 1)

    def test_no_open_dart_data_marks_bde_stage_not_evaluated(self):
        stage = {"status": "EVALUATED", "params": {},
            "periods": {"development": {}}}
        report = _earnings_policy_stage([], {}, 1, stage,
            {"available_symbols_n": 0, "requested_symbols_n": 1, "coverage_pct": 0})
        self.assertEqual(report["status"], "NOT_EVALUATED")
        self.assertEqual(report["periods"]["validation"]["trades"], 0)

    def test_three_way_report_uses_named_probabilities_and_actual_majority_baseline(self):
        report = _three_way_report([
            {"label": "UP", "prediction": "UP",
             "probabilities": {"UP": .8, "DOWN": .1, "SIDEWAYS": .1}},
            {"label": "DOWN", "prediction": "UP",
             "probabilities": {"UP": .6, "DOWN": .2, "SIDEWAYS": .2}},
            {"label": "DOWN", "prediction": "DOWN",
             "probabilities": {"UP": .1, "DOWN": .8, "SIDEWAYS": .1}},
        ])
        self.assertEqual(report["evaluation_rows_n"], 3)
        self.assertAlmostEqual(report["accuracy_pct"], 200 / 3)
        self.assertAlmostEqual(report["majority_class_baseline_accuracy_pct"], 200 / 3)
        self.assertEqual(report["class_metrics"]["DOWN"]["actual"], 2)
        self.assertIn("confusion_matrix", report)

    def test_earnings_ablation_pairs_only_identical_candidates(self):
        base = [{"ticker": "A", "signal_date": "2025-01-02",
                 "entry_date": "2025-01-03", "strategy": "trend_pullback",
                 "net_pct": 1, "realized_r": .5, "model_predicted_r": .6,
                 "model_probability": .7},
                {"ticker": "B", "signal_date": "2025-01-02",
                 "entry_date": "2025-01-03", "strategy": "trend_pullback",
                 "net_pct": -1, "realized_r": -.5, "model_predicted_r": -.6,
                 "model_probability": .3},
                {"ticker": "C", "signal_date": "2025-01-02",
                 "entry_date": "2025-01-03", "strategy": "trend_pullback",
                 "net_pct": .5, "realized_r": .25, "model_predicted_r": .3,
                 "model_probability": .6}]
        with_earnings = [{**base[0], "model_predicted_r": .8},
            {**base[1], "model_predicted_r": -.8},
            {**base[2], "model_predicted_r": .4},
            {"ticker": "D", "signal_date": "2025-01-02",
             "entry_date": "2025-01-03", "strategy": "trend_pullback",
             "net_pct": -1, "realized_r": -.5, "model_predicted_r": -.6,
             "model_probability": .3}]
        result = _paired_earnings_ablation(base, with_earnings)
        self.assertEqual(result["same_sample_n"], 3)
        self.assertEqual(result["plus_flow"]["trades"], 3)
        self.assertEqual(result["plus_earnings"]["trades"], 3)
        self.assertAlmostEqual(result["delta_plus_earnings_minus_plus_flow"]["rank_spearman_pooled"], 0)


if __name__ == "__main__":
    unittest.main()
