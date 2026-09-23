from __future__ import annotations

from experiment.rcaeval_service_delta_v2 import (
    SOFT_VOTE,
    TAIL,
    empirical_tail_rankings,
    evaluate_gate,
)


def test_empirical_tail_top2_has_no_hard_cap_and_ranks_service() -> None:
    train = [
        {
            "metric_features": {
                "a_x__mean_delta": 0.0,
                "a_y__mean_delta": 0.0,
                "b_x__mean_delta": 0.0,
                "b_y__mean_delta": 0.0,
            }
        }
        for _ in range(4)
    ]
    test = [
        {
            "metric_features": {
                "a_x__mean_delta": 5.0,
                "a_y__mean_delta": 4.0,
                "b_x__mean_delta": 0.0,
                "b_y__mean_delta": 0.0,
            }
        }
    ]

    rankings, details = empirical_tail_rankings(
        train, test, ["a", "b"], epsilon=1e-9
    )

    assert rankings == [["a", "b"]]
    assert details[0]["exact_numeric_tie"] is False
    assert details[0]["primary_margin"] > 0


def _model(macro: float, top3: float, rep: float, delay: float, fraction=0.3, ties=0.0):
    return {
        "out_of_fold": {"macro_f1": macro, "top_3_accuracy": top3},
        "folds": [
            {"held_out": "1", "metrics": {"macro_f1": rep}},
            {"held_out": "delay", "metrics": {"macro_f1": delay}},
        ],
        "balance_and_ties": {
            "maximum_predicted_class_fraction": fraction,
            "exact_numeric_tie_fraction": ties,
        },
    }


def test_v2_gate_applies_minimums_maximums_and_selection_order() -> None:
    systems = {}
    for name in ("OB", "SS"):
        systems[name] = {
            "designs": {
                "repetition-held-out": {
                    "models": {
                        "frozen_full_metrics_lr": _model(0.8, 0.95, 0.6, 0.6),
                        TAIL: _model(0.85, 1.0, 0.7, 0.7),
                        SOFT_VOTE: _model(0.85, 1.0, 0.7, 0.7),
                    }
                },
                "fault-type-held-out": {
                    "models": {
                        "frozen_full_metrics_lr": _model(0.8, 0.95, 0.6, 0.6),
                        TAIL: _model(0.85, 1.0, 0.7, 0.7),
                        SOFT_VOTE: _model(0.85, 1.0, 0.7, 0.7),
                    }
                },
            }
        }
    spec = {
        "promotion_gate": {
            "selection_order": [TAIL, SOFT_VOTE],
            "minimums": {
                "mean_macro_f1_minus_reference": 0.0,
                "worst_system_design_macro_f1_minus_reference": -0.05,
                "minimum_system_design_macro_f1": 0.75,
                "minimum_repetition_fold_macro_f1": 0.55,
                "mean_delay_macro_f1_minus_reference": -0.02,
                "minimum_top3_accuracy": 0.95,
            },
            "maximums": {
                "maximum_predicted_class_fraction": 0.4,
                "maximum_exact_numeric_tie_fraction": 0.05,
            },
        }
    }

    result = evaluate_gate(systems, spec)

    assert result["selected_for_third_system_confirmation"] == TAIL
