from __future__ import annotations

from experiment.rcaeval_delta_candidates import (
    _select_delta_metrics,
    evaluate_promotion,
    robust_service_rankings,
)


def test_delta_selector_excludes_absolute_levels() -> None:
    rows = [
        {
            "metric_features": {
                "svc_cpu__pre_mean": 1.0,
                "svc_cpu__post_mean": 2.0,
                "svc_cpu__mean_delta": 1.0,
            }
        }
    ]

    assert _select_delta_metrics(rows) == [{"svc_cpu__mean_delta": 1.0}]


def test_robust_service_score_ranks_shifted_service_first() -> None:
    services = ["a", "b"]
    train = [
        {"metric_features": {"a_cpu__mean_delta": 0.0, "b_cpu__mean_delta": 0.0}},
        {"metric_features": {"a_cpu__mean_delta": 0.0, "b_cpu__mean_delta": 0.0}},
    ]
    test = [
        {"metric_features": {"a_cpu__mean_delta": 5.0, "b_cpu__mean_delta": 0.0}}
    ]

    rankings, scores = robust_service_rankings(
        train, test, services, epsilon=1e-9, score_cap=20.0
    )

    assert rankings == [["a", "b"]]
    assert scores[0]["a"] == 20.0
    assert scores[0]["b"] == 0.0


def test_promotion_uses_pre_registered_selection_order() -> None:
    def model(overall: float, focus: float, focus_name: str):
        return {
            "out_of_fold": {"macro_f1": overall},
            "folds": [{"held_out": focus_name, "metrics": {"macro_f1": focus}}],
        }

    designs = {
        "repetition-held-out": {
            "models": {
                "delta_only_lr": model(0.9, 0.8, "1"),
                "logs_plus_delta_lr": model(0.9, 0.8, "1"),
                "robust_service_delta_score": model(0.9, 0.8, "1"),
            }
        },
        "fault-type-held-out": {
            "models": {
                "delta_only_lr": model(0.9, 0.8, "delay"),
                "logs_plus_delta_lr": model(0.9, 0.8, "delay"),
                "robust_service_delta_score": model(0.9, 0.8, "delay"),
            }
        },
    }
    spec = {
        "promotion_gate": {
            "selection_order": [
                "delta_only_lr",
                "robust_service_delta_score",
                "logs_plus_delta_lr",
            ],
            "requirements": {
                "repetition_held_out_macro_f1": 0.8,
                "repetition_1_macro_f1": 0.7,
                "fault_type_held_out_macro_f1": 0.8,
                "delay_fold_macro_f1": 0.7,
            },
        }
    }

    result = evaluate_promotion(designs, spec)

    assert result["selected_for_external_confirmation"] == "delta_only_lr"
