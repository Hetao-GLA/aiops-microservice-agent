from __future__ import annotations

from experiment.rcaeval_re2tt_confirmation import (
    CANDIDATE,
    REFERENCE,
    evaluate_confirmation_gate,
)


def _model(
    overall: float,
    top3: float,
    folds: dict[str, float],
    *,
    class_fraction: float = 0.3,
    tie_fraction: float = 0.0,
) -> dict[str, object]:
    return {
        "out_of_fold": {"macro_f1": overall, "top_3_accuracy": top3},
        "folds": [
            {"held_out": held_out, "metrics": {"macro_f1": value}}
            for held_out, value in folds.items()
        ],
        "balance_and_ties": {
            "maximum_predicted_class_fraction": class_fraction,
            "exact_numeric_tie_fraction": tie_fraction,
        },
    }


def _spec() -> dict[str, object]:
    return {
        "confirmation_gate": {
            "minimums": {
                "mean_macro_f1_minus_reference": 0.0,
                "worst_design_macro_f1_minus_reference": -0.05,
                "minimum_design_macro_f1": 0.75,
                "minimum_repetition_fold_macro_f1": 0.55,
                "delay_macro_f1_minus_reference": -0.02,
                "minimum_top3_accuracy": 0.95,
            },
            "maximums": {
                "maximum_predicted_class_fraction": 0.4,
                "maximum_exact_numeric_tie_fraction": 0.05,
            },
        }
    }


def _designs(candidate_rep: float = 0.9) -> dict[str, object]:
    return {
        "repetition-held-out": {
            "models": {
                REFERENCE: _model(0.85, 0.96, {"1": 0.7, "2": 0.8, "3": 0.9}),
                CANDIDATE: _model(
                    candidate_rep, 1.0, {"1": 0.8, "2": 0.9, "3": 1.0}
                ),
            }
        },
        "fault-type-held-out": {
            "models": {
                REFERENCE: _model(0.85, 0.96, {"delay": 0.7}),
                CANDIDATE: _model(0.9, 1.0, {"delay": 0.75}),
            }
        },
    }


def test_confirmation_gate_passes_only_when_every_check_passes() -> None:
    result = evaluate_confirmation_gate(_designs(), _spec())

    assert result["confirmation_passed"] is True
    assert result["decision"] == "three_system_confirmation_passed"
    assert all(result["checks"].values())


def test_confirmation_gate_rejects_negative_mean_improvement() -> None:
    result = evaluate_confirmation_gate(_designs(candidate_rep=0.78), _spec())

    assert result["confirmation_passed"] is False
    assert result["checks"]["mean_macro_f1_minus_reference"] is False
