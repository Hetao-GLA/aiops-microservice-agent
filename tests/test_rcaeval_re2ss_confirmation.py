from __future__ import annotations

from experiment.rcaeval_re2ss_confirmation import (
    CANDIDATE,
    REFERENCE,
    evaluate_confirmation_gate,
)


def _model(overall: float, top3: float, folds: dict[str, float]):
    return {
        "out_of_fold": {"macro_f1": overall, "top_3_accuracy": top3},
        "folds": [
            {"held_out": held_out, "metrics": {"macro_f1": value}}
            for held_out, value in folds.items()
        ],
    }


def _spec() -> dict[str, object]:
    return {
        "confirmation_gate": {
            "requirements": {
                "candidate_repetition_macro_f1": 0.8,
                "candidate_fault_macro_f1": 0.8,
                "candidate_min_repetition_fold_macro_f1": 0.65,
                "candidate_delay_macro_f1": 0.55,
                "candidate_repetition_top3_accuracy": 0.95,
                "candidate_fault_top3_accuracy": 0.95,
                "candidate_minus_reference_repetition_macro_f1": 0.0,
                "candidate_minus_reference_fault_macro_f1": 0.0,
                "candidate_minus_reference_delay_macro_f1": 0.0,
            }
        }
    }


def test_confirmation_gate_requires_every_check() -> None:
    designs = {
        "repetition-held-out": {
            "models": {
                REFERENCE: _model(0.8, 0.95, {"1": 0.7, "2": 0.8, "3": 0.9}),
                CANDIDATE: _model(0.9, 1.0, {"1": 0.8, "2": 0.9, "3": 1.0}),
            }
        },
        "fault-type-held-out": {
            "models": {
                REFERENCE: _model(0.8, 0.95, {"delay": 0.55}),
                CANDIDATE: _model(0.9, 1.0, {"delay": 0.65}),
            }
        },
    }

    result = evaluate_confirmation_gate(designs, _spec())

    assert result["confirmation_passed"] is True
    assert result["decision"] == "cross_system_confirmation_passed"


def test_confirmation_gate_fails_when_candidate_loses_to_reference() -> None:
    designs = {
        "repetition-held-out": {
            "models": {
                REFERENCE: _model(0.95, 1.0, {"1": 0.9}),
                CANDIDATE: _model(0.9, 1.0, {"1": 0.9}),
            }
        },
        "fault-type-held-out": {
            "models": {
                REFERENCE: _model(0.8, 1.0, {"delay": 0.6}),
                CANDIDATE: _model(0.9, 1.0, {"delay": 0.65}),
            }
        },
    }

    result = evaluate_confirmation_gate(designs, _spec())

    assert result["confirmation_passed"] is False
    assert result["checks"][
        "candidate_minus_reference_repetition_macro_f1"
    ] is False
