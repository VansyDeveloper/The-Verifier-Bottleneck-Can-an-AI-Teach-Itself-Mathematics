import pytest
import torch


def test_candidate_loss_control_matches_original_objective():
    from run_ablation import candidate_objective

    logp = torch.tensor(-2.0)
    kl = torch.tensor(0.4)
    entropies = [torch.tensor(0.5), torch.tensor(0.8)]
    actual = candidate_objective(logp, kl, entropies, advantage=1.25,
                                 kl_beta=0.01, entropy_coef=0.0)
    assert actual.item() == pytest.approx(2.5 + 0.004)


def test_candidate_entropy_bonus_uses_mean_normalized_action_entropy():
    from run_ablation import candidate_objective

    actual = candidate_objective(torch.tensor(0.0), torch.tensor(0.0),
                                 [torch.tensor(0.4), torch.tensor(0.8)],
                                 advantage=0.0, kl_beta=0.01, entropy_coef=0.01)
    assert actual.item() == pytest.approx(-0.006)


def test_run_directory_refuses_to_overwrite_completed_or_failed_evidence(tmp_path):
    from run_ablation import reserve_run_directory

    path = tmp_path / "seed0_control"
    reserve_run_directory(path)
    with pytest.raises(FileExistsError):
        reserve_run_directory(path)


def test_fp32_recovery_loader_uses_explicit_float32(monkeypatch):
    import run_ablation

    seen = {}

    class Base:
        def __init__(self):
            self.config = type("Config", (), {"use_cache": True})()

        def cuda(self):
            return self

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            seen["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

        def enable_input_require_grads(self):
            seen["input_grads_enabled"] = True

    class Auto:
        @staticmethod
        def from_pretrained(path, *, dtype, trust_remote_code):
            seen["dtype"] = dtype
            return Base()

    class Peft:
        @staticmethod
        def from_pretrained(base, path, *, is_trainable):
            seen["trainable"] = is_trainable
            return base

    monkeypatch.setattr(run_ablation, "AutoModelForCausalLM", Auto, raising=False)
    monkeypatch.setattr(run_ablation, "PeftModel", Peft, raising=False)
    model = run_ablation.load_fp32_model("base", "adapter", trainable=True)
    assert seen == {"dtype": torch.float32, "trainable": True,
                    "gradient_checkpointing_kwargs": {"use_reentrant": False},
                    "input_grads_enabled": True}
    assert model.config.use_cache is False
