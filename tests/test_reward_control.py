import json

import torch

from iclr.common import write_json
from iclr.reward_control import gradient_check, run
from iclr.smoke import create_tiny_model
from iclr.data import generate
from iclr.modeling import load


def test_reward_identity_and_both_training_paths(tmp_path):
    torch.set_num_threads(1)
    checks = gradient_check(tmp_path / 'identity')
    assert max(r['max_error'] for r in checks) < 1e-12
    source = create_tiny_model(tmp_path / 'source')
    model, tokenizer, _ = load(source, initialize=True, train=True, device='cpu', lora_dropout=0,
                               gradient_checkpointing=False)
    model = model.merge_and_unload()
    base = tmp_path / 'base'
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    model, tokenizer, _ = load(base, train=True, device='cpu', lora_dropout=.2, gradient_checkpointing=False)
    adapter = tmp_path / 'initial_adapter'
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    data = tmp_path / 'data'
    generate(data, size=20, eval_size=2, atomic_per_op=5, smoke=True)
    budgets = []
    for method, optimizer in ((m, o) for m in ('sampled', 'exact') for o in ('sgd', 'adamw')):
        out = tmp_path / (method + '_' + optimizer)
        cfg = {'base': str(base), 'adapter': str(adapter), 'data': str(data), 'output': str(out),
               'method': method, 'optimizer': optimizer, 'steps': 2, 'learning_rate': 1e-5,
               'device': 'cpu', 'dtype': 'float32', 'prefix_batch': 8, 'group_size': 4}
        run(cfg)
        run(cfg)
        budgets.append(json.loads((out / 'budget.json').read_text()))
        assert budgets[-1]['gradient_clip'] == (None if optimizer == 'sgd' else 1)
        assert json.loads((out / 'reload_check.json').read_text())['max_error'] == 0
    assert all(b['initial_kl'] == 0 and b['optimizer_steps'] == 2 for b in budgets)
