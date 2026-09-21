"""Conditional losses and finite-policy diagnostics for the separate v3 protocol."""

from collections import defaultdict
import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from .upgrade_data import signatures


def interaction(scores):
    if scores.shape != (4, 2):
        raise ValueError('Crossed scores must be four cells by two programs')
    d = scores[:, 0] - scores[:, 1]
    return d[0] - d[1] - d[2] + d[3]


def assignment_cycles(scores):
    if scores.shape != (4, 4):
        raise ValueError('Four-TARGET scores must be four cells by four programs')
    return torch.stack([scores[i, i] + scores[j, j] - scores[i, j] - scores[j, i]
                        for i, j in itertools.combinations(range(4), 2)])


def conditional_loss(scores, kind, tau=1.):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    values = interaction(scores) if kind == 'crossed' else assignment_cycles(scores)
    return F.softplus(-values / tau).mean()


def entropy(logq):
    return -(logq.exp() * logq).sum(-1)


def entropy_report(scores, correct, programs, p, degree, temperatures=(.5, 1., 2.), affine=True):
    scores, correct = np.asarray(scores, dtype=float), np.asarray(correct, dtype=bool)
    if scores.shape != correct.shape or scores.shape != (len(programs),) or not np.isfinite(scores).all():
        raise ValueError('Expected finite complete program scores and matching labels')
    maps = signatures(p, degree, len(programs[0])) if affine else None
    classes = defaultdict(list)
    for index, program in enumerate(programs):
        if maps is not None:
            classes[maps[tuple(program)]].append(index)
    def h(q):
        return float(-sum(x * math.log(x) for x in q if x > 0))
    rank = sorted(range(len(scores)), key=lambda i: (-scores[i], tuple(programs[i])))
    first = next((i + 1 for i, index in enumerate(rank) if correct[index]), len(scores) + 1)
    reports = []
    for temperature in temperatures:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Program-score temperature must be finite and positive')
        q = np.exp((scores - scores.max()) / temperature)
        q /= q.sum()
        mass = float(q[correct].sum())
        good = h(q[correct] / mass) if mass > 0 else 0.
        bad = h(q[~correct] / (1 - mass)) if mass < 1 else 0.
        binary = h([mass, 1 - mass])
        function_mass = [float(q[indices].sum()) for indices in classes.values()]
        function_entropy = h(function_mass)
        conditional = sum(weight * h(q[indices] / weight) for weight, indices in zip(function_mass, classes.values()) if weight > 0)
        total = h(q)
        if not np.isclose(total, binary + mass * good + (1 - mass) * bad, atol=1e-10):
            raise ValueError('Correct/incorrect entropy identity failed')
        if affine and not np.isclose(total, function_entropy + conditional, atol=1e-10):
            raise ValueError('Program/function entropy identity failed')
        reports.append({'temperature': temperature, 'correct_mass': mass, 'entropy': total,
            'binary_entropy': binary, 'correct_entropy': good, 'incorrect_entropy': bad,
            'function_entropy': function_entropy, 'program_given_function_entropy': conditional,
            'semantic_classes': len(classes), 'hit1': int(first <= 1), 'hit8': int(first <= 8),
            'hit32': int(first <= 32), 'first_correct_rank': first})
        if not affine:
            for key in ('function_entropy', 'program_given_function_entropy', 'semantic_classes'):
                reports[-1].pop(key)
    return reports


def sigreg(representations, state_keys, step, slices=256):
    """LeJEPA Epps-Pulley statistic on unit slices; one mean per repeated state.

    Formula: https://arxiv.org/html/2511.08544v1#S4.SS2.SSS3 .
    This small projection regularizer does not import LeJEPA's task guarantees.
    """
    if representations.ndim != 2 or len(state_keys) != len(representations) or slices < 1:
        raise ValueError('Expected a matrix and one exact state identity per row')
    groups = defaultdict(list)
    for index, key in enumerate(state_keys):
        groups[key].append(index)
    x = torch.stack([representations[indices].mean(0) for indices in groups.values()]).float()
    if len(x) < 2:
        return x.sum() * 0
    generator = torch.Generator(device=x.device).manual_seed(step)
    directions = F.normalize(torch.randn(x.shape[1], slices, generator=generator, device=x.device), dim=0)
    t = torch.linspace(-5, 5, 17, device=x.device)
    phase = (x @ directions).unsqueeze(-1) * t
    target = (-t.square() / 2).exp()
    discrepancy = (phase.cos().mean(0) - target).square() + phase.sin().mean(0).square()
    return (torch.trapezoid(discrepancy * target, t, dim=-1) * len(x)).mean()


class StateProjection(torch.nn.Module):
    """Small state representation has a causal path into action logits."""
    def __init__(self, hidden_size, dimension=32, max_coefficients=5, max_field=29):
        super().__init__()
        self.config = dict(hidden_size=hidden_size, dimension=dimension,
                           max_coefficients=max_coefficients, max_field=max_field)
        self.project = torch.nn.Linear(hidden_size, dimension)
        self.action = torch.nn.Linear(dimension, 5, bias=False)
        self.state = torch.nn.Linear(dimension, max_coefficients * max_field)
        torch.nn.init.zeros_(self.action.weight)

    def forward(self, hidden, ablate=False):
        z = self.project(hidden.float())
        return z, self.action(torch.zeros_like(z) if ablate else z)

    def state_loss(self, z, states, fields):
        logits = self.state(z).reshape(len(z), self.config['max_coefficients'], self.config['max_field'])
        losses, correct, total = [], 0, 0
        for row, state, p in zip(logits, states, fields):
            if not 2 <= p <= self.config['max_field'] or len(state) > self.config['max_coefficients']:
                raise ValueError('State projection does not cover this field/dimension')
            target = torch.tensor(state, device=z.device)
            value = row[:len(state), :p]
            losses.append(F.cross_entropy(value, target))
            correct += int((value.argmax(-1) == target).sum())
            total += len(state)
        return torch.stack(losses).mean(), {'correct_coefficients': correct, 'coefficients': total}
