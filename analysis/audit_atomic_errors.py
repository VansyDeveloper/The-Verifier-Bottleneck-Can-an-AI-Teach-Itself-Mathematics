"""Recheck archived atomic labels/parser, strata and raw errors without loading weights."""

import argparse
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iclr
import composition_core as core
from iclr.common import file_hash, verify_receipt, write_json


def audit(runs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    strata, examples, sources = [], [], {}
    label_errors = parser_errors = 0
    for directory in map(Path, runs):
        receipt = verify_receipt(directory)
        sources[directory.name] = {'receipt_sha256': file_hash(directory / 'DONE'), 'binding': receipt['binding']}
        result = json.loads((directory / 'atomic_retention.json').read_text())
        counts = defaultdict(lambda: [0,0,0])
        selected = defaultdict(int)
        for row in result['apply']:
            op = row['witness'][0]
            expected = list(core.trajectory(row['start'], [op], row['p'])[-1])
            label_errors += expected != row['target']
            parsed = None
            try:
                value = ast.literal_eval(row['raw_completion'].strip())
                if isinstance(value, list) and len(value) == len(expected) and all(type(v) is int for v in value):
                    parsed = value
            except (ValueError, SyntaxError, RecursionError):
                pass
            correct = parsed is not None and [v % row['p'] for v in parsed] == expected
            parser_errors += row['parse_ok'] != (parsed is not None) or correct != row['semantic_correct']
            key = op, row['p'], len(row['start'])-1
            counts[key][0] += 1; counts[key][1] += int(parsed is not None); counts[key][2] += int(correct)
            if op in ('SH1','SC2') and not correct and selected[op] < 10 and directory.name.endswith('initial'):
                selected[op] += 1
                examples.append({'run': directory.name, 'task_id': row['task_id'], 'operation': op, 'p': row['p'],
                    'degree': len(row['start'])-1, 'prompt': core.apply_prompt({**row, 'program': [op]}),
                    'start': row['start'], 'expected': expected, 'raw_answer': row['raw_completion'], 'parsed': parsed, 'correct': correct,
                    'reversed_answer_matches': parsed is not None and [v % row['p'] for v in parsed[::-1]] == expected})
        strata.extend({'run': directory.name, 'operation': op, 'field': p, 'degree': d, 'n': n,
                       'parse_rate': parsed/n, 'accuracy': correct/n} for (op,p,d),(n,parsed,correct) in sorted(counts.items()))
    with (out / 'operation_field_degree.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(strata[0])); writer.writeheader(); writer.writerows(strata)
    write_json(out / 'errors.json', examples)
    write_json(out / 'audit.json', {'label_mismatches': label_errors, 'parser_or_accuracy_mismatches': parser_errors,
        'examples': len(examples), 'sources': sources, 'coefficient_order': 'ascending: constant coefficient first',
        'prompt_check': 'canonical APPLY prompt reconstructed from saved rows; no evidence here for the original base-training tokenizer or prompts',
        'token_ids_check': 'NOT VERIFIED against original atomic training: its receipt/tokenizer snapshot was not supplied',
        'interpretation': 'valid canonical labels and repeatable evaluator outcomes do not establish the cause of model errors'})
    if label_errors or parser_errors:
        raise ValueError('Atomic audit differs from the saved evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    audit(args.runs, args.out)
