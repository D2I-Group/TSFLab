"""Select bounded independent review after deterministic experimental checks.

Usage: review_gate.py completion.json [--implementation-changed] [--suspect REASON]
Returns a decision only; a healthy reused implementation incurs no reviewer call.
"""
import argparse
import json
import math
from pathlib import Path

SUCCESS = {'succeeded', 'success', 'complete', 'completed'}


def review_decision(result, *, implementation_changed=False, suspects=()):
    reasons = list(suspects)
    if isinstance(result, list):
        result = {"cells": result}
    cells = result.get('runs', result.get('cells', result.get('training_cells', [])))
    if not isinstance(cells, list):
        cells = []
    if not cells:
        reasons.append('no completed experimental cells')
    for index, cell in enumerate(cells):
        if 'status' in cell and cell['status'] not in SUCCESS:
            reasons.append(f'cell {index}: unsuccessful status')
        metrics = cell.get('metrics', cell)
        for key in ['mse', 'mae']:
            value = metrics.get(key, cell.get('validation_'+key))
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                reasons.append(f'cell {index}: missing or nonfinite {key}')
    if result.get('deviations'):
        reasons.append('reported protocol deviations')
    if implementation_changed:
        reasons.append('new or changed model/training implementation')
    return {'review_required': bool(reasons), 'reasons': sorted(set(reasons)),
            'max_review_turns': 4, 'max_evidence_bytes': 65536, 'max_output_tokens': 2000,
            'policy': 'deterministic checks first; independent review for new implementations or suspect outcomes; charge review usage separately and in method-plus-review totals'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('completion')
    parser.add_argument('--implementation-changed', action='store_true')
    parser.add_argument('--suspect', action='append', default=[])
    args = parser.parse_args()
    result = json.loads(Path(args.completion).read_text())
    print(json.dumps(review_decision(result, implementation_changed=args.implementation_changed,
                                     suspects=args.suspect), separators=(',', ':')))


if __name__ == '__main__':
    main()
