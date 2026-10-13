"""Native validation-only candidate screens with optional durable budget stopping."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tsflab.experiments.infra.comparison import protocol_fingerprint
from tsflab.experiments.infra.storage import canonical_hash, write_json


def compare_screen(baseline, candidate, *, metric='mse', min_improvement=0.0):
    """Require strict improvement on every supplied, protocol-matched seed/horizon."""
    if metric not in {'mse', 'mae'} or not math.isfinite(min_improvement) or not 0 <= min_improvement < 1:
        raise ValueError('screen requires mse/mae and 0 <= min_improvement < 1')
    groups = []
    for directories in (baseline, candidate):
        rows = {}
        variants = set()
        for directory in directories:
            directory = Path(directory).expanduser().resolve()
            manifest = json.loads((directory / 'manifest.json').read_text())
            config = manifest['config']
            if manifest['status'] != 'succeeded' or config['evaluation']['split'] != 'val' or config['evaluation'].get('checkpoint_path'):
                raise ValueError('screen requires successful, non-frozen validation runs')
            result = json.loads((directory / 'result.json').read_text())
            value = float(result['metrics'][metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError('screen metric must be finite and nonnegative')
            key = (config['experiment']['random_seed'], config['task']['pred_len'])
            if key in rows:
                raise ValueError('duplicate seed/horizon in screen panel')
            variants.add(canonical_hash(config['model']))
            rows[key] = {'run_id': manifest['run_id'], 'round': manifest.get('round'),
                         'directory': str(directory), 'value': value,
                         'protocol': protocol_fingerprint(config, manifest['data'])}
        if not rows or len(variants) != 1:
            raise ValueError('screen panel must contain exactly one model variant')
        groups.append(rows)
    reference, treatment = groups
    if reference.keys() != treatment.keys():
        raise ValueError('baseline and candidate seed/horizon coverage differs')
    comparisons = []
    for seed, horizon in sorted(reference):
        b, c = reference[seed, horizon], treatment[seed, horizon]
        if b['protocol'] != c['protocol']:
            raise ValueError('screen protocols or data fingerprints differ')
        comparisons.append({'seed': seed, 'pred_len': horizon, 'baseline': b, 'candidate': c,
                            'passed': c['value'] < b['value'] * (1 - min_improvement)})
    return {'passed': all(c['passed'] for c in comparisons), 'metric': metric,
            'min_improvement': min_improvement, 'comparisons': comparisons}


def _reference_identity(directory):
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    return canonical_hash({'manifest': manifest, 'result': result})


def register_references(round_id, directories):
    """Reuse saved validation baselines without claiming new training or hiding origin."""
    from tsflab.research.rounds import _round_dir, _state_lock, _write_json, load_round, add_event
    records = {}
    for directory in directories:
        # The ordinary screen reader enforces successful, non-frozen validation.
        row = compare_screen([directory], [directory])['comparisons'][0]['baseline']
        record = dict(row, identity=_reference_identity(row['directory']),
                      accounting='Reused validation artifact; original execution cost remains at origin')
        if row['run_id'] in records and records[row['run_id']] != record:
            raise ValueError('conflicting reference run identity')
        records[row['run_id']] = record
    if not records:
        raise ValueError('at least one validation reference is required')
    with _state_lock(round_id):
        state = load_round(round_id)
        if state['status'] != 'running':
            raise ValueError('research round is not running')
        references = state.setdefault('reference_runs', {})
        for run_id, record in records.items():
            if run_id in references and references[run_id] != record:
                raise ValueError('registered reference changed; preserve the original artifact')
        added = [run_id for run_id in records if run_id not in references]
        references.update(records)
        _write_json(_round_dir(round_id) / 'round.json', state)
    if added:
        add_event(round_id, 'observation', 'Reused native validation references',
                  details={'run_ids': added, 'new_training_claims': 0})
    return {'round': round_id, 'references': list(records.values()), 'new_training_claims': 0}


def screen_candidate(round_id, baseline, candidate, *, metric='mse', min_improvement=0.0, stop_on_fail=False):
    """Record a matched screen; an explicit failed stop closes future run claims."""
    from tsflab.research.rounds import _round_dir, _state_lock, _write_json, _now, load_round, add_event
    result = compare_screen(baseline, candidate, metric=metric, min_improvement=min_improvement)
    records = [row[side] for row in result['comparisons'] for side in ('baseline', 'candidate')]
    identity = canonical_hash({'records': records, 'metric': metric,
                               'min_improvement': min_improvement, 'stop_on_fail': stop_on_fail})
    path = _round_dir(round_id) / 'screen-gates' / (identity + '.json')
    with _state_lock(round_id):
        state = load_round(round_id)
        for comparison in result['comparisons']:
            c, b = comparison['candidate'], comparison['baseline']
            if c['round'] != round_id or c['run_id'] not in state.get('run_claims', {}):
                raise ValueError('candidate records must belong to the supplied research round claims')
            if b['round'] == round_id and b['run_id'] in state.get('run_claims', {}):
                continue
            reference = state.get('reference_runs', {}).get(b['run_id'])
            if not reference or reference['directory'] != b['directory'] or reference['identity'] != _reference_identity(b['directory']):
                raise ValueError('baseline must be a native round claim or an unchanged registered reference')
        if path.exists():
            return json.loads(path.read_text())
        if state['status'] != 'running':
            raise ValueError('research round is not running')
        stopped = stop_on_fail and not result['passed']
        if stopped and state.get('active_runs'):
            raise ValueError('wait for active runs before stopping a failed screen')
        result.update(directory=str(path), round=round_id, stopped=bool(stopped))
        if stopped:
            state.update(status='stopped', updated_at=_now())
            _write_json(_round_dir(round_id) / 'round.json', state)
        write_json(path, result)
    add_event(round_id, 'decision', 'Native validation screen passed' if result['passed'] else 'Native validation screen failed',
              details={'gate': str(path), 'stopped': bool(stopped)})
    return result
