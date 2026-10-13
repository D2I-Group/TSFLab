"""Screen failures must stop further spending without consulting test results."""
import json

import pytest

from tsflab.research.screening import compare_screen, screen_candidate


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from tsflab.research.rounds import create_round, claim_run
    monkeypatch.setenv('TSFLAB_WORK_DIR', str(tmp_path / 'work_dirs'))
    round_id = create_round(task='screen', goal='lower validation error', max_runs=10)['id']
    baseline, candidate = [], []
    for model, values, directories in [('baseline', [1., 2.], baseline), ('candidate', [.8, 2.1], candidate)]:
        for horizon, value in zip([96, 192], values):
            directory = tmp_path / f'{model}-{horizon}';directory.mkdir()
            config = {'model': {'name': model, 'params': {}}, 'dataset': {'name': 'custom'},
                      'task': {'pred_len': horizon, 'seq_len': 96},
                      'training': {'epochs': 10}, 'evaluation': {'split': 'val'},
                      'experiment': {'random_seed': 42}}
            manifest = {'run_id': directory.name, 'round': round_id, 'status': 'succeeded',
                        'config': config, 'data': {'sha256': 'same-data'}}
            (directory / 'manifest.json').write_text(json.dumps(manifest))
            (directory / 'result.json').write_text(json.dumps({'metrics': {'mse': value}}))
            claim_run(round_id, {'run_id': directory.name})
            directories.append(directory)
    return round_id, baseline, candidate


def test_one_worse_horizon_blocks_confirmation_and_closes_claim_budget(panel):
    from tsflab.research.rounds import load_round, claim_run
    round_id, baseline, candidate = panel
    result = screen_candidate(round_id, baseline, candidate, stop_on_fail=True)
    assert not result['passed'] and result['stopped']
    assert load_round(round_id)['status'] == 'stopped'
    with pytest.raises(ValueError, match='not running'):
        claim_run(round_id, {'run_id': 'unnecessary-confirmation'})
    assert screen_candidate(round_id, baseline, candidate, stop_on_fail=True) == result
    assert load_round(round_id)['runs_used'] == 4


def test_failed_first_screen_can_leave_budget_for_declared_second_candidate(panel):
    from tsflab.research.rounds import load_round
    round_id, baseline, candidate = panel
    result = screen_candidate(round_id, baseline, candidate)
    assert not result['passed'] and not result['stopped']
    assert load_round(round_id)['status'] == 'running'


def test_all_horizon_gain_passes_and_keeps_round_open(panel):
    from tsflab.research.rounds import load_round
    round_id, baseline, candidate = panel
    (candidate[1] / 'result.json').write_text(json.dumps({'metrics': {'mse': 1.5}}))
    result = screen_candidate(round_id, baseline, candidate, min_improvement=.1, stop_on_fail=True)
    assert result['passed'] and not result['stopped']
    assert load_round(round_id)['status'] == 'running'
    assert not compare_screen(baseline, candidate, min_improvement=.3)['passed']


@pytest.mark.parametrize('change', ['test', 'frozen', 'training', 'data', 'variant', 'seed'])
def test_screen_rejects_leakage_and_incomparable_panels(panel, change):
    _, baseline, candidate = panel
    path = candidate[1] / 'manifest.json';m = json.loads(path.read_text())
    if change == 'test':m['config']['evaluation']['split'] = 'test'
    elif change == 'frozen':m['config']['evaluation']['checkpoint_path'] = 'winner.pth'
    elif change == 'training':m['config']['training']['epochs'] = 100
    elif change == 'data':m['data']['sha256'] = 'different-data'
    elif change == 'variant':m['config']['model']['params']['hidden_size'] = 99
    elif change == 'seed':m['config']['experiment']['random_seed'] = 43
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):compare_screen(baseline, candidate)


def test_incomplete_duplicate_and_nonfinite_panels_are_rejected(panel):
    _, baseline, candidate = panel
    with pytest.raises(ValueError):compare_screen(baseline, candidate[:1])
    with pytest.raises(ValueError):compare_screen(baseline, candidate + candidate[:1])
    (candidate[0] / 'result.json').write_text(json.dumps({'metrics': {'mse': float('nan')}}))
    with pytest.raises(ValueError):compare_screen(baseline, candidate)


def test_gate_cli_returns_failed_screen_without_spending_more_runs(panel, capsys):
    from tsflab.cli.commands.research import research_command
    round_id, baseline, candidate = panel
    assert research_command(['screen', round_id, '--baseline', *map(str, baseline),
                             '--candidate', *map(str, candidate), '--stop-on-fail']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['stopped'] and not result['passed']


def _new_round_with_candidates(panel):
    from tsflab.research.rounds import create_round, claim_run
    _, baseline, candidate = panel
    new = create_round(task='reuse', goal='improve shared baselines', max_runs=2)['id']
    for directory in candidate:
        path = directory / 'manifest.json'
        m = json.loads(path.read_text());m['round'] = new;path.write_text(json.dumps(m))
        claim_run(new, {'run_id': m['run_id']})
    return new, baseline, candidate


def test_shared_validation_references_do_not_retrain_or_mutate_origin(panel):
    from tsflab.research.rounds import load_round
    from tsflab.research.screening import register_references
    round_id, baseline, candidate = _new_round_with_candidates(panel)
    originals = [(p / 'manifest.json').read_bytes() for p in baseline]
    with pytest.raises(ValueError, match='registered reference'):
        screen_candidate(round_id, baseline, candidate)
    first = register_references(round_id, baseline)
    assert register_references(round_id, baseline) == first
    assert load_round(round_id)['runs_used'] == 2
    assert first['new_training_claims'] == 0
    assert len(load_round(round_id)['reference_runs']) == 2
    assert not screen_candidate(round_id, baseline, candidate, stop_on_fail=True)['passed']
    assert load_round(round_id)['status'] == 'stopped'
    assert originals == [(p / 'manifest.json').read_bytes() for p in baseline]


@pytest.mark.parametrize('file', ['manifest.json', 'result.json'])
def test_changed_reference_is_rejected_instead_of_silently_reused(panel, file):
    from tsflab.research.screening import register_references
    round_id, baseline, candidate = _new_round_with_candidates(panel)
    register_references(round_id, baseline)
    path = baseline[0] / file;data = json.loads(path.read_text())
    if file == 'result.json':data['metrics']['mse'] = 100.0
    else:data['extra_origin_fact'] = 'changed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unchanged registered reference'):
        screen_candidate(round_id, baseline, candidate)
    with pytest.raises(ValueError, match='registered reference changed'):
        register_references(round_id, baseline)


def test_test_and_frozen_records_cannot_be_registered_as_search_references(panel):
    from tsflab.research.screening import register_references
    round_id, baseline, _ = panel
    path = baseline[0] / 'manifest.json';m = json.loads(path.read_text())
    for field, value in [('split', 'test'), ('checkpoint_path', 'frozen.pth')]:
        m['config']['evaluation'] = {'split': 'val', field: value}
        path.write_text(json.dumps(m))
        with pytest.raises(ValueError, match='non-frozen validation'):
            register_references(round_id, baseline)


def test_reference_cli_and_candidate_claim_guard(panel, capsys):
    from tsflab.cli.commands.research import research_command
    from tsflab.research.screening import register_references
    from tsflab.research.rounds import create_round
    new, baseline, candidate = _new_round_with_candidates(panel)
    assert research_command(['reference', new, *map(str, baseline)]) == 0
    assert json.loads(capsys.readouterr().out)['new_training_claims'] == 0
    other = create_round(task='unclaimed', goal='must not accept free candidate', max_runs=1)['id']
    register_references(other, baseline + candidate)
    with pytest.raises(ValueError, match='candidate records'):
        screen_candidate(other, baseline, candidate)
