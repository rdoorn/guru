"""Cross-day ledger aggregation for ``bench/ledger_report.py``.

Pure functions over lists of ledger rows (as returned by
``JsonlLedger.rows``): which models are called how often, task latency
percentiles per (kind, complexity), fallback/retry rates, and how the shadow
judges agree with the heuristics and with the user's labels. The thin bench
script loads the streams and prints :func:`render_markdown`.

Agreement semantics: ``judge vs heuristic`` uses each decision row's
``agree`` field (judge's ``chosen`` == the heuristic's answer). ``judge vs
labels`` is a cross-tab, not a score: for every decision row that was
labelled (see below; the last label per target wins) it counts, per
decision point and label, how often the judge chose True/False/undecided
and how often it agreed with the heuristic. Whether ``chosen=True`` is the
"good" answer depends on the point, so the reader judges the cross-tab.

Two label granularities share the ``labels`` stream:

* turn/task labels (``/good``, ``/bad``): ``target_id`` is a ``turn_id`` or
  ``task_id``, label ``good``/``bad`` — a verdict on the outcome;
* decision labels (``guru.ledger_cli review``): ``target_id`` is
  :func:`decision_key` = ``<point>:<question>:<input_sha>``, label ``yes``
  / ``no`` — the *correct answer* to that question for that input. Rows
  with the same input share the key, so one label covers repeats.

:func:`judge_metrics` scores judge and heuristic against the decision
labels (precision / recall / F1 / false-positive rate, positive = ``yes``)
and suggests the P(yes) threshold maximising F1 on the labelled rows.
"""
from __future__ import annotations

import math
from typing import Optional

from guru.domain import ledger

_NA = 'n/a'
REVIEW_LABELS = ('yes', 'no')
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05..0.95


def finish_rows(tasks_rows: list) -> list:
    """The latest non-``running`` row per task id, in first-seen order."""
    latest: dict = {}
    for r in tasks_rows:
        if r.get('status') == 'running':
            continue
        latest[r.get('task_id')] = r
    return list(latest.values())


def percentile(values: list, q: float) -> Optional[float]:
    """Nearest-rank percentile ``q`` (0-100) of ``values``; None if empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


UNLABELLED = 'unlabelled'


def latency_by_kind(tasks_rows: list) -> dict:
    """``{(kind, complexity): {'n', 'p50', 'p95'}}`` over finished tasks
    that report ``seconds``. Rows without a ``kind``/``complexity`` (written
    before routing labelled tasks) bucket as ``unlabelled`` rather than
    masquerading as the defaults."""
    groups: dict = {}
    for r in finish_rows(tasks_rows):
        secs = r.get('seconds')
        if secs is None:
            continue
        key = (r.get('kind') or UNLABELLED,
               r.get('complexity') or UNLABELLED)
        groups.setdefault(key, []).append(float(secs))
    return {k: {'n': len(v), 'p50': percentile(v, 50),
                'p95': percentile(v, 95)}
            for k, v in sorted(groups.items())}


def fallback_retry(tasks_rows: list) -> dict:
    """Fallback and retry counts and rates over finished tasks.

    ``fell_back`` counts ``status == 'fell_back'`` (status always exists, so
    the rate is 0.0 until routing produces fallbacks). ``retries`` counts
    rows with a non-empty ``retry_of`` and is None (reported ``n/a``) while
    no row carries the field at all (it arrives with phase 4).
    """
    fin = finish_rows(tasks_rows)
    n = len(fin)
    fell = sum(1 for r in fin if r.get('status') == 'fell_back')
    has_retry_field = any('retry_of' in r for r in fin)
    retries: Optional[int] = (
        sum(1 for r in fin if r.get('retry_of')) if has_retry_field else None)
    return {'tasks': n, 'fell_back': fell,
            'fell_back_rate': fell / n if n else None,
            'retries': retries,
            'retry_rate': (retries / n if (n and retries is not None)
                           else None)}


def judge_vs_heuristic(decisions_rows: list) -> dict:
    """Per decision point: rows, agree/disagree/undecided counts and the
    agreement rate over decided rows (None when none were decided)."""
    out: dict = {}
    for r in decisions_rows:
        p = out.setdefault(r.get('point') or '?', {
            'n': 0, 'agree': 0, 'disagree': 0, 'undecided': 0, 'rate': None})
        p['n'] += 1
        agree = r.get('agree')
        if agree is None:
            p['undecided'] += 1
        elif agree:
            p['agree'] += 1
        else:
            p['disagree'] += 1
    for p in out.values():
        decided = p['agree'] + p['disagree']
        p['rate'] = p['agree'] / decided if decided else None
    return dict(sorted(out.items()))


def decision_key(row: dict) -> str:
    """``<point>:<question>:<input_sha>``: the ``labels`` target id for one
    decision (identical inputs share it)."""
    return (f"{row.get('point') or ''}:{row.get('question') or ''}:"
            f"{row.get('input_sha') or ''}")


def review_note(row: dict) -> str:
    """The ``note`` a review label carries: ``point:<p>;question:<q>``."""
    return (f"point:{row.get('point') or ''};"
            f"question:{row.get('question') or ''}")


def _label_index(labels_rows: list) -> dict:
    """``target_id -> label`` with the last label per target winning."""
    return {r.get('target_id'): r.get('label') for r in labels_rows
            if r.get('target_id')}


def _row_label(r: dict, labels: dict) -> Optional[str]:
    """The label for a decision row: its own decision label first, then its
    task's, then its turn's."""
    return (labels.get(decision_key(r)) or labels.get(r.get('task_id'))
            or labels.get(r.get('turn_id')))


def judge_vs_labels(decisions_rows: list, labels_rows: list) -> dict:
    """Cross-tab ``{point: {label: {'n', 'chosen_true', 'chosen_false',
    'undecided', 'agree_heuristic'}}}`` over labelled decision rows
    (decision label preferred, then task, then turn). See the module
    docstring."""
    labels = _label_index(labels_rows)
    out: dict = {}
    for r in decisions_rows:
        label = _row_label(r, labels)
        if not label:
            continue
        cell = out.setdefault(r.get('point') or '?', {}).setdefault(label, {
            'n': 0, 'chosen_true': 0, 'chosen_false': 0, 'undecided': 0,
            'agree_heuristic': 0})
        cell['n'] += 1
        chosen = r.get('chosen')
        if chosen is None:
            cell['undecided'] += 1
        elif chosen:
            cell['chosen_true'] += 1
        else:
            cell['chosen_false'] += 1
        if r.get('agree'):
            cell['agree_heuristic'] += 1
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


# --- review loop: decision labels, precision/recall, thresholds -------------

def decision_labels(labels_rows: list) -> dict:
    """``decision_key -> True/False`` from ``yes``/``no`` labels (last label
    per key wins); other labels are ignored."""
    out: dict = {}
    for r in labels_rows:
        target, label = r.get('target_id'), r.get('label')
        if target and label in REVIEW_LABELS:
            out[target] = label == 'yes'
    return out


def prf(*, tp: int, fp: int, fn: int, tn: int) -> dict:
    """Precision, recall, F1 and false-positive rate (None when the
    denominator is zero) plus the four counts."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if (precision and recall) else None)
    if f1 is None and precision is not None and recall is not None:
        f1 = 0.0
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision,
            'recall': recall, 'f1': f1, 'fpr': fpr}


def _score(pairs: list) -> dict:
    """:func:`prf` over ``(predicted, truth)`` bool pairs."""
    tp = sum(1 for p, t in pairs if p and t)
    fp = sum(1 for p, t in pairs if p and not t)
    fn = sum(1 for p, t in pairs if not p and t)
    tn = sum(1 for p, t in pairs if not p and not t)
    return prf(tp=tp, fp=fp, fn=fn, tn=tn)


def suggest_threshold(pairs: list) -> Optional[dict]:
    """The grid threshold (0.05..0.95) maximising F1 over ``(p_yes, truth)``
    pairs, ties broken towards 0.5. ``{'threshold', 'f1', 'n'}`` or None
    when there are no pairs."""
    if not pairs:
        return None
    best: Optional[tuple] = None
    for t in THRESHOLD_GRID:
        f1 = _score([(p >= t, truth) for p, truth in pairs])['f1'] or 0.0
        rank = (f1, -abs(t - 0.5))
        if best is None or rank > best[0]:
            best = (rank, t, f1)
    assert best is not None
    return {'threshold': best[1], 'f1': best[2], 'n': len(pairs)}


def _p_yes(row: dict) -> Optional[float]:
    """``dist['yes']`` as a float, or None when the row has none."""
    dist = row.get('dist')
    p = dist.get('yes') if isinstance(dist, dict) else None
    return float(p) if isinstance(p, (int, float)) else None


def judge_metrics(decisions_rows: list, labels_rows: list,
                  point: Optional[str] = None) -> dict:
    """``{point: {judge: metrics}}`` for the review loop.

    ``metrics``: ``rows``, ``used`` (``{'judge': n, 'heuristic': n}``),
    ``fallbacks`` (``{reason: n}``), ``agree_rate`` (judge vs heuristic
    over decided rows), ``labelled`` (rows with a decision label),
    ``judge`` and ``heuristic`` (:func:`prf` vs the labels, plus
    ``undecided`` rows excluded from the counts), ``threshold``
    (:func:`suggest_threshold` over labelled rows with ``dist['yes']``, or
    None) and ``queued_ms`` (``{'n', 'p50', 'p95'}`` of the rows' worker
    queue wait; a high value with a normal judge ``ms`` means the worker
    was busy, not the judge slow).
    """
    truth = decision_labels(labels_rows)
    groups: dict = {}
    for r in decisions_rows:
        pt = r.get('point') or '?'
        if point is not None and pt != point:
            continue
        groups.setdefault(pt, {}).setdefault(r.get('judge') or '?',
                                             []).append(r)
    out: dict = {}
    for pt, judges in sorted(groups.items()):
        out[pt] = {}
        for judge, rows in sorted(judges.items()):
            used: dict = {'judge': 0, 'heuristic': 0}
            fallbacks: dict = {}
            agree = disagree = 0
            judge_pairs: list = []
            heur_pairs: list = []
            thr_pairs: list = []
            undecided = {'judge': 0, 'heuristic': 0}
            labelled = 0
            queued: list = []
            for r in rows:
                if isinstance(r.get('queued_ms'), (int, float)):
                    queued.append(float(r['queued_ms']))
                u = r.get('used') or 'heuristic'
                used[u] = used.get(u, 0) + 1
                if r.get('fallback_reason'):
                    fallbacks[r['fallback_reason']] = fallbacks.get(
                        r['fallback_reason'], 0) + 1
                if r.get('agree') is True:
                    agree += 1
                elif r.get('agree') is False:
                    disagree += 1
                key = decision_key(r)
                if key not in truth:
                    continue
                labelled += 1
                t = truth[key]
                if r.get('chosen') is None:
                    undecided['judge'] += 1
                else:
                    judge_pairs.append((bool(r['chosen']), t))
                if r.get('heuristic') is None:
                    undecided['heuristic'] += 1
                else:
                    heur_pairs.append((bool(r['heuristic']), t))
                p = _p_yes(r)
                if p is not None:
                    thr_pairs.append((p, t))
            decided = agree + disagree
            out[pt][judge] = {
                'rows': len(rows), 'used': used, 'fallbacks': fallbacks,
                'agree_rate': agree / decided if decided else None,
                'labelled': labelled,
                'judge': {**_score(judge_pairs),
                          'undecided': undecided['judge']},
                'heuristic': {**_score(heur_pairs),
                              'undecided': undecided['heuristic']},
                'threshold': suggest_threshold(thr_pairs),
                'queued_ms': {'n': len(queued), 'p50': percentile(queued, 50),
                              'p95': percentile(queued, 95)}}
    return out


def review_queue(decisions_rows: list, labels_rows: list, *, point: str,
                 n: int) -> list:
    """The ``n`` most recent decision rows for ``point`` that carry an
    ``input_sha`` and have no decision label yet, newest first, one row
    per :func:`decision_key` (the newest)."""
    labelled = decision_labels(labels_rows)
    newest: dict = {}
    for r in decisions_rows:
        if r.get('point') != point or not r.get('input_sha'):
            continue
        key = decision_key(r)
        if key in labelled:
            continue
        if key not in newest or str(r.get('ts') or '') >= str(
                newest[key].get('ts') or ''):
            newest[key] = r
    rows = sorted(newest.values(), key=lambda r: str(r.get('ts') or ''),
                  reverse=True)
    return rows[:n]


def unlabelled_tasks(tasks_rows: list, labels_rows: list, *,
                     n: int) -> list:
    """The ``n`` most recent finished tasks without any label on their
    ``task_id``, newest first, flattened for triage: ``task_id``, ``task``
    (whitespace-collapsed), ``route`` (``adapter|model``), ``reason``,
    ``kind``, ``complexity``, ``status``, ``seconds``, ``cost_usd``,
    ``transcript_path``, ``turn_id``, ``ts``."""
    labelled = {r.get('target_id') for r in labels_rows if r.get('target_id')}
    fin = [r for r in finish_rows(tasks_rows)
           if r.get('task_id') and r['task_id'] not in labelled]
    fin.sort(key=lambda r: str(r.get('ts') or ''), reverse=True)
    return [{'task_id': r['task_id'],
             'task': ' '.join(str(r.get('task') or '').split()),
             'route': ledger.model_key(r),
             'reason': list(r.get('reason') or []),
             'kind': r.get('kind') or '',
             'complexity': r.get('complexity') or '',
             'status': r.get('status') or '', 'seconds': r.get('seconds'),
             'cost_usd': r.get('cost_usd'),
             'transcript_path': r.get('transcript_path') or '',
             'turn_id': r.get('turn_id') or '', 'ts': r.get('ts') or ''}
            for r in fin[:n]]


def turns_summary(turns_rows: list) -> dict:
    """Turn count, total cost (None if any unknown), seconds p50/p95, how
    many the controller executed itself and tasks spawned in total."""
    secs = [float(r['seconds']) for r in turns_rows
            if r.get('seconds') is not None]
    cost = ledger.sum_cost([r.get('cost_usd') for r in turns_rows])
    return {'n': len(turns_rows), 'cost_usd': cost,
            'seconds_p50': percentile(secs, 50),
            'seconds_p95': percentile(secs, 95),
            'controller_executed': sum(
                1 for r in turns_rows if r.get('controller_executed')),
            'tasks_spawned': sum(int(r.get('tasks_spawned') or 0)
                                 for r in turns_rows)}


def controller_labelled(tasks_rows: list) -> bool:
    """True once any task row carries a controller ``kind`` (routing
    writes it); until then controller-vs-judge agreement is ``n/a``."""
    return any(r.get('kind') for r in tasks_rows)


def build_report(*, calls: list, tasks: list, turns: list, decisions: list,
                 labels: list) -> dict:
    """Every aggregation in one dict (keys: ``models``, ``latency``,
    ``fallback``, ``judge_vs_heuristic``, ``judge_vs_labels``,
    ``judge_metrics``, ``turns``, ``controller_labelled``, ``counts``)."""
    return {'models': ledger.per_model_usage(calls),
            'controller_labelled': controller_labelled(tasks),
            'latency': latency_by_kind(tasks),
            'fallback': fallback_retry(tasks),
            'judge_vs_heuristic': judge_vs_heuristic(decisions),
            'judge_vs_labels': judge_vs_labels(decisions, labels),
            'judge_metrics': judge_metrics(decisions, labels),
            'turns': turns_summary(turns),
            'counts': {'calls': len(calls), 'tasks': len(tasks),
                       'turns': len(turns), 'decisions': len(decisions),
                       'labels': len(labels)}}


# --- Markdown ----------------------------------------------------------------

def _money(v: Optional[float]) -> str:
    """``$0.1234`` or ``$?`` for an unknown cost."""
    return '$?' if v is None else f'${v:.4f}'


def _secs(v: Optional[float]) -> str:
    """Seconds with two decimals, or ``n/a``."""
    return _NA if v is None else f'{v:.2f}'


def _rate(v: Optional[float]) -> str:
    """A 0..1 ratio as a whole percentage, or ``n/a``."""
    return _NA if v is None else f'{100 * v:.0f}%'


def _ms(v: Optional[float]) -> str:
    """Milliseconds as an integer, or ``n/a``."""
    return _NA if v is None else f'{v:.0f}'


def _table(header: list, rows: list) -> list:
    """Markdown table lines (``(no rows)`` when empty)."""
    if not rows:
        return ['(no rows)', '']
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return lines + ['']


def render_metrics(metrics: dict) -> str:
    """Plain-text rendering of :func:`judge_metrics` (one block per point
    and judge) for ``guru.ledger_cli report``."""
    if not metrics:
        return 'no decision rows\n'
    out: list = []
    for point, judges in metrics.items():
        for judge, m in judges.items():
            used = m['used']
            fb = ', '.join(f'{k}={v}' for k, v in sorted(
                m['fallbacks'].items())) or 'none'
            out += [f'{point} / {judge}',
                    f"  rows {m['rows']}  used judge={used.get('judge', 0)}"
                    f" heuristic={used.get('heuristic', 0)}  fallbacks {fb}",
                    f"  agreement with heuristic {_rate(m['agree_rate'])}",
                    f"  queued_ms p50 {_ms(m['queued_ms']['p50'])}  p95 "
                    f"{_ms(m['queued_ms']['p95'])}  (n {m['queued_ms']['n']})",
                    f"  labelled {m['labelled']}"]
            for who in ('judge', 'heuristic'):
                s = m[who]
                out.append(
                    f"  {who:9s} precision {_rate(s['precision'])}  recall "
                    f"{_rate(s['recall'])}  f1 {_rate(s['f1'])}  fpr "
                    f"{_rate(s['fpr'])}  (tp {s['tp']} fp {s['fp']} fn "
                    f"{s['fn']} tn {s['tn']} undecided {s['undecided']})")
            thr = m['threshold']
            out.append(
                '  threshold  n/a (no labelled rows with dist)' if thr is None
                else f"  threshold  suggested {thr['threshold']:.2f} (f1 "
                     f"{_rate(thr['f1'])} on {thr['n']} rows)")
            out.append('')
    return '\n'.join(out).rstrip() + '\n'


def render_markdown(report: dict) -> str:
    """Render :func:`build_report`'s dict as Markdown."""
    c = report['counts']
    out = ['# Ledger report', '',
           f"rows: {c['calls']} calls, {c['tasks']} tasks, {c['turns']} "
           f"turns, {c['decisions']} decisions, {c['labels']} labels", '']
    out += ['## Calls per model', '']
    out += _table(
        ['adapter', 'model', 'calls', 'tokens in', 'tokens out',
         'cache read', 'cache write', 'cost'],
        [[m['adapter'], m['model'], m['calls'], m['tokens_in'],
          m['tokens_out'], m['cache_read'], m['cache_write'],
          _money(m['cost_usd'])]
         for _, m in sorted(report['models'].items(),
                            key=lambda kv: -kv[1]['calls'])])
    out += ['## Task latency (seconds, finished tasks; rows without '
            f'kind/complexity bucket as `{UNLABELLED}`)', '']
    out += _table(['kind', 'complexity', 'n', 'p50', 'p95'],
                  [[k, cx, v['n'], _secs(v['p50']), _secs(v['p95'])]
                   for (k, cx), v in report['latency'].items()])
    fb = report['fallback']
    out += ['## Fallbacks and retries', '']
    out += _table(['finished tasks', 'fell back', 'rate', 'retries', 'rate'],
                  [[fb['tasks'], fb['fell_back'], _rate(fb['fell_back_rate']),
                    _NA if fb['retries'] is None else fb['retries'],
                    _rate(fb['retry_rate'])]] if fb['tasks'] else [])
    out += ['## Judge vs heuristic', '']
    if not report.get('controller_labelled'):
        out += ['Controller vs judge agreement: n/a (fields arrive with '
                'routing)', '']
    out += _table(['point', 'n', 'agree', 'disagree', 'undecided', 'rate'],
                  [[p, v['n'], v['agree'], v['disagree'], v['undecided'],
                    _rate(v['rate'])]
                   for p, v in report['judge_vs_heuristic'].items()])
    out += ['## Judge vs labels', '',
            'Labelled decision rows, joined on the decision key '
            '(`<point>:<question>:<input_sha>`, from `guru.ledger_cli '
            'review`) first, then on the task id, then on the turn id '
            '(`/good`, `/bad`); `agree` = judge agreed with the heuristic.',
            '']
    out += _table(['point', 'label', 'n', 'chosen True', 'chosen False',
                   'undecided', 'agree'],
                  [[p, label, v['n'], v['chosen_true'], v['chosen_false'],
                    v['undecided'], v['agree_heuristic']]
                   for p, cells in report['judge_vs_labels'].items()
                   for label, v in cells.items()])
    out += ['## Judge vs review labels', '',
            'Decision rows labelled with `guru.ledger_cli review` '
            '(positive = `yes`).', '']
    out += _table(['point', 'judge', 'rows', 'labelled', 'judge P/R/F1',
                   'heuristic P/R/F1', 'suggested threshold'],
                  [[p, j, m['rows'], m['labelled'],
                    '/'.join(_rate(m['judge'][k])
                             for k in ('precision', 'recall', 'f1')),
                    '/'.join(_rate(m['heuristic'][k])
                             for k in ('precision', 'recall', 'f1')),
                    _NA if m['threshold'] is None
                    else f"{m['threshold']['threshold']:.2f}"]
                   for p, judges in report.get('judge_metrics', {}).items()
                   for j, m in judges.items()])
    t = report['turns']
    out += ['## Turns', '']
    out += _table(['turns', 'cost', 'seconds p50', 'seconds p95',
                   'controller executed', 'tasks spawned'],
                  [[t['n'], _money(t['cost_usd']), _secs(t['seconds_p50']),
                    _secs(t['seconds_p95']), t['controller_executed'],
                    t['tasks_spawned']]] if t['n'] else [])
    return '\n'.join(out).rstrip() + '\n'
