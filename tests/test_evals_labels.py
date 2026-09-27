"""Tests for the hand rubric grades (guru.evals.labels): the
``rubric-labels.toml`` format, lookup and agreement, and the shipped
file."""
from pathlib import Path

import pytest

from guru.evals import labels
from guru.evals.labels import HandLabel

VALID = '''
[[label]]
case = "a"
run = "*"
score = 2
note = "why   two"

[[label]]
case = "a"
run = "run1"
score = 1

[[label]]
case = "b"
score = 0
'''


class TestParseLabels:
    def test_parses_tables_with_defaults(self) -> None:
        got = labels.parse_labels(VALID)
        assert got == [HandLabel('a', '*', 2, 'why two'),
                       HandLabel('a', 'run1', 1, ''),
                       HandLabel('b', '*', 0, '')]

    def test_empty_text_is_no_labels(self) -> None:
        assert labels.parse_labels('') == []
        assert labels.parse_labels('# only a comment\n') == []

    @pytest.mark.parametrize('text, message', [
        ('[[label]]\ncase = "a"\nscore = 3\n', 'score must be one of 0, 1, 2'),
        ('[[label]]\ncase = "a"\nscore = true\n', 'score must be one of'),
        ('[[label]]\ncase = "a"\nscore = "2"\n', 'score must be one of'),
        ('[[label]]\ncase = "a"\n', 'score must be one of'),
        ('[[label]]\nscore = 2\n', 'case must be a non-empty string'),
        ('[[label]]\ncase = ""\nscore = 2\n', 'case must be a non-empty'),
        ('[[label]]\ncase = "a"\nscore = 2\nrun = ""\n', 'run must be a run'),
        ('[[label]]\ncase = "a"\nscore = 2\nnote = 1\n', 'note must be'),
        ('[[label]]\ncase = "a"\nscore = 2\nrubric = "x"\n',
         "unknown key 'rubric'"),
        ('[label]\ncase = "a"\nscore = 2\n', 'array of tables'),
        ('[[labels]]\ncase = "a"\nscore = 2\n', "unknown top-level key"),
        ('[[label]]\ncase = "a"\nscore = 2\n[[label]]\ncase = "a"\n'
         'score = 1\n', "duplicate grade for case 'a', run '*'"),
        ('[[label]\n', 'invalid TOML'),
    ])
    def test_rejects_naming_the_file_and_offender(self, text,
                                                  message) -> None:
        with pytest.raises(ValueError, match='lab.toml') as e:
            labels.parse_labels(text, 'lab.toml')
        assert message in str(e.value)


class TestLoadLabels:
    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert labels.load_labels(tmp_path / 'none.toml') == []

    def test_reads_and_validates(self, tmp_path: Path) -> None:
        p = tmp_path / 'lab.toml'
        p.write_text(VALID, encoding='utf-8')
        assert len(labels.load_labels(p)) == 3
        p.write_text('[[label]]\ncase = "a"\nscore = 9\n')
        with pytest.raises(ValueError, match=str(p)):
            labels.load_labels(p)

    def test_default_is_the_shipped_file(self) -> None:
        assert labels.DEFAULT_LABELS_FILE.name == 'rubric-labels.toml'
        assert labels.DEFAULT_LABELS_FILE.parent.name == 'evals'


class TestLookup:
    LABELS = labels.parse_labels(VALID)

    def test_specific_run_wins_over_any(self) -> None:
        assert labels.hand_label(self.LABELS, 'a', 'run1') == \
            HandLabel('a', 'run1', 1, '')
        assert labels.hand_label(self.LABELS, 'a', 'run2') == \
            HandLabel('a', '*', 2, 'why two')
        assert labels.hand_label(self.LABELS, 'b', 'run1') == \
            HandLabel('b', '*', 0, '')
        assert labels.hand_label(self.LABELS, 'c', 'run1') is None

    def test_agreement_skips_pairs_missing_a_side(self) -> None:
        assert labels.agreement([(2, 2), (1, 2), (None, 2), (0, None),
                                 (0, 0)]) == (2, 3)
        assert labels.agreement([]) == (0, 0)


class TestShippedLabels:
    def test_three_hand_grades_of_2026_09_24(self) -> None:
        got = labels.load_labels()
        by_case = {lab.case: lab for lab in got}
        assert set(by_case) >= {'guru-add-version-flag',
                                'guru-explain-gpu-fit',
                                'guru-review-adapters'}
        for name in ('guru-add-version-flag', 'guru-explain-gpu-fit',
                     'guru-review-adapters'):
            lab = by_case[name]
            assert (lab.run, lab.score) == ('*', 2), name
            assert '2026-09-24' in lab.note, name
