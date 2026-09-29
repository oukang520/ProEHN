"""Dependency-free guards for retired entry points introduced by remote updates."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RetiredEntryPointTests(unittest.TestCase):
    def function(self, filename, name):
        tree = ast.parse((ROOT / filename).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(body=[node], type_ignores=[])
        namespace = {'parse_args': lambda: None}
        exec(compile(module, filename, 'exec'), namespace)
        return namespace[name]

    def test_surrogate_cannot_be_default_benchmark(self):
        with self.assertRaisesRegex(RuntimeError, 'surrogate is not ProEHN'):
            self.function('proehn/paca_proehn_metrics.py', 'run_paca_proehn_metrics')()

    def test_single_table_training_cannot_fit_before_split(self):
        with self.assertRaisesRegex(ValueError, 'split raw patients'):
            self.function('proehn/training.py', 'train_kinetic_from_csv')()

    def test_legacy_cli_fails_before_cohort_io(self):
        with self.assertRaisesRegex(RuntimeError, 'No cohort file was read'):
            self.function('scripts/run_paca_examples.py', 'main')()

    def test_all_python_sources_parse(self):
        for folder in ('proehn', 'scripts', 'tests'):
            for path in (ROOT / folder).rglob('*.py'):
                ast.parse(path.read_text(), filename=str(path))


if __name__ == '__main__':
    unittest.main()
