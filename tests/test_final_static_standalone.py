"""Standard-library source contracts; no environment installation or cohort IO."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scientific_config_test',ROOT/'proehn/scientific_config.py')
config=importlib.util.module_from_spec(spec);spec.loader.exec_module(config)


def numeric_section(text,name):
    lines=text.splitlines(); start=lines.index(name+':')+1; values={}
    for line in lines[start:]:
        if line and not line.startswith(' '): break
        if not line.strip(): continue
        key,value=line.strip().split(':',1); value=value.strip()
        values[key]=value if key=='calibration' else float(value)
    return values


class SourceContracts(unittest.TestCase):
    def test_explicit_frozen_numeric_configurations(self):
        for cohort in ('paca','mela','luad'):
            text=(ROOT/f'configs/cohorts/{cohort}_scientific.yaml').read_text()
            config.validate_model_parameters(numeric_section(text,'kinetic'),numeric_section(text,'topology'),'prespecified')
            self.assertIn('method: sequential_importance',text)
            self.assertIn('label_protocol:\n  frozen: false',text)
        with self.assertRaises(ValueError): config.validate_model_parameters({}, {}, 'prespecified')

    def test_fitting_has_no_exact_posterior_size_veto(self):
        tree=ast.parse((ROOT/'proehn/benchmark.py').read_text())
        trainer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FormalProEHNTrainer')
        fit=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='_fit_prespecified')
        names={n.id for n in ast.walk(fit) if isinstance(n,ast.Name)}
        self.assertNotIn('ExactInferenceLimitError',names)
        self.assertIn('fit_topology_model',names)

    def test_ci_has_only_source_and_synthetic_checks(self):
        text=(ROOT/'.github/workflows/scientific-tests.yml').read_text()
        self.assertIn('on: [push, pull_request]',text)
        self.assertIn('PROEHN_SYNTHETIC_ONLY',text)
        self.assertNotIn('test_model_adapter.py',text)
        self.assertNotIn('run_paca',text)
        self.assertNotIn('train_topology_from_csv',text)

    def test_formal_metric_layer_cannot_select_thresholds(self):
        tree=ast.parse((ROOT/'proehn/metrics.py').read_text())
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertNotIn('calibrate_threshold',names)
        self.assertNotIn('max_mcc_descriptive',names)
        self.assertNotIn('roc_curve',names)

    def test_changed_python_sources_parse(self):
        for root in ('proehn','tests','scripts','proehn_service'):
            for path in (ROOT/root).rglob('*.py'): ast.parse(path.read_text(),filename=str(path))


if __name__=='__main__': unittest.main()
