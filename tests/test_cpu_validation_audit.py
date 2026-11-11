import copy
import json
from pathlib import Path
import unittest

from src.experiments.cpu_validation_audit import validate


class CpuAuditScopeTests(unittest.TestCase):
    def spec(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/cpu_validation_audit_20261030.json').read_text())

    def test_frozen_60_pass_scope(self):
        validate(self.spec())
        for field,value in [('passes_per_checkpoint',3),('torch_threads',2),('batch_size',32)]:
            spec=self.spec();spec[field]=value
            with self.assertRaises(ValueError):validate(spec)

    def test_missing_or_duplicate_checkpoint_rejected(self):
        spec=self.spec();spec['rows'][-1]=copy.deepcopy(spec['rows'][0])
        with self.assertRaises(ValueError):validate(spec)


if __name__=='__main__':unittest.main()
