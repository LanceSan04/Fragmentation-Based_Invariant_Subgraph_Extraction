import unittest
import numpy as np
from src.experiments.reload_diagnostic import compare


class ReloadDiagnosticTests(unittest.TestCase):
    def trace(self):
        return dict(labels=np.array([0,1]),probabilities=np.array([.3,.7]),
            edge_row=np.array([0,0,1]),mask0=np.array([True,False,True]),
            scores0=np.array([.5,.5,.8]))

    def test_compares_selections_and_probabilities_independently(self):
        a=self.trace();b=self.trace();b['mask0'][:2]=[False,True]
        r=compare(a,b,1)
        self.assertEqual(r['changed_probabilities'],0)
        self.assertEqual(r['views'][0]['changed_selection_bits'],2)
        self.assertEqual(r['views'][0]['changed_molecule_rows'],[0])

    def test_membership_change_rejected(self):
        a=self.trace();b=self.trace();b['labels']=np.array([1,0])
        with self.assertRaises(ValueError):compare(a,b,1)


if __name__=='__main__':unittest.main()
