"""Verify the treatment changes only the intended trainable parameter groups."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
import torch
from model import build_model
from week5_experiment import configure

class Week5Tests(unittest.TestCase):
    def test_gradient_boundary_and_shared_start(self):
        torch.manual_seed(42)
        model=build_model(9,pretrained=False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        frozen=configure(model,'frozen')
        self.assertEqual(set(frozen),{'fc.weight','fc.bias'})
        partial=configure(model,'layer4')
        self.assertTrue(any(n.startswith('layer4.') for n in partial))
        self.assertTrue(all(n.startswith(('layer4.','fc.')) for n in partial))
        self.assertEqual(partial,configure(model,'layer4_aug'))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        model.train()
        model(torch.randn(2,3,64,64)).sum().backward()
        for name,p in model.named_parameters():
            self.assertEqual(p.grad is not None,name in partial,name)

if __name__=='__main__':
    unittest.main()
