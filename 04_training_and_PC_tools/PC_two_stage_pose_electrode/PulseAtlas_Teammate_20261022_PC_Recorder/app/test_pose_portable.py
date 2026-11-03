"""Run original decoder/geometry regressions using the portable ONNX loader."""
from pathlib import Path
import sys, unittest
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor_pose'))
import test_ps_onnx_pose as original
from pose_pipeline import PosePipeline

class PortablePoseDecoderTests(original.PoseDecoderTests):
    def test_bundled_graph_loads_and_blank_frame_has_no_pose(self):
        pipeline=PosePipeline()
        blank=np.zeros((480,640,3),dtype=np.uint8)
        self.assertIsNone(pipeline.estimator.detect(blank))
        _,info,_=pipeline.process(blank)
        self.assertEqual(info['status'],'no_valid_torso')
        self.assertFalse(info['chest_reference_points'])
        self.assertFalse(info['torso_reference_points'])

if __name__=='__main__':unittest.main(verbosity=2)
