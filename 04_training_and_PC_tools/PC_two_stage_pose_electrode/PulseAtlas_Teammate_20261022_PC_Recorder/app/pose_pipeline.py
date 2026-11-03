"""Original R12 pose adapter and calibration, reused on the same camera frame."""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent/'vendor_pose'
sys.path.insert(0,str(ROOT))
from ps_onnx_pose import OpenCVDNNPersonPoseEstimator, draw_pose_overlay
from electrode_placement import TorsoLeadCalibration, ChestLeadCalibration, PersonPose

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class PosePipeline:
    def __init__(self):
        manifest=json.loads((ROOT/'vendor_manifest.json').read_text(encoding='utf-8-sig'))
        for item in manifest:
            assert sha(ROOT/item['relative_path'])==item['sha256'],f"Copied pose asset changed: {item['relative_path']}"
        model_manifest=json.loads((ROOT/'models/model_manifest.json').read_text(encoding='utf-8'))
        weights=ROOT/'models/yolo11n-pose-cv454.onnx'
        # The original model_manifest has a 63-character typo. The package's
        # SHA256SUMS and the untouched source ONNX agree on this full digest.
        actual_sha=sha(weights)
        assert actual_sha=='78543e228d6eeeb3788bc8fd7246c32ccee21ad5e83c0a12e5f1dba17e5ec8db'
        self.estimator=OpenCVDNNPersonPoseEstimator(weights, net=cv2.dnn.readNetFromONNX(np.frombuffer(weights.read_bytes(),dtype=np.uint8)))
        self.torso=TorsoLeadCalibration.load(ROOT/'calibration/torso_lead_calibration.json')
        self.chest=ChestLeadCalibration.load(ROOT/'calibration/chest_lead_calibration.json')
        self.contract={'model':'R12 YOLO11n-pose OpenCV raw-head ONNX','checkpoint':str(weights),
                       'sha256':actual_sha,'input_shape':[1,3,192,256],'input_color':'RGB',
                       'original_model_manifest_declared_sha256':model_manifest['sha256'],
                       'hash_verification_basis':'original handoff SHA256SUMS + copied asset hashes; model_manifest has a 63-character typo',
                       'confidence':self.estimator.confidence,'keypoint_confidence':self.estimator.keypoint_confidence,
                       'nms_iou':self.estimator.nms_iou,'opencv_threads':1,
                       'coordinate_space':'original_unmirrored_camera_pixels',
                       'calibration_origin':'copied original handoff v2; not newly verified for this camera/person',
                       'torso_calibration_sha256':sha(ROOT/'calibration/torso_lead_calibration.json'),
                       'chest_calibration_sha256':sha(ROOT/'calibration/chest_lead_calibration.json'),
                       'torso_marker_note':'RA/LA/RL/LL are old torso references, not standard wrist/ankle targets',
                       'placement_verdict_enabled':True,'placement_rule':'corresponding_preset_point_inside_detection_bbox',
                       'anatomical_placement_verified':False,'contact_or_connection_verified':False}

    def process(self,raw,show_reference_points=True):
        started=time.perf_counter(); pose=self.estimator.detect(raw)
        return self.render(raw,pose,show_reference_points,(time.perf_counter()-started)*1000)

    def render(self,raw,pose,show_reference_points=True,inference_ms=0.0):
        torso=self.torso.positions(pose); chest=self.chest.positions(pose)
        # No previous-frame pose/targets are reused when current torso is invalid.
        info={'status':'valid_torso' if pose is not None else 'no_valid_torso','pose':None,
              'torso_reference_points':{label:p.to_dict() for label,p in torso.items()},
              'chest_reference_points':{label:p.to_dict() for label,p in chest.items()},
              'reference_points_displayed':bool(show_reference_points),'reference_points_verified_for_current_use':False,
              'placement_verdict':'pending_overlap_matching','inference_ms':round(inference_ms,2),'model_sha256':self.contract['sha256'],
              'coordinate_space':'original_unmirrored_camera_pixels'}
        if pose is not None:
            info['pose']={'bbox_xyxy':list(pose.bbox),'confidence':float(pose.confidence),
                          'keypoints_xy':np.asarray(pose.keypoints_xy).tolist(),'keypoint_scores':np.asarray(pose.keypoint_scores).tolist()}
        drawn=draw_pose_overlay(raw,pose,torso if show_reference_points else {},chest if show_reference_points else {},mirror=False)
        return drawn,info,inference_ms
