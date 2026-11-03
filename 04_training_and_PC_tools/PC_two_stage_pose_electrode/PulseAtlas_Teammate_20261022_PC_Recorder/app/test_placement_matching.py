"""Meaningful matching/stability failures; no camera, model or medical landmarks needed."""
import unittest,copy
from placement_matching import match_frame, StableOverlapMatcher, LEAD_TARGET

def pose(frame=1,points=None):
    points=points or {'V1':{'x_px':50,'y_px':50},'V2':{'x_px':150,'y_px':50}}
    return {'status':'valid_torso','frame_id':frame,'chest_reference_points':points,'torso_reference_points':{}}
def box(lead='C1',bbox=None,confidence=.9):
    return {'lead_id':lead,'bbox_xyxy':bbox or [40,40,60,60],'confidence':confidence}

class PlacementMatchingTests(unittest.TestCase):
    def test_identity_mapping_including_f_n(self):
        self.assertEqual({k:LEAD_TARGET[k] for k in ['R','L','N','F']},{'R':'RA','L':'LA','N':'RL','F':'LL'})
        self.assertEqual([LEAD_TARGET['C'+str(i)] for i in range(1,7)],['V'+str(i) for i in range(1,7)])
    def test_point_inside_and_inclusive_edge(self):
        for b in [[40,40,60,60],[50,50,60,60]]:
            r=match_frame([box(bbox=b)],pose(),1,200,200)
            self.assertEqual(r['per_lead']['C1']['status'],'correct')
            self.assertEqual(r['success_count'],1)
    def test_wrong_identity_does_not_match_nearby_point(self):
        r=match_frame([box('C2')],pose(),1,200,200)
        self.assertEqual(r['per_lead']['C1']['reason'],'missing_detection')
        self.assertEqual(r['per_lead']['C2']['status'],'adjust')
        self.assertEqual(r['success_count'],0)
    def test_outside_gives_adjustment_vector(self):
        r=match_frame([box(bbox=[20,20,40,40])],pose(),1,200,200)['per_lead']['C1']
        self.assertEqual(r['status'],'adjust');self.assertEqual((r['move_box_dx_px'],r['move_box_dy_px']),(10,10))
    def test_missing_detection_is_unknown(self):
        r=match_frame([],pose(),1,200,200)
        self.assertEqual(r['unknown_count'],10);self.assertEqual(r['adjust_count'],0)
    def test_duplicate_identity_is_not_picked_by_confidence(self):
        r=match_frame([box(confidence=.9),box(bbox=[140,40,160,60],confidence=.2)],pose(),1,200,200)
        self.assertEqual(r['per_lead']['C1']['reason'],'duplicate_identity')
    def test_conflicting_overlapping_classes_are_unknown(self):
        r=match_frame([box(),box('C2',[41,41,61,61])],pose(),1,200,200)
        self.assertEqual(r['per_lead']['C1']['reason'],'ambiguous_overlap')
        self.assertEqual(r['per_lead']['C2']['reason'],'ambiguous_overlap')
    def test_invalid_pose_frame_and_staleness(self):
        for p,f,age,reason in [(None,1,0,'no_pose'),(pose(2),1,0,'frame_mismatch'),(pose(),1,1.01,'stale_result')]:
            r=match_frame([box()],p,f,200,200,age)
            self.assertEqual(r['unknown_count'],10);self.assertEqual(r['per_lead']['C1']['reason'],reason)
    def test_target_outside_and_nan(self):
        for x,reason in [(200,'point_outside_frame'),(float('nan'),'invalid_point')]:
            r=match_frame([box()],pose(points={'V1':{'x_px':x,'y_px':50}}),1,200,200)
            self.assertEqual(r['per_lead']['C1']['reason'],reason)
    def test_invalid_bbox_and_low_confidence(self):
        for b,c in [([60,40,40,60],.9),([-1,0,100,100],.9),([40,40,60,60],.1)]:
            r=match_frame([box(bbox=b,confidence=c)],pose(),1,200,200)
            self.assertEqual(r['per_lead']['C1']['reason'],'invalid_detection')
    def test_one_rectangle_is_not_reused_for_two_points(self):
        r=match_frame([box(bbox=[40,40,160,60])],pose(),1,200,200)
        self.assertEqual(r['success_count'],1);self.assertEqual(r['per_lead']['C2']['status'],'unknown')
    def test_three_consecutive_frames_and_immediate_revocation(self):
        matcher=StableOverlapMatcher()
        for frame in [1,2,3]:
            r=matcher.update([box()],pose(frame),frame,200,200)
            self.assertEqual(r['per_lead']['C1']['raw_status'],'correct')
            self.assertEqual(r['per_lead']['C1']['status'],'correct' if frame==3 else 'unknown')
        r=matcher.update([box(bbox=[20,20,40,40])],pose(4),4,200,200)
        self.assertEqual(r['per_lead']['C1']['status'],'adjust')
        r=matcher.update([box()],None,5,200,200)
        self.assertEqual(r['success_count'],0)
    def test_same_or_skipped_frame_does_not_accumulate_stability(self):
        matcher=StableOverlapMatcher()
        for frame in [1,1,1,8]:
            r=matcher.update([box()],pose(frame),frame,200,200)
            self.assertEqual(r['success_count'],0)
    def test_capture_gap_resets_stability(self):
        matcher=StableOverlapMatcher()
        for frame,timestamp in [(1,1.0),(2,1.1),(3,1.2)]:r=matcher.update([box()],pose(frame),frame,200,200,0,timestamp)
        self.assertEqual(r['success_count'],1)
        r=matcher.update([box()],pose(4),4,200,200,0,3.0)
        self.assertEqual(r['success_count'],0)
    def test_complete_ten_leads_and_no_input_mutation(self):
        points={};pred=[]
        for i,(lead,label) in enumerate(LEAD_TARGET.items()):
            x,y=30+(i%5)*80,30+(i//5)*80
            points[label]={'x_px':x,'y_px':y};pred.append(box(lead,[x-10,y-10,x+10,y+10]))
        p=pose(points=points);before=copy.deepcopy((p,pred));matcher=StableOverlapMatcher()
        for frame in [1,2,3]:r=matcher.update(pred,dict(p,frame_id=frame),frame,500,300)
        self.assertEqual(r['success_count'],10);self.assertTrue(r['all_success']);self.assertEqual((p,pred),before)

if __name__=='__main__':unittest.main(verbosity=2)
