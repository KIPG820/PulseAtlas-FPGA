"""Identity-matched preset-point containment; visual overlap only."""
import math
LEAD_TARGET={'R':'RA','L':'LA','N':'RL','F':'LL','C1':'V1','C2':'V2','C3':'V3','C4':'V4','C5':'V5','C6':'V6'}
MAX_RESULT_AGE_SECONDS=1.0
STABLE_FRAMES=3
CONTRACT={'rule':'identity_then_preset_point_inside_original_pixel_bbox_inclusive',
          'lead_target_mapping':LEAD_TARGET,'point_margin_pixels':0,'max_result_age_seconds':MAX_RESULT_AGE_SECONDS,
          'stable_success_frames':STABLE_FRAMES,'ambiguity_iou':.5,'missing_detection_status':'unknown',
          'coordinate_space':'original_unmirrored_camera_pixels','meaning':'visual_overlap_with_current_preset_only',
          'contact_or_connection_verified':False,'anatomical_placement_verified':False}
REASONS={'no_pose':'人体点无效','frame_mismatch':'不同帧','stale_result':'结果过期','missing_point':'缺少预设点',
         'point_outside_frame':'预设点出画','invalid_point':'预设点无效','missing_detection':'未检出',
         'duplicate_identity':'同类多框','invalid_detection':'检测框无效','ambiguous_overlap':'重叠框身份不明确',
         'inside':'预设点在对应框内','outside':'预设点在框外','pending_stability':'等待连续稳定'}

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union>0 else 0.0

def summarize(result):
    counts={s:sum(item['status']==s for item in result['per_lead'].values()) for s in ['correct','adjust','unknown']}
    result.update(success_count=counts['correct'],adjust_count=counts['adjust'],unknown_count=counts['unknown'],
                  all_success=counts['correct']==len(LEAD_TARGET))
    return result

def match_frame(predictions,pose_info,frame_id,width,height,age_seconds=0.0):
    result={'frame_id':frame_id,'age_seconds':age_seconds,'per_lead':{},'meaning':CONTRACT['meaning'],
            'anatomical_placement_verified':False,'contact_or_connection_verified':False}
    common_reason=None
    if not math.isfinite(age_seconds) or age_seconds<0 or age_seconds>MAX_RESULT_AGE_SECONDS:common_reason='stale_result'
    elif not pose_info or pose_info.get('status')!='valid_torso':common_reason='no_pose'
    elif pose_info.get('frame_id')!=frame_id:common_reason='frame_mismatch'
    targets={} if not pose_info else {**pose_info.get('torso_reference_points',{}),**pose_info.get('chest_reference_points',{})}
    indexed={lead:[(i,p) for i,p in enumerate(predictions) if p.get('lead_id')==lead] for lead in LEAD_TARGET}
    valid_boxes={}
    for index,p in enumerate(predictions):
        b=p.get('bbox_xyxy',[]); confidence=p.get('confidence',0)
        try:
            if len(b)==4 and all(math.isfinite(float(v)) for v in b) and math.isfinite(float(confidence)) and confidence>=.15:
                if 0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height:valid_boxes[index]=b
        except (TypeError,ValueError):pass
    ambiguous=set()
    for i,a in valid_boxes.items():
        for j,b in valid_boxes.items():
            if j>i and predictions[i].get('lead_id')!=predictions[j].get('lead_id') and iou(a,b)>.5:
                ambiguous.update([i,j])
    for lead,label in LEAD_TARGET.items():
        item={'lead_id':lead,'target_label':label,'status':'unknown','reason':common_reason,
              'target_xy':None,'prediction_index':None,'bbox_xyxy':None,'confidence':None}
        result['per_lead'][lead]=item
        if common_reason:continue
        point=targets.get(label)
        if not point:item['reason']='missing_point';continue
        try:
            x,y=float(point['x_px']),float(point['y_px'])
            if not all(math.isfinite(v) for v in [x,y]):raise ValueError()
        except (ValueError,TypeError,KeyError):item['reason']='invalid_point';continue
        item['target_xy']=[x,y]
        if not (0<=x<width and 0<=y<height):item['reason']='point_outside_frame';continue
        found=indexed[lead]
        if not found:item['reason']='missing_detection';continue
        if len(found)!=1:item['reason']='duplicate_identity';continue
        index,p=found[0]; item.update(prediction_index=index,bbox_xyxy=p.get('bbox_xyxy'),confidence=p.get('confidence'))
        if index not in valid_boxes:item['reason']='invalid_detection';continue
        if index in ambiguous:item['reason']='ambiguous_overlap';continue
        x1,y1,x2,y2=valid_boxes[index]; inside=x1<=x<=x2 and y1<=y<=y2
        dx=x-min(x2,max(x1,x)); dy=y-min(y2,max(y1,y))
        item.update(status='correct' if inside else 'adjust',reason='inside' if inside else 'outside',
                    move_box_dx_px=dx,move_box_dy_px=dy,distance_to_bbox_px=math.hypot(dx,dy))
    return summarize(result)

class StableOverlapMatcher:
    def __init__(self):self.clear()
    def clear(self):self.streaks={lead:0 for lead in LEAD_TARGET};self.last_frame_id=None;self.last_capture_monotonic=None
    def update(self,predictions,pose_info,frame_id,width,height,age_seconds=0.0,captured_monotonic=None):
        if self.last_frame_id is not None and frame_id!=self.last_frame_id+1:self.clear()
        if captured_monotonic is not None and self.last_capture_monotonic is not None:
            gap=captured_monotonic-self.last_capture_monotonic
            if gap<0 or gap>MAX_RESULT_AGE_SECONDS:self.clear()
        self.last_capture_monotonic=captured_monotonic
        self.last_frame_id=frame_id
        result=match_frame(predictions,pose_info,frame_id,width,height,age_seconds)
        result['stable_success_frames']=STABLE_FRAMES
        for lead,item in result['per_lead'].items():
            item['raw_status']=item['status'];item['raw_reason']=item['reason']
            self.streaks[lead]=self.streaks[lead]+1 if item['status']=='correct' else 0
            item['success_streak']=self.streaks[lead]
            if item['raw_status']=='correct' and self.streaks[lead]<STABLE_FRAMES:
                item.update(status='unknown',reason='pending_stability')
        return summarize(result)

def draw_matching(image,result,show_points=True):
    import cv2
    im=image.copy()
    for item in result['per_lead'].values():
        status=item['status']; color={'correct':(60,220,60),'adjust':(70,70,240),'unknown':(150,150,150)}[status]
        if item['reason']=='pending_stability':color=(30,200,240)
        point=item['target_xy'];box=item['bbox_xyxy']
        if show_points and point and 0<=point[0]<im.shape[1] and 0<=point[1]<im.shape[0]:
            center=tuple(round(v) for v in point);radius=max(4,min(im.shape[:2])//80)+2
            cv2.circle(im,center,radius,color,-1,cv2.LINE_AA)
            cv2.circle(im,center,radius+2,(245,245,245),1,cv2.LINE_AA)
        if box and item['prediction_index'] is not None:
            a=tuple(round(v) for v in box[:2]);b=tuple(round(v) for v in box[2:])
            cv2.rectangle(im,a,b,color,2)
            label={'correct':'OK','adjust':'MOVE','unknown':'?'}[status]
            cv2.putText(im,f"{item['lead_id']}/{item['target_label']} {label}",(a[0],min(im.shape[0]-8,max(18,a[1]+18))),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            if status=='adjust' and point:
                center=(round((box[0]+box[2])/2),round((box[1]+box[3])/2))
                cv2.arrowedLine(im,center,tuple(round(v) for v in point),color,2,tipLength=.2)
    return im
