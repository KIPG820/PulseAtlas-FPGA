"""Same-object temporal identity, no anatomy rules and no predicted boxes without a current proposal."""
import numpy as np
from scipy.optimize import linear_sum_assignment
GROUP=(2,7,8)
def iou(a,b):
    wh=np.maximum(0,np.minimum(a[2:4],b[2:4])-np.maximum(a[:2],b[:2]));inter=float(wh.prod());return inter/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
class TemporalIdentity:
    def __init__(self,confirm_frames=3,min_confidence=.55,max_gap_seconds=.5):
        self.confirm_frames=confirm_frames;self.min_confidence=min_confidence;self.max_gap=max_gap_seconds;self.tracks=[];self.next_id=0
    def clear(self):self.tracks=[]
    def update(self,boxes,timestamp):
        current=[list(p) for p in boxes if int(p[5]) in GROUP];unrelated=[list(p) for p in boxes if int(p[5]) not in GROUP]
        # Resolve spatially duplicate class hypotheses before tracking. Keep their raw votes in the report.
        clusters=[]
        for p in sorted(current,key=lambda p:-p[4]):
            options=[(iou(np.array(p),np.array(c[0])),j) for j,c in enumerate(clusters)]
            overlap,j=max(options,default=(0,None))
            if overlap>=.5:clusters[j].append(p)
            else:clusters.append([p])
        self.tracks=[t for t in self.tracks if 0<=timestamp-t['time']<=self.max_gap]
        costs=np.full((len(self.tracks),len(clusters)),100.,float);ambiguity=set()
        for i,track in enumerate(self.tracks):
            dt=timestamp-track['time'];pred=np.array(track['box'][:4])+np.tile(track['velocity']*dt,2)
            for j,c in enumerate(clusters):
                b=np.array(c[0][:4]);ov=iou(pred,b);center=(b[:2]+b[2:])/2;tc=(pred[:2]+pred[2:])/2;scale=max(1,np.linalg.norm(pred[2:]-pred[:2]));distance=np.linalg.norm(center-tc)/scale
                if ov>=.08 and distance<.8:costs[i,j]=1-ov+distance*.15
        matches={}
        if costs.size:
            a,b=linear_sum_assignment(costs)
            for i,j in zip(a,b):
                if costs[i,j]<1.05:
                    # Ambiguous assignment must not carry a previously confirmed identity across a crossing.
                    alternatives=sorted(costs[:,j]);near=sorted(costs[i,:])
                    if (len(alternatives)>1 and alternatives[1]-alternatives[0]<.12) or (len(near)>1 and near[1]-near[0]<.12):ambiguity.add(j)
                    matches[j]=int(i)
        output=unrelated;pending=[];decisions=[]
        for j,candidates in enumerate(clusters):
            box=candidates[0].copy();votes={int(p[5]):float(p[4]) for p in candidates};order=sorted(votes,key=lambda k:-votes[k]);raw=int(box[5]);strong=box[4]>=self.min_confidence and (len(order)==1 or votes[order[0]]-votes[order[1]]>=.15)
            if j in matches:
                track=self.tracks[matches[j]];dt=max(.001,timestamp-track['time']);old=np.array(track['box']);now=np.array(box[:4]);velocity=((now[:2]+now[2:])-(old[:2]+old[2:]))/(2*dt);track['velocity']=.5*track['velocity']+.5*velocity
            else:
                self.next_id+=1;track=dict(id=self.next_id,identity=None,candidate=None,streak=0,velocity=np.zeros(2));self.tracks.append(track)
            if j in ambiguity:
                track['identity']=None;track['streak']=0;strong=False
            if track['identity'] is None:
                if strong:
                    track['streak']=track['streak']+1 if track['candidate']==raw else 1;track['candidate']=raw
                    if track['streak']>=self.confirm_frames:track['identity']=raw
                else:track['streak']=0
            track.update(box=box[:4],time=timestamp)
            if track['identity'] is None:
                pending.append(box);action='pending'
            else:
                box[5]=track['identity'];output.append(box);action='stabilized' if raw!=track['identity'] or len(candidates)>1 else 'confirmed'
            decisions.append(dict(track_id=track['id'],action=action,raw_class=raw,result_class=track['identity'],raw_hypotheses=candidates,ambiguity=j in ambiguity))
        return sorted(output,key=lambda p:-p[4]),dict(decisions=decisions,pending=pending,positions_from_current_frame_only=True)
