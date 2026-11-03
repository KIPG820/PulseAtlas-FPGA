"""Optional ROI identity/background guard. Main detector and seven unrelated identities stay untouched."""
import copy
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn

GROUP=(2,7,8)
LABELS=('RL','V4','V5','other_or_background')


def encoder_from_detector(model):
    layers=list(model.model.model.children())[:11]
    if len(layers)!=11 or any(layer.f!=-1 for layer in layers):raise RuntimeError('Unexpected YOLO11 backbone graph')
    encoder=nn.Sequential(*copy.deepcopy(layers),nn.AdaptiveAvgPool2d(1),nn.Flatten()).eval()
    for parameter in encoder.parameters():parameter.requires_grad=False
    return encoder


def signature(encoder):
    digest=hashlib.sha256()
    for name,tensor in encoder.state_dict().items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def head_for(channels):return nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,4))


def crop_roi(frame,box,expand=1.18):
    height,width=frame.shape[:2];x1,y1,x2,y2=box[:4]
    cx,cy=(x1+x2)/2,(y1+y2)/2;w,h=(x2-x1)*expand,(y2-y1)*expand
    left,top=max(0,int(cx-w/2)),max(0,int(cy-h/2));right,bottom=min(width,int(np.ceil(cx+w/2))),min(height,int(np.ceil(cy+h/2)))
    if right<=left or bottom<=top:raise ValueError('Empty ROI')
    return frame[top:bottom,left:right]


def blob(patch,size=224):
    h,w=patch.shape[:2];scale=min(size/w,size/h);nw,nh=round(w*scale),round(h*scale)
    image=np.full((size,size,3),114,np.uint8);x,y=(size-nw)//2,(size-nh)//2
    image[y:y+nh,x:x+nw]=cv2.resize(patch,(nw,nh))
    return np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255.


def apply_probabilities(boxes,probabilities,change=.9,reject=.98,margin=.25,source_confidence_limit=.65):
    """Probabilities only correspond to GROUP proposals; retain detector scores and original coordinates."""
    output=[];decisions=[];cursor=0;changed=False
    for index,source in enumerate(boxes):
        box=list(source);cls=int(box[5])
        if cls not in GROUP:output.append(box);continue
        probs=np.asarray(probabilities[cursor]);cursor+=1;order=np.argsort(probs)[::-1]
        winner=int(order[0]);gap=float(probs[winner]-probs[order[1]])
        action='keep'
        if winner==3 and cls==7 and probs[3]>=reject and gap>=margin and box[4]<.8:
            action='reject_other_or_background'
        elif winner==1 and cls in (2,8) and probs[winner]>=change and gap>=margin and box[4]<source_confidence_limit:
            box[5]=GROUP[winner];action='relabel';changed=True
        if action!='reject_other_or_background':output.append(box)
        decisions.append(dict(proposal=index,original_class_id=cls,result_class_id=None if action.startswith('reject') else int(box[5]),
            action=action,probabilities=probs.tolist(),score_margin=gap,detector_confidence=source[4]))
    if cursor!=len(probabilities):raise RuntimeError('ROI/probability alignment mismatch')
    # Only group classes need deduplication after relabeling; other seven classes remain verbatim.
    if changed:
        retained=[b for b in output if int(b[5]) not in GROUP]
        for cls in GROUP:
            subset=[b for b in output if int(b[5])==cls]
            if not subset:continue
            rects=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in subset]
            keep=cv2.dnn.NMSBoxes(rects,[b[4] for b in subset],0.,.5)
            retained.extend(subset[int(i)] for i in np.asarray(keep).reshape(-1))
        output=retained
    return sorted(output,key=lambda b:-b[4]),decisions


class ROIIdentityGuard:
    def __init__(self,detector,checkpoint,device='0'):
        self.device=torch.device('cpu' if str(device)=='cpu' else 'cuda:'+str(device))
        checkpoint=Path(checkpoint);self.checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.encoder=encoder_from_detector(detector)
        if signature(self.encoder)!=self.checkpoint['backbone_signature']:
            raise RuntimeError('Guard must initialize before main detector fusion and use the original baseline backbone')
        self.encoder.to(self.device).eval();self.head=head_for(self.checkpoint['channels']).to(self.device)
        self.head.load_state_dict(self.checkpoint['head_state']);self.head.eval()
        self.policy=self.checkpoint['policy']

    def refine(self,frame,boxes):
        selected=[b for b in boxes if int(b[5]) in GROUP]
        if not selected:return boxes,[]
        images=np.stack([blob(crop_roi(frame,b,self.policy['expand'])) for b in selected])
        with torch.inference_mode():
            features=self.encoder(torch.from_numpy(images).to(self.device))
            probabilities=self.head(features).softmax(-1).cpu().numpy()
        return apply_probabilities(boxes,probabilities,change=self.policy['change'],reject=self.policy['reject'],margin=self.policy['margin'],source_confidence_limit=self.policy.get('source_confidence_limit',.65))
