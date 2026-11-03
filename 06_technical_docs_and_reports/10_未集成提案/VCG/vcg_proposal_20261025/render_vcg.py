"""Review-only derived VCG UI. All ECG data generated here are synthetic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecg_display_proposal_20261025"))
from render_proposal import (BG, WHITE, INK, BLUE, MUTED, RED, GREEN, BORDER,
                             rounded, header, logo, txt)

# PhysioNet GEH Kors regression matrix. Input: I, II, III, aVR, aVL, aVF, V1..V6.
# The redundant frontal leads have zero coefficients. Output: X, Y, Z.
KORS = np.array([
    [.38,-.07,0,0,0,0,-.13,.05,-.01,.14,.06,.54],
    [-.07,.93,0,0,0,0,.06,-.02,-.05,.06,-.17,.13],
    [.11,-.23,0,0,0,0,-.43,-.06,-.14,-.20,-.11,.31],
], dtype=np.float64)
LEADS = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")


def synthetic_derived_beat():
    time = np.arange(0, 1, .001)
    # Build a synthetic spatial beat, then solve for a consistent 12-lead input.
    # This gives a clear review diagram without pretending it was measured.
    target = np.zeros((len(time), 3), dtype=np.float64)
    for start, stop, radii, phase in [
        (.25,.41,(1.0,.84,.62),-.55),
        (.45,.70,(.30,.19,.15),-.30),
    ]:
        inside = (time >= start) & (time <= stop)
        u = (time[inside]-start)/(stop-start)
        env = np.sin(np.pi*u)**1.2
        theta = 2*np.pi*u+phase
        target[inside,0] += radii[0]*env*np.cos(theta)
        target[inside,1] += radii[1]*env*np.sin(theta)
        target[inside,2] += radii[2]*env*np.sin(theta+.75)
    independent = np.array([0,1,6,7,8,9,10,11])
    ecg = np.zeros((len(time),12),dtype=np.float64)
    ecg[:,independent] = target @ np.linalg.pinv(KORS[:,independent].T)
    ecg[:,2] = ecg[:,1] - ecg[:,0]
    ecg[:,3] = -(ecg[:,0]+ecg[:,1])/2
    ecg[:,4] = ecg[:,0]-ecg[:,1]/2
    ecg[:,5] = ecg[:,1]-ecg[:,0]/2
    xyz = ecg @ KORS.T
    return time, ecg, xyz


def project(draw, box, xyz, dims, names, *, compact=False):
    x0,y0,x1,y1=box
    draw.rounded_rectangle(box,radius=8 if compact else 12,fill=WHITE,outline=BORDER)
    padx=17 if compact else 28
    pady=17 if compact else 34
    cx=(x0+x1)/2;cy=(y0+y1)/2
    w=(x1-x0)/2-padx;h=(y1-y0)/2-pady
    points=xyz[:,[dims[0],dims[1]]]
    peak=max(.2,float(np.max(np.abs(points)))*1.18)
    def xy(p):
        return (cx+p[0]/peak*w,cy-p[1]/peak*h)
    for u in (-.5,0,.5):
        xx=cx+u*w;yy=cy+u*h
        draw.line((xx,y0+10,xx,y1-10),fill="#EDF2F7",width=1)
        draw.line((x0+10,yy,x1-10,yy),fill="#EDF2F7",width=1)
    draw.line((cx,y0+11,cx,y1-11),fill="#BDCBD9",width=1)
    draw.line((x0+11,cy,x1-11,cy),fill="#BDCBD9",width=1)
    qrs=points[252:407]
    for a,b in zip(qrs[:-1],qrs[1:]):
        draw.line((*xy(a),*xy(b)),fill=BLUE,width=2 if compact else 3)
    tloop=points[445:705]
    for a,b in zip(tloop[:-1],tloop[1:]):
        draw.line((*xy(a),*xy(b)),fill="#D68B4B",width=2 if compact else 3)
    r=3 if compact else 5
    px,py=xy(points[252])
    draw.ellipse((px-r,py-r,px+r,py+r),fill=GREEN)
    txt(draw,(x0+12,y0+7),f"{names[0]} – {names[1]}",
        11 if compact else 17,INK,True)
    txt(draw,(x1-18,cy+4),names[0],11 if compact else 14,MUTED)
    txt(draw,(cx+5,y0+20),names[1],11 if compact else 14,MUTED)


def render_3d(image,box,xyz,compact=False):
    """Perspective, shaded planes and depth-sorted trails, rendered at 2x."""
    x0,y0,x1,y1=map(int,box)
    scale=2
    w,h=(x1-x0)*scale,(y1-y0)*scale
    yy=np.linspace(0,1,h)[:,None,None]
    top=np.array([18,39,63]);bottom=np.array([10,25,47])
    bg=np.broadcast_to((top*(1-yy)+bottom*yy).astype('uint8'),(h,w,3)).copy()
    tile=Image.fromarray(bg)
    d=ImageDraw.Draw(tile,"RGBA")
    camera=np.array([1.4,1.6,4.5],dtype=float)
    camera/=np.linalg.norm(camera)
    right=np.cross([0,1,0],camera);right/=np.linalg.norm(right)
    up=np.cross(camera,right);up/=np.linalg.norm(up)
    cx=w*.51;cy=h*.53
    factor=min(w*.65,h*.76)
    def screen(p):
        p=np.asarray(p,dtype=float)
        depth=float(np.dot(p,camera))
        magnification=3.8/(3.8-depth)
        return (cx+np.dot(p,right)*factor*.48*magnification,
                cy-np.dot(p,up)*factor*.48*magnification,depth)
    # Three reference sheets make the XYZ orientation visible without implying
    # a diagnostic anatomical surface.
    ext=1.12
    planes=[
        ([(a,b,-.86) for a,b in [(-ext,-ext),(ext,-ext),(ext,ext),(-ext,ext)]],(47,117,164,28)),
        ([(a,-ext,b) for a,b in [(-ext,-ext),(ext,-ext),(ext,ext),(-ext,ext)]],(39,144,151,20)),
    ]
    for corners,fill in planes:
        d.polygon([(screen(p)[0],screen(p)[1]) for p in corners],fill=fill)
        d.line([(screen(p)[0],screen(p)[1]) for p in corners+[corners[0]]],
               fill=(94,156,182,67),width=2)
    for fixed_axis,fixed_value in [(2,-.86),(1,-ext)]:
        free=[ax for ax in range(3) if ax!=fixed_axis]
        for axis in free:
            other=free[1] if axis==free[0] else free[0]
            for v in np.arange(-1,1.01,.5):
                a=np.zeros(3);b=np.zeros(3)
                a[fixed_axis]=fixed_value;b[fixed_axis]=fixed_value
                a[other]=v;b[other]=v
                a[axis]=-ext;b[axis]=ext
                pa,pb=screen(a),screen(b)
                d.line((pa[0],pa[1],pb[0],pb[1]),fill=(115,173,195,39),width=2)
    origin=screen((0,0,0))
    for axis,label,col in [(0,"X",(94,198,241,240)),
                           (1,"Y",(126,220,187,240)),
                           (2,"Z",(212,175,246,240))]:
        endpoint=np.zeros(3);endpoint[axis]=1.35
        px,py,_=screen(endpoint)
        d.line((origin[0],origin[1],px,py),fill=col,width=3)
        d.ellipse((px-4,py-4,px+4,py+4),fill=col)
        d.text((px+7,py-12),label,fill=col,font=ImageFont.truetype(
            r"C:\Windows\Fonts\arialbd.ttf",14*scale if not compact else 12*scale))
    # Paint farther portions first. Near portions are brighter and wider.
    for begin,end,base in [(252,407,(62,181,240)),(445,705,(242,163,87))]:
        samples=xyz[begin:end]
        spans=[]
        for a,b in zip(samples[:-1],samples[1:]):
            pa,pb=screen(a),screen(b)
            spans.append(((pa[2]+pb[2])/2,pa,pb))
        for depth,pa,pb in sorted(spans,key=lambda row:row[0]):
            v=np.clip((depth+1)/2,0,1)
            color=tuple(int(c*(.54+.46*v)) for c in base)
            width=int((2.3+2.2*v)*scale)
            d.line((pa[0],pa[1],pb[0],pb[1]),fill=(*color,245),width=width)
    start=screen(xyz[252]);d.ellipse((start[0]-5,start[1]-5,start[0]+5,start[1]+5),
                                   fill=(85,215,160,255))
    if not compact:
        d.text((18*scale,15*scale),"XYZ  空间轨迹",fill=(232,246,252,255),
               font=ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc",19*scale))
        d.text((18*scale,h-29*scale),"透视视角 · 左右滑动可旋转",fill=(151,187,212,255),
               font=ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc",13*scale))
    tile=tile.resize((x1-x0,y1-y0),Image.Resampling.LANCZOS)
    image.paste(tile,(x0,y0))
    ImageDraw.Draw(image).rounded_rectangle(box,radius=11,outline="#6D9BBC",width=1)


def hdmi():
    im=Image.new("RGB",(1280,720),BG);d=ImageDraw.Draw(im)
    header(d,"心电向量图")
    txt(d,(828,116),"12 导联推导 · Kors 变换",17,MUTED)
    rounded(d,(35,178,1245,642),16)
    txt(d,(57,196),"选中单搏 · QRS 与 T 空间环",23,INK,True)
    txt(d,(917,201),"信号质量：有效",17,GREEN,True)
    t,ecg,xyz=synthetic_derived_beat()
    render_3d(im,(57,239,718,557),xyz)
    d=ImageDraw.Draw(im)
    project(d,(741,239,1224,338),xyz,(0,1),("X","Y"),compact=True)
    project(d,(741,348,1224,447),xyz,(0,2),("X","Z"),compact=True)
    project(d,(741,458,1224,557),xyz,(1,2),("Y","Z"),compact=True)
    txt(d,(57,579),"数据源：同一心搏的 I、II、V1–V6 · 采样 1000 SPS",16,MUTED)
    d.ellipse((848,584,858,594),fill=BLUE);txt(d,(867,577),"QRS 环",15,MUTED)
    d.ellipse((953,584,963,594),fill="#D68B4B");txt(d,(972,577),"T 环",15,MUTED)
    d.ellipse((1040,584,1050,594),fill=GREEN);txt(d,(1059,577),"起点",15,MUTED)
    txt(d,(57,607),"由标准 12 导联推导；非 Frank 电极直接测量，不用于独立诊断",15,MUTED)
    txt(d,(35,673),"设计示意 · 合成波形；实际幅值与方向需完成导联极性、时间同步和增益校验",14,MUTED)
    p=HERE/"HDMI_12导联推导VCG_评审图.png";im.save(p);return p


def tft():
    im=Image.new("RGB",(320,240),BG);d=ImageDraw.Draw(im)
    rounded(d,(5,5,315,38),6)
    logo(d,11,10,.42)
    txt(d,(38,11),"PulseAtlas",15,INK,True)
    txt(d,(8,43),"←",18,BLUE,True)
    txt(d,(35,43),"心电向量图",20,INK,True)
    _,_,xyz=synthetic_derived_beat()
    render_3d(im,(8,74,312,200),xyz,compact=True)
    d=ImageDraw.Draw(im)
    txt(d,(16,211),"3D 轨迹",10,BLUE,True)
    txt(d,(113,211),"XY 正面",10,MUTED)
    txt(d,(215,211),"XZ / YZ",10,MUTED)
    p=HERE/"TFT_VCG_XY_320x240.png";im.save(p)
    im.resize((960,720),Image.Resampling.NEAREST).save(HERE/"TFT_VCG_XY_3倍预览.png")
    return p


if __name__=="__main__":
    HERE.mkdir(parents=True,exist_ok=True)
    for fn in (hdmi,tft):print(fn())
