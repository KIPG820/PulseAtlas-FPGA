"""PIL renderer for the PulseAtlas R9 HDMI test application.

This intentionally has no PYNQ imports so page layout can be rendered and
reviewed on a desktop before the package is copied to the board.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ecg_display_metrics import (
    SPECTROGRAM_DB_MAX, SPECTROGRAM_DB_MIN, SPECTROGRAM_INVALID,
    TREND_INVALID, spectrogram_dbfs, trend_points,
)


BG = (244, 248, 253)
WHITE = (255, 255, 255)
NAVY = (28, 52, 77)
BLUE = (55, 108, 174)
BLUE_DARK = (43, 89, 151)
BLUE_LIGHT = (225, 236, 249)
MUTED = (104, 132, 159)
RED = (221, 60, 83)
RED_LIGHT = (250, 217, 224)
GREEN = (54, 143, 110)
GRID_MINOR = (250, 226, 231)
GRID_MAJOR = (239, 180, 192)
LINE = (216, 229, 243)

LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6")


class Fonts:
    def __init__(self):
        self._cache = {}
        self._cjk_path = self._find_cjk_font()
        self._latin_path = self._find_latin_font()

    @staticmethod
    def _find_cjk_font():
        candidates = [
            os.environ.get("PULSEATLAS_CJK_FONT", ""),
            str(Path(__file__).resolve().parent / "fonts" / "PulseAtlasCJK.ttf"),
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _find_latin_font():
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    def get(self, size, bold=False):
        key = (int(size), bool(bold))
        if key not in self._cache:
            path = self._cjk_path or self._latin_path
            if path:
                try:
                    font = ImageFont.truetype(path, key[0])
                    if self._cjk_path and hasattr(font, "set_variation_by_axes"):
                        try:
                            font.set_variation_by_axes([650 if key[1] else 450])
                        except (OSError, ValueError):
                            pass
                    self._cache[key] = font
                except OSError:
                    self._cache[key] = ImageFont.load_default()
            else:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]

    @property
    def has_cjk(self):
        return self._cjk_path is not None


FONTS = Fonts()


def _text(draw, xy, text, size=20, fill=NAVY, bold=False, anchor=None):
    draw.text(xy, str(text), font=FONTS.get(size, bold), fill=fill, anchor=anchor)


def _card(draw, box, radius=16, fill=WHITE, outline=LINE, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _base(width, height):
    return Image.new("RGB", (width, height), BG)


def _logo(draw, x=34, y=25, scale=1.0):
    r = int(27 * scale)
    cx, cy = x + r, y + r
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline=BLUE, width=max(2, int(2*scale)))
    draw.arc((cx-int(r*.55), cy-r+2, cx+int(r*.55), cy+r-2), 180, 360,
             fill=(193, 214, 235), width=max(1, int(scale)))
    draw.line((cx-r+4, cy, cx+r-4, cy), fill=(206, 221, 238), width=max(1, int(scale)))
    points = [(cx-r+5, cy+4), (cx-8, cy+4), (cx-3, cy-8),
              (cx+2, cy+13), (cx+8, cy-16), (cx+13, cy+4),
              (cx+20, cy+4), (cx+r-5, cy+4)]
    draw.line(points, fill=RED, width=max(2, int(3*scale)), joint="curve")
    draw.ellipse((cx+r-4, cy-r+2, cx+r+4, cy-r+10), fill=BLUE)
    return x + 2*r + int(12*scale)


def _header(image, state, page_title=None, back="首页"):
    draw = ImageDraw.Draw(image)
    _logo(draw, 34, 22, 0.95)
    _text(draw, (91, 28), "PulseAtlas", 30, NAVY, True)
    if page_title:
        _button(draw, (34, 82, 132, 126), "← "+back, size=15)
        _text(draw, (151, 83), page_title, 36, NAVY, True)

    _status_row(draw,state)
    return draw


def _status_row(draw,state,start_x=790,y=38):
    status=state.get("status",{})
    groups=[("ECG",bool(status.get("id_ok",False))),
            ("USB",bool(state.get("camera_connected",False))),
            ("Wi-Fi",bool(state.get("wifi_connected",False)))]
    xs=(start_x,start_x+152,start_x+303)
    for (label,ok),x in zip(groups,xs):
        color=GREEN if ok else MUTED
        if label=="ECG":
            draw.rounded_rectangle((x,y+3,x+26,y+22),radius=3,outline=BLUE,width=2)
            draw.line([(x+4,y+14),(x+9,y+14),(x+12,y+9),(x+15,y+18),
                       (x+18,y+11),(x+22,y+14)],fill=RED,width=2)
        elif label=="USB":
            draw.rounded_rectangle((x,y+7,x+26,y+23),radius=4,outline=BLUE,width=2)
            draw.rectangle((x+7,y+3,x+15,y+7),fill=BG,outline=BLUE,width=2)
            draw.ellipse((x+9,y+10,x+18,y+19),outline=BLUE,width=2)
        else:
            draw.arc((x+2,y+3,x+25,y+25),218,322,fill=BLUE,width=2)
            draw.arc((x+6,y+8,x+21,y+24),218,322,fill=BLUE,width=2)
            draw.ellipse((x+12,y+23,x+16,y+27),fill=BLUE)
        draw.ellipse((x+32,y+11,x+40,y+19),fill=color)
        _text(draw,(x+46,y+4),label,16,color,True)


def _button(draw, box, label, primary=False, size=18):
    fill, outline, color = (BLUE, BLUE, WHITE) if primary else (WHITE, BLUE, BLUE)
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=1)
    _text(draw, ((box[0]+box[2])//2, (box[1]+box[3])//2), label,
          size, color, True, anchor="mm")


def _draw_ecg_grid(draw, box, major=50):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=WHITE, outline=LINE, width=1)
    minor = max(8, major // 5)
    for x in range(x0+1, x1, minor):
        is_major = ((x-x0) % major) == 0
        draw.line((x, y0+1, x, y1-1), fill=GRID_MAJOR if is_major else GRID_MINOR,
                  width=1 if is_major else 1)
    for y in range(y0+1, y1, minor):
        is_major = ((y-y0) % major) == 0
        draw.line((x0+1, y, x1-1, y), fill=GRID_MAJOR if is_major else GRID_MINOR,
                  width=1 if is_major else 1)


def _wave_points(signal, box, gain=1.0, center=None):
    x0, y0, x1, y1 = box
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return []
    values = values[-max(2, x1-x0):]
    baseline = float(np.median(values)) if center is None else float(center)
    robust = float(np.percentile(np.abs(values-baseline), 98))
    scale = max(robust, 1.0) / max(gain, 0.05)
    mid = (y0+y1)/2.0
    amp = (y1-y0)*0.43
    xs = np.linspace(x0+1, x1-1, values.size)
    ys = mid - np.clip((values-baseline)/scale, -1.0, 1.0)*amp
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _plot_ecg(draw, signal, box, gain=1.0, grid=True, width=2):
    if grid:
        _draw_ecg_grid(draw, box)
    else:
        draw.rounded_rectangle(box, radius=8, fill=WHITE, outline=LINE, width=1)
    points = _wave_points(signal, box, gain)
    if len(points) >= 2:
        draw.line(points, fill=RED, width=width, joint="curve")


def _status_text(state):
    status = state.get("status", {})
    if state.get("monitor_running"):
        return "采集中", GREEN
    if status.get("id_ok"):
        return "设备已连接", GREEN
    return "等待连接", MUTED


def render_home(state, width, height):
    image = _base(width, height)
    draw = ImageDraw.Draw(image)
    brand_x=_logo(draw,34,16,1.65)
    _text(draw,(brand_x+8,35),"PulseAtlas",46,NAVY,True)
    _status_row(draw,state,start_x=790,y=40)
    cards = [
        ("实时监测", "连续心电 · 心率", RED, "ECG"),
        ("12 导联", "同步观察 · 12 路信号", BLUE, "12"),
        ("AI 分析", "代表性片段辅助判读", BLUE, "AI"),
        ("视觉贴片引导", "人体定位 · 贴片识别", BLUE, "VIS"),
        ("历史数据库", "测量记录 · 变化趋势", BLUE, "DB"),
        ("设置", "采集 · 显示 · 提示", BLUE, "CFG"),
    ]
    margin, gap_x, gap_y = 34, 18, 18
    top, bottom = 143, height-24
    card_w = (width-2*margin-2*gap_x)//3
    card_h = (bottom-top-gap_y)//2
    focused = int(state.get("home_focus", 0)) % len(cards)
    for i, (title, subtitle, accent, tag) in enumerate(cards):
        row, col = divmod(i, 3)
        x = margin + col*(card_w+gap_x)
        y = top + row*(card_h+gap_y)
        selected = i == focused
        _card(draw, (x, y, x+card_w, y+card_h), 18,
              fill=(249, 252, 255) if selected else WHITE,
              outline=BLUE if selected else LINE, width=3 if selected else 1)
        icon = (x+20, y+card_h//2-39, x+98, y+card_h//2+39)
        draw.rounded_rectangle(icon, radius=15, fill=(241, 246, 252))
        _draw_home_icon(draw,tag,icon)
        text_x = x+118
        _text(draw, (text_x, y+32), title, 26, NAVY, True)
        _text(draw, (text_x, y+70), subtitle, 17, MUTED)
        _button(draw, (x+card_w-116, y+card_h-53, x+card_w-18, y+card_h-15),
                "进入  →", primary=selected, size=16)
    return image


def _draw_home_icon(draw,tag,box):
    x0,y0,x1,y1=box
    blue=(55,108,174)
    red=(221,60,83)
    cx=(x0+x1)//2
    if tag=="ECG":
        draw.rounded_rectangle((x0+15,y0+16,x1-15,y1-18),7,outline=blue,width=3)
        draw.line([(x0+22,y0+42),(x0+34,y0+42),(x0+41,y0+30),
                   (x0+49,y0+53),(x0+57,y0+25),(x0+65,y0+42),(x1-22,y0+42)],fill=red,width=3)
        draw.line((cx,y1-18,cx,y1-8),fill=blue,width=3)
        draw.line((cx-14,y1-8,cx+14,y1-8),fill=blue,width=3)
    elif tag=="12":
        pts=[(cx-17,y0+14),(cx-25,y0+29),(cx-20,y0+52),(cx-18,y1-10),
             (cx+18,y1-10),(cx+20,y0+52),(cx+25,y0+29),(cx+17,y0+14)]
        draw.line(pts,fill=blue,width=3,joint="curve")
        for xx,yy,c in [(cx-9,y0+32,red),(cx+9,y0+32,blue),(cx-9,y0+48,blue),(cx+9,y0+48,red),(cx-9,y0+63,red),(cx+9,y0+63,blue)]:
            draw.ellipse((xx-3,yy-3,xx+3,yy+3),fill=c)
    elif tag=="AI":
        left=[(x0+20,y0+20),(x0+20,y0+39),(x0+20,y0+58)]
        mid=[(cx,y0+29),(cx,y0+49)]
        right=[(x1-20,y0+20),(x1-20,y0+39),(x1-20,y0+58)]
        for a in left:
            for b in mid: draw.line((a,b),fill=blue,width=1)
        for a in mid:
            for b in right: draw.line((a,b),fill=blue,width=1)
        for xx,yy in left+right: draw.ellipse((xx-4,yy-4,xx+4,yy+4),fill=blue)
        for xx,yy in mid: draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=red)
    elif tag=="VIS":
        draw.line([(cx-12,y0+17),(cx-22,y0+31),(cx-17,y0+48),(cx-14,y1-13),
                   (cx+14,y1-13),(cx+17,y0+48),(cx+22,y0+31),(cx+12,y0+17)],fill=blue,width=2,joint="curve")
        for xx,yy in [(cx-8,y0+36),(cx+8,y0+36),(cx-8,y0+51),(cx+8,y0+51)]:
            draw.ellipse((xx-2,yy-2,xx+2,yy+2),fill=red)
        for bx,by,dx,dy in [(x0+11,y0+11,1,1),(x1-11,y0+11,-1,1),(x0+11,y1-11,1,-1),(x1-11,y1-11,-1,-1)]:
            draw.line((bx,by,bx+dx*11,by),fill=blue,width=3); draw.line((bx,by,bx,by+dy*11),fill=blue,width=3)
    elif tag=="DB":
        draw.rectangle((x0+24,y0+13,x1-22,y1-11),outline=blue,width=3)
        draw.line((x1-34,y0+14,x1-22,y0+26,x1-34,y0+26,x1-34,y0+14),fill=blue,width=2)
        draw.line([(x0+31,y0+42),(x0+39,y0+42),(x0+43,y0+33),(x0+48,y0+51),(x0+53,y0+38),(x1-28,y0+38)],fill=red,width=2)
        draw.line((x0+31,y0+57,x1-29,y0+57),fill=(143,170,198),width=2)
    else:
        for yy,knob in [(y0+23,x0+39),(y0+39,x0+58),(y0+55,x0+34)]:
            draw.line((x0+18,yy,x1-18,yy),fill=blue,width=2)
            draw.ellipse((knob-5,yy-5,knob+5,yy+5),fill=WHITE,outline=red if yy==y0+39 else blue,width=2)


def render_monitor(state, width, height):
    image = _base(width, height)
    draw = _header(image, state, "实时监测")
    lead = np.asarray(state.get("leads", np.empty((0, 12))), dtype=np.float64)
    signal = lead[:, 1] if lead.ndim == 2 and lead.shape[1] >= 2 else np.empty(0)
    hr = state.get("heart_rate")
    hr_text = "-- bpm" if hr is None or not np.isfinite(hr) else "{:.0f} bpm".format(hr)
    _text(draw, (width-38, 82), hr_text, 54, RED, True, anchor="ra")
    status, color = _status_text(state)
    _text(draw, (36, 147), status, 18, color, True)
    _plot_ecg(draw, signal, (36, 186, width-36, height-135), gain=1.0, grid=True, width=2)
    labels = ["12 导联", "频谱", "心率趋势", "AI 分析", "开始监测" if not state.get("monitor_running") else "停止监测"]
    button_w = (width-72-4*16)//5
    for i, label in enumerate(labels):
        x = 36+i*(button_w+16)
        _button(draw, (x, height-102, x+button_w, height-54), label,
                primary=(i==4), size=16)
    return image


def render_twelve(state, width, height):
    image = _base(width, height)
    draw = _header(image, state, "12 导联总览")
    data = np.asarray(state.get("leads", np.empty((0, 12))), dtype=np.float64)
    _text(draw, (width-36, 98), "当前高亮为选中导联 · 6 秒同步窗口", 16, MUTED, anchor="ra")
    margin, top, gap_x, gap_y = 36, 142, 18, 12
    card_w = (width-2*margin-2*gap_x)//3
    card_h = (height-top-28-3*gap_y)//4
    for i, name in enumerate(LEAD_NAMES):
        row, col = divmod(i, 3)
        x, y = margin+col*(card_w+gap_x), top+row*(card_h+gap_y)
        selected = i == int(state.get("focused_lead_index", 1))
        _card(draw, (x,y,x+card_w,y+card_h), 12,
              outline=BLUE if selected else LINE, width=2 if selected else 1)
        _text(draw, (x+10,y+6), name, 18, RED, True)
        signal = data[-600:, i] if data.ndim == 2 and data.shape[1] > i else np.empty(0)
        _plot_ecg(draw, signal, (x+10,y+30,x+card_w-10,y+card_h-9), gain=1.4, grid=True, width=1)
    return image


def render_lead_focus(state, width, height):
    image=_base(width,height)
    draw=_header(image,state,state.get("focused_lead_name","II 导联"),back="12 导联")
    data=np.asarray(state.get("leads",np.empty((0,12))),dtype=np.float64)
    index=int(state.get("focused_lead_index",1))
    signal=data[:,index] if data.ndim==2 and data.shape[1]>index else np.empty(0)
    _plot_ecg(draw,signal,(36,166,width-36,height-86),gain=1.0,grid=True,width=2)
    _text(draw,(width-38,135),"1000 SPS  ·  6 秒窗口",16,MUTED,anchor="ra")
    return image


def _spectrogram(signal, width=900, height=340, sample_rate=1000):
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if values.size < 128:
        return np.zeros((height, width), dtype=np.uint8)
    nfft, hop = 256, 50
    values = values[-max(nfft, min(values.size, sample_rate*10)):]
    starts = range(0, max(1, values.size-nfft+1), hop)
    window = np.hanning(nfft).astype(np.float32)
    columns = []
    for start in starts:
        chunk = values[start:start+nfft]
        if chunk.size != nfft:
            continue
        chunk = chunk - np.mean(chunk)
        columns.append(np.abs(np.fft.rfft(chunk*window)))
    if not columns:
        return np.zeros((height, width), dtype=np.uint8)
    spec = np.asarray(columns, dtype=np.float32).T
    max_bin = min(spec.shape[0], int(100*nfft/sample_rate)+1)
    spec = spec[:max_bin]
    db = 20*np.log10(spec+1e-5)
    lo, hi = np.percentile(db, (8, 99.5))
    scaled = np.clip((db-lo)/max(hi-lo, 1e-6), 0, 1)
    # Preserve the requested time x frequency intensity view without mpl.
    rgb = _heat_palette(scaled)
    src = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    src = src.transpose(Image.Transpose.FLIP_TOP_BOTTOM).resize((width,height), Image.Resampling.BILINEAR)
    return np.asarray(src)


def _heat_palette(values):
    stops = np.asarray([[40, 39, 103], [45, 100, 151], [31, 171, 148],
                        [115, 211, 82], [247, 224, 45]], dtype=np.float32)
    v = np.clip(values, 0, 1)*(len(stops)-1)
    lo = np.floor(v).astype(int)
    hi = np.minimum(lo+1,len(stops)-1)
    f = (v-lo)[...,None]
    return stops[lo]*(1-f)+stops[hi]*f


def render_spectrum(state, width, height):
    image = _base(width,height)
    draw = _header(image,state,"心电时频图",back="监测")
    data = np.asarray(state.get("leads", np.empty((0,12))),dtype=np.float64)
    signal = data[:,1] if data.ndim==2 and data.shape[1]>=2 else np.empty(0)
    _text(draw,(48,151),"II 导联 · 最近 6 s",14,NAVY,True)
    _plot_ecg(draw,signal[-6000:],(116,148,width-104,245),gain=1.0,grid=True,width=1)
    box=(116,288,width-122,height-126)
    _card(draw,(34,148,width-34,height-112),14)
    _text(draw,(48,270),"频率 (Hz)",14,MUTED)
    # Keep the unit label above the plot at its right edge, away from both
    # the color scale and the dB tick labels.
    _text(draw,(box[2],270),"PSD (dBFS/Hz)",14,MUTED,anchor="ra")
    matrix = spectrogram_dbfs(state)
    valid = matrix != SPECTROGRAM_INVALID
    if np.any(valid):
        scaled = np.clip((matrix.astype(np.float32)-SPECTROGRAM_DB_MIN) /
                         (SPECTROGRAM_DB_MAX-SPECTROGRAM_DB_MIN),0,1)
        rgb = _heat_palette(scaled)
        rgb[~valid] = np.asarray((220,227,234),dtype=np.float32)
        heat = Image.fromarray(rgb.astype(np.uint8),"RGB").resize(
            (box[2]-box[0],box[3]-box[1]),Image.Resampling.BILINEAR)
        image.paste(heat,(box[0],box[1]))
    for i, hz in enumerate((0,20,40,60,80,100)):
        y=box[3]-int(i*(box[3]-box[1])/5)
        draw.line((box[0],y,box[2],y),fill=(255,255,255),width=1)
        _text(draw,(box[0]-10,y),str(hz),13,MUTED,anchor="rm")
    for i, sec in enumerate((-10,-8,-6,-4,-2,0)):
        x=box[0]+int(i*(box[2]-box[0])/5)
        draw.line((x,box[1],x,box[3]),fill=(255,255,255),width=1)
        _text(draw,(x,box[3]+8),str(sec),13,MUTED,anchor="mt")
    # Fixed scale keeps colours comparable from one refresh to the next.
    bar_x = width-105
    for y in range(box[1],box[3]):
        value = SPECTROGRAM_DB_MAX-(y-box[1])*(SPECTROGRAM_DB_MAX-SPECTROGRAM_DB_MIN)/max(1,box[3]-box[1])
        rgb = _heat_palette(np.asarray((value-SPECTROGRAM_DB_MIN)/(SPECTROGRAM_DB_MAX-SPECTROGRAM_DB_MIN)))[0:3]
        draw.line((bar_x,y,bar_x+14,y),fill=tuple(int(v) for v in rgb))
    _text(draw,((box[0]+box[2])//2,height-91),"相对时间 (s)",15,NAVY,anchor="mt")
    _text(draw,(48,height-62),"Hann 512点 · 0.512 s · 步长64 ms · 频格1.95 Hz",13,MUTED)
    return image


def render_ai(state,width,height):
    image=_base(width,height)
    draw=_header(image,state,"AI 分析")
    result=state.get("ai_result")
    backend=state.get("ai_backend_mode","ps")
    _button(draw,(602,100,782,146),"PS NumPy",primary=backend=="ps",size=16)
    _button(draw,(790,100,995,146),"FPGA + DMA",primary=backend=="pl",size=16)
    _card(draw,(34,143,418,356),16)
    _text(draw,(54,162),"模型推理",20,BLUE,True)
    if result:
        probs=result["probabilities"]
        top=max(probs,key=probs.get)
        _text(draw,(54,204),top,42,BLUE,True)
        _text(draw,(54,255),"最高类别得分",16,MUTED)
        _text(draw,(282,246),"{:.0f}%".format(probs[top]*100),30,BLUE,True)
    else:
        _text(draw,(54,214),"尚无推理结果",28,NAVY,True)
        _text(draw,(54,266),"采集到稳定的 10 秒心电后可开始推理",15,MUTED)
    _text(draw,(455,154),"五分类得分",20,NAVY,True)
    labels=("正常 NORM","心肌梗死 MI","ST-T 改变 STTC","传导异常 CD","心室肥厚 HYP")
    probs=result["probabilities"] if result else {}
    class_keys=("NORM","MI","STTC","CD","HYP")
    for i,(label,key) in enumerate(zip(labels,class_keys)):
        y=201+i*31
        value=float(probs.get(key,0.0))
        _text(draw,(455,y),label,16,MUTED)
        draw.rounded_rectangle((610,y+4,1144,y+19),radius=7,fill=(232,239,247))
        if result:
            draw.rounded_rectangle((610,y+4,610+int(534*np.clip(value,0,1)),y+19),radius=7,
                                   fill=BLUE if i==0 else (143,170,198))
            _text(draw,(1157,y-1),"{:.0f}%".format(value*100),15,NAVY)
    _card(draw,(34,381,width-34,height-80),16)
    _text(draw,(54,399),"模型输入 · 代表性片段",20,NAVY,True)
    data=np.asarray(state.get("leads",np.empty((0,12))),dtype=np.float64)
    for i,name in enumerate(LEAD_NAMES):
        row,col=divmod(i,3)
        x=52+col*390
        y=430+row*53
        _text(draw,(x,y),name,15,BLUE,True)
        signal=data[-1000:,i] if data.ndim==2 and data.shape[1]>i else np.empty(0)
        draw.rounded_rectangle((x+42,y-2,x+365,y+43),radius=7,fill=WHITE,outline=LINE)
        pts=_wave_points(signal,(x+47,y+3,x+360,y+38),gain=1.25)
        if len(pts)>1:
            draw.line(pts,fill=RED,width=1)
    if result:
        ps_time=result.get("ps_seconds")
        pl_time=result.get("pl_seconds")
        speedup=result.get("speedup")
        timing="PS {}    FPGA+DMA {}    ".format(
            "{:.3f}s".format(ps_time) if ps_time is not None else "--",
            "{:.3f}s".format(pl_time) if pl_time is not None else "--")
        if speedup is not None:
            timing+="加速 {:.2f}×    ".format(speedup)
        _text(draw,(52,height-62),timing.rstrip(),16,BLUE,True)
    golden=state.get("golden_status","未验证")
    golden_color=GREEN if state.get("golden_passed") else (RED if golden.startswith("失败") else MUTED)
    _text(draw,(54,340),"硬件自检："+golden,15,golden_color,True)
    if state.get("ai_error"):
        _text(draw,(455,340),state["ai_error"][:50],14,RED,True)
    _button(draw,(width-252,82,width-34,126),"运行 AI 分析",primary=True,size=16)
    return image


def render_trend(state,width,height):
    image=_base(width,height)
    draw=_header(image,state,"心率趋势")
    _card(draw,(34,148,width-34,height-46),16)
    box=(94,202,width-82,height-122)
    draw.rectangle(box,fill=WHITE,outline=LINE,width=1)
    for bpm in (40,60,80,100,120):
        y=box[3]-int((bpm-40)/80*(box[3]-box[1]))
        draw.line((box[0],y,box[2],y),fill=(232,239,247),width=1)
        _text(draw,(box[0]-10,y),str(bpm),13,MUTED,anchor="rm")
    values=trend_points(state)
    valid=values != TREND_INVALID
    if np.count_nonzero(valid)>=2:
        segment=[]
        for i,value in enumerate(values):
            if value == TREND_INVALID:
                if len(segment)>1: draw.line(segment,fill=BLUE,width=3,joint="curve")
                segment=[]
                continue
            x=box[0]+3+int(i*(box[2]-box[0]-6)/(values.size-1))
            y=box[3]-int(np.clip((value-40)/80,0,1)*(box[3]-box[1]))
            segment.append((x,y))
        if len(segment)>1: draw.line(segment,fill=BLUE,width=3,joint="curve")
        last=values[np.flatnonzero(valid)[-1]]
        _text(draw,(width-56,158),"{} bpm".format(int(last)),25,BLUE,True,anchor="ra")
    else:
        _text(draw,((box[0]+box[2])//2,(box[1]+box[3])//2),"等待 QRS 心率数据",20,MUTED,anchor="mm")
    _text(draw,((box[0]+box[2])//2,height-96),"逐搏心率 · 最近120 s · 缺失数据留空",15,MUTED,anchor="mt")
    return image


def render_simple(state,width,height,page):
    titles={"vision":"视觉贴片引导","history":"历史数据库","settings":"设置"}
    if page == "vision":
        return render_vision(state, width, height)
    image=_base(width,height)
    draw=_header(image,state,titles[page])
    _card(draw,(34,151,width-34,height-34),16)
    if page=="vision":
        headline="R9 当前未接入 USB 摄像头识别"
        detail="视觉引导需要额外的相机采集与人体 / 电极模型。"
    elif page=="history":
        rows=list(state.get("history_records") or ())
        selected_id=state.get("history_selected_id")
        detail=state.get("history_detail") or {}
        left=(50,181,814,648)
        right=(832,181,width-52,648)
        _card(draw,left,12)
        _card(draw,right,12)
        _text(draw,(72,198),"真实采集记录 · 最近 {} 条".format(len(rows)),20,BLUE,True)
        for index,row in enumerate(rows[:4]):
            y=270+index*82
            selected=int(row.get("session_id",-1))==selected_id
            _card(draw,(64,y,800,y+72),8,fill=BLUE_LIGHT if selected else WHITE,
                  outline=BLUE if selected else LINE)
            started=str(row.get("started_at") or "").replace("T"," ")[:16]
            bpm=row.get("avg_heart_rate")
            bpm_text="-- bpm" if bpm is None else "{:.0f} bpm".format(float(bpm))
            result=str(row.get("result_label") or "未分析")
            _text(draw,(80,y+10),"PA-{:06d}".format(int(row.get("session_id",0))),16,NAVY,True)
            _text(draw,(226,y+10),started,15,MUTED)
            _text(draw,(500,y+10),bpm_text,15,NAVY,True)
            _text(draw,(650,y+10),result[:12],15,BLUE,True)
            _text(draw,(80,y+42),"{} 导联 · {:,} 个样本 · 质量异常 {} 帧".format(
                int(row.get("lead_count",12)),int(row.get("sample_count",0)),
                int(row.get("quality_bad_frames",0))),13,MUTED)
        if not rows:
            _text(draw,(430,386),"暂无真实采集记录",20,MUTED,True,anchor="mm")
            _text(draw,(430,418),"开始并停止一次 ECG 监测后，记录会自动保存。",15,MUTED,anchor="mm")
        _text(draw,(854,198),"选中记录波形预览",20,BLUE,True)
        preview=np.asarray(detail.get("preview",()),dtype=np.float64)
        _plot_ecg(draw,preview,(855,242,width-76,394),gain=1.0,grid=True,width=1)
        if detail:
            _text(draw,(855,418),"状态：{}".format(detail.get("status","--")),15,NAVY,True)
            _text(draw,(855,448),"采样率：{:.0f} SPS · 保存 {} 点".format(
                float(detail.get("sample_rate",0)),len(preview)),14,MUTED)
            result=str(detail.get("result_label") or "未分析")
            confidence=detail.get("result_confidence")
            if confidence is not None:
                result+=" · {:.0f}%".format(float(confidence)*100)
            _text(draw,(855,478),"AI 结果："+result,15,NAVY)
            if int(detail.get("dropped_chunks",0)):
                _text(draw,(855,516),"有 {} 个存储块丢失，此记录不完整。".format(
                    int(detail["dropped_chunks"])),14,RED,True)
        else:
            _text(draw,(855,418),"选择左侧的一条采集记录查看详情。",15,MUTED)
        return image
    elif page=="settings":
        _text(draw,(72,195),"采集与显示",24,NAVY,True)
        _text(draw,(72,278),"经典滤波",19,MUTED)
        filter_on=bool(state.get("filter_enabled",True))
        _button(draw,(190,465,270,515),"开启" if filter_on else "关闭",
                primary=filter_on,size=16)
        _text(draw,(72,392),"波形增益",19,MUTED)
        gain=float(state.get("gain",1.0))
        _text(draw,(650,382),"×{:.2g}".format(gain),28,BLUE,True)
        _button(draw,(850,445,895,490),"−",size=20)
        _button(draw,(910,445,955,490),"+",size=20)
        _text(draw,(72,545),"滤波切换只影响显示；AI 分析前会自动恢复标准滤波。",15,MUTED)
        return image
    else:
        status=state.get("status",{})
        headline="ECG 采集设备 {}".format("已连接" if status.get("id_ok") else "等待连接")
        detail="经典滤波 + 小波 + 12 导联 + QRS；波形采样率 1000 SPS。"
        _text(draw,(72,218),headline,30,NAVY,True)
    _text(draw,(72,274),detail,19,MUTED)
    _text(draw,(72,342),"页面外观接入 PulseAtlas；本页功能不属于当前 R9 完成范围。",16,BLUE)
    return image


def render_vision(state, width, height):
    """Render the PS camera stream and honest live inference status."""
    image=_base(width,height)
    draw=_header(image,state,"视觉贴片引导")
    _card(draw,(34,151,width-34,height-34),16)

    preview=(58,174,708,662)
    draw.rounded_rectangle(preview,radius=12,fill=(26,39,55),outline=LINE,width=1)
    frame=state.get("vision_frame_bgr")
    if frame is not None:
        rgb=np.ascontiguousarray(np.asarray(frame)[:,:,::-1])
        preview_image=Image.fromarray(rgb)
        max_size=(preview[2]-preview[0]-12,preview[3]-preview[1]-12)
        resampling=getattr(Image,"Resampling",Image).BILINEAR
        preview_image.thumbnail(max_size,resampling)
        x=preview[0]+(preview[2]-preview[0]-preview_image.width)//2
        y=preview[1]+(preview[3]-preview[1]-preview_image.height)//2
        image.paste(preview_image,(x,y))
    else:
        status=state.get("vision_status") or "按 BTN2 启动视觉识别"
        _text(draw,((preview[0]+preview[2])//2,(preview[1]+preview[3])//2),
              status,22,WHITE,True,anchor="mm")

    panel_x=744
    _text(draw,(panel_x,183),"摄像头预览与模型推理",26,NAVY,True)
    active=bool(state.get("vision_active",False))
    connected=bool(state.get("vision_camera_connected",False))
    inference_running=bool(state.get("vision_inference_running",False))
    replay=state.get("vision_source_mode") == "recorded_demo"
    run_status=("演示视频播放中" if replay and active else
                "相机在线 · 模型正在推理" if connected and inference_running else
                "相机在线 · 等待下一次推理" if connected and active else
                state.get("vision_status") or "等待启动")
    _text(draw,(panel_x,229),run_status,19,
          GREEN if connected else MUTED,True)

    fps=float(state.get("vision_fps") or 0.0)
    camera_fps=float(state.get("vision_camera_fps") or 0.0)
    preview_fps=float(state.get("vision_preview_fps") or 0.0)
    inference_ms=state.get("vision_inference_ms")
    _text(draw,(panel_x,275),"HDMI 预览刷新",16,MUTED)
    _text(draw,(panel_x+220,275),"摄像头采集",16,MUTED)
    preview_text="-- FPS" if preview_fps <= 0 else "{:.1f} FPS".format(preview_fps)
    camera_text="-- FPS" if camera_fps <= 0 else "{:.1f} FPS".format(camera_fps)
    _text(draw,(panel_x,301),preview_text,25,BLUE,True)
    _text(draw,(panel_x+220,301),camera_text,25,NAVY,True)

    _text(draw,(panel_x,345),"模型推理",16,MUTED)
    _text(draw,(panel_x+220,345),"单帧耗时",16,MUTED)
    latency="--" if inference_ms is None else "{:.0f} ms".format(float(inference_ms))
    fps_text="-- FPS" if inference_ms is None else "{:.2f} FPS".format(fps)
    _text(draw,(panel_x,371),fps_text,25,BLUE,True)
    _text(draw,(panel_x+220,371),latency,24,NAVY,True)

    confidence=state.get("vision_confidence")
    confidence_text="--" if confidence is None else "{:.0f}%".format(100*float(confidence))
    marker_count=int(state.get("vision_marker_count") or 0)
    result_status=state.get("vision_result_status") or "等待推理"
    if state.get("vision_pose_valid"):
        result_status="定位有效 · {} 个标记".format(marker_count)
    age_ms=state.get("vision_result_age_ms")
    age_text="--" if age_ms is None else "{:.1f} 秒前".format(float(age_ms)/1000.0)
    inference_count=int(state.get("vision_inference_count") or 0)
    progress=("记录帧 · #{}".format(inference_count) if replay else
              "进行中 · #{}".format(inference_count) if inference_running else
              "已完成 {} 次".format(inference_count))
    _text(draw,(panel_x,414),"模型任务",16,MUTED)
    _text(draw,(panel_x+220,414),"最近结果 / 年龄",16,MUTED)
    _text(draw,(panel_x,440),progress,19,NAVY,True)
    result_text="{} · {}".format(result_status,age_text)
    _text(draw,(panel_x+220,440),result_text[:25],15,NAVY,True)

    _card(draw,(panel_x,480,width-62,553),10,fill=(239,245,251),outline=LINE)
    _text(draw,(panel_x+16,490),"ECG PL 采集与推理保持原样",16,BLUE,True)
    _text(draw,(panel_x+16,518),"标记来自最近一次 YOLO 定位；移动后请等新结果。",13,MUTED)
    error=state.get("vision_error")
    if error:
        _text(draw,(panel_x+16,540),str(error)[:56],12,RED)

    button=(panel_x,575,width-62,625)
    _button(draw,button,"停止视觉识别" if active else "启动视觉识别",
            primary=not active,size=18)
    _text(draw,(panel_x,641),"模型结果仅供辅助参考，不作为临床诊断。",14,MUTED)
    if state.get("vision_source_mode") == "recorded_demo":
        layer=Image.new("RGBA",image.size,(0,0,0,0))
        badge=ImageDraw.Draw(layer)
        label="演示回放 · 非实时推理"
        font=FONTS.get(12,False)
        box=badge.textbbox((0,0),label,font=font)
        tw=box[2]-box[0]
        x=width-tw-42
        y=height-29
        badge.rounded_rectangle((x-8,y-4,width-26,height-10),radius=5,
                                fill=(23,34,48,108))
        badge.text((x,y),label,font=font,fill=(255,255,255,188))
        image.paste(layer,(0,0),layer)
    return image


def render_page(page,state,width=1280,height=720):
    pages={"home":render_home,"monitor":render_monitor,"leads":render_twelve,
           "lead_focus":render_lead_focus,"spectrum":render_spectrum,"ai":render_ai,
           "trend":render_trend}
    if page in pages:
        image = pages[page](state,int(width),int(height))
    elif page in ("vision","history","settings"):
        image = render_simple(state,int(width),int(height),page)
    else:
        image = render_home(state,int(width),int(height))
    _draw_button_hint(ImageDraw.Draw(image), page, state, int(width), int(height))
    return image


def _draw_button_hint(draw, page, state, width, height):
    if page == "home":
        hint = "BTN0/BTN1 选择    BTN2 确认进入"
    elif page == "monitor":
        hint = "BTN0/BTN1 切换页面    BTN2 开始/停止采集    BTN3 返回首页"
    elif page == "leads":
        hint = "BTN0/BTN1 选择导联    BTN2 放大    BTN3 返回首页"
    elif page == "lead_focus":
        hint = "BTN0/BTN1 切换导联    BTN3 返回12导联"
    elif page == "ai":
        hint = "BTN0/BTN1 切换页面    BTN2 运行分析    BTN3 返回首页"
    elif page == "vision":
        hint = "BTN2 启动/停止视觉识别    BTN3 返回首页"
    else:
        hint = "BTN0/BTN1 切换页面    BTN3 返回首页"
    if not state.get("buttons_available", False):
        hint = "按键GPIO未启用 · " + hint
    _text(draw, (width//2, height-15), hint, 14, MUTED, anchor="mm")
