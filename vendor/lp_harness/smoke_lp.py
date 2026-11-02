import sys, os, time, json, numpy as np, cv2, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LivePortrait'))
from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapperAnimal
from src.utils.camera import get_rotation_matrix
from src.utils.helper import calc_motion_multiplier

DEV = sys.argv[1] if len(sys.argv) > 1 else 'xpu'
NF = int(sys.argv[2]) if len(sys.argv) > 2 else 45
HALF = (sys.argv[3] != 'fp32') if len(sys.argv) > 3 else True

cfg = InferenceConfig(flag_force_cpu=True, flag_use_half_precision=HALF, flag_stitching=False, flag_pasteback=False, flag_do_crop=False)
t0 = time.time()
w = LivePortraitWrapperAnimal(cfg)   # loads on CPU with torch.load defaults (weights_only=True on torch>=2.6)
print('load s', round(time.time()-t0, 1), 'torch', torch.__version__)
if DEV != 'cpu':
    for m in [w.appearance_feature_extractor, w.motion_extractor, w.warping_module, w.spade_generator]:
        m.to(DEV)
    w.device = DEV   # 'xpu' -> device[:4] == 'xpu' works for autocast
print('device', w.device, 'half', HALF)

# ---- driver: RIGHT performer 1:1 crops from work/cut.mp4 around 9-12 s using analysis.json tracks (15 Hz)
# RECAST_ROOT (the recast/ dir holding work/) or, by default, three levels up from this file
root = os.environ.get('RECAST_ROOT') or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
a = json.load(open(os.path.join(root, 'work/analysis.json')))
R = {round(r['t'], 3): r for r in a['tracks']['R'] if r}
cap = cv2.VideoCapture(os.path.join(root, 'work/cut.mp4'))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
idx = 0
while len(frames) < NF:
    ok, fr = cap.read()
    if not ok: break
    t = idx / fps
    idx += 1
    if t < 9.0 or idx % 2: continue
    key = min(R.keys(), key=lambda k: abs(k - t))
    r = R[key]
    side = int(2.2 * r['w'])
    cx, cy = int(r['cx']), int(r['cy'])
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = fr[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (256, 256))
    frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
print('driver frames', len(frames), 'native crop side px', side)
# source: use first driver frame as the "portrait" (content irrelevant for timing/feasibility)
src = frames[0]

with torch.no_grad():
    I_s = w.prepare_source(src)
    x_s_info = w.get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    f_s = w.extract_feature_3d(I_s)
    x_s = w.transform_keypoint(x_s_info)
    I_d = w.prepare_videos(frames)
    # motion template
    tm = time.time()
    motions = []
    for i in range(len(frames)):
        info = w.get_kp_info(I_d[i])
        info['R'] = get_rotation_matrix(info['pitch'], info['yaw'], info['roll'])
        motions.append(info)
    if DEV == 'xpu': torch.xpu.synchronize()
    t_me = (time.time() - tm) / len(frames)
    print('motion extractor s/frame', round(t_me, 4))
    outs = []
    times = []
    for i, info in enumerate(motions):
        t1 = time.time()
        R_d_i = info['R']; delta_new = info['exp']; t_new = info['t']; t_new[..., 2].fill_(0)
        x_d_i = x_s_info['scale'] * (x_c_s @ R_d_i + delta_new) + t_new
        if i == 0:
            x_d_0 = x_d_i
            mm = calc_motion_multiplier(x_s, x_d_0)
        x_d_i = (x_d_i - x_d_0) * mm + x_s
        x_d_i = x_s + (x_d_i - x_s) * 1.75
        out = w.warp_decode(f_s, x_s, x_d_i)
        img = w.parse_output(out['out'])[0]
        if DEV == 'xpu': torch.xpu.synchronize()
        times.append(time.time() - t1)
        outs.append(img)
    times = np.array(times)
    print('warp_decode s/frame: first', round(times[0], 3), 'median(rest)', round(float(np.median(times[1:])), 4), 'mean(rest)', round(float(times[1:].mean()), 4))
    print('TOTAL est s/frame (motion+warp_decode, steady)', round(t_me + float(np.median(times[1:])), 4))
    o = np.stack(outs)
    print('out shape', o.shape, 'dtype', o.dtype, 'min/max', o.min(), o.max(), 'nan', np.isnan(o.astype(np.float32)).any())
    print('mean abs diff between frame 0 and last output (motion present?)', round(float(np.abs(o[0].astype(int) - o[-1].astype(int)).mean()), 2))
    sheet = np.concatenate([cv2.resize(cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR), (256, 256)) for k in [0, len(frames)//2, -1]], 1)
    sheet2 = np.concatenate([cv2.resize(cv2.cvtColor(outs[k], cv2.COLOR_RGB2BGR), (256, 256)) for k in [0, len(frames)//2, -1]], 1)
    cv2.imwrite(f'smoke_{DEV}_sheet.jpg', np.concatenate([sheet, sheet2], 0))
    if DEV == 'xpu':
        print('xpu max mem MB', round(torch.xpu.max_memory_allocated() / 2**20))
