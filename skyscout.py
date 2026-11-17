'''SkyScout: fine-tune and benchmark YOLO and D-FINE for small objects in VisDrone drone images.

Terminal usage:
    python skyscout.py status
    python skyscout.py all
    python skyscout.py run dfine_full_640
    python skyscout.py eval yolo11s_640 --split test --sahi
    python skyscout.py time dfine_full_640
    python skyscout.py report
    python skyscout.py demo

All settings live in config.yaml. Every step writes its state to results/progress so a run can
continue after a disconnect or on another machine.
'''
import os

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import contextlib
import csv
import io
import json
import platform
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / 'downloads'
RUNS = ROOT / 'runs'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
PROGRESS = RESULTS / 'progress'
PLOTS = RESULTS / 'plots'
os.environ.setdefault('KAGGLEHUB_CACHE', str(DOWNLOADS / 'kagglehub'))

# Validated categorical palette (fixed order) plus text and grid tokens for plots
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BOX_COLORS = PALETTE + ['#00a3c4', '#8a5a00']
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'


def load_config(path=ROOT / 'config.yaml'):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def detect_profile():
    forced = os.environ.get('SKYSCOUT_PROFILE')
    if forced:
        return forced
    if 'google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ:
        return 'colab'
    return 'server' if platform.system() == 'Linux' else 'local'


CFG = load_config()
PROFILE = detect_profile()
PROF = CFG['profiles'][PROFILE]
NAMES = CFG['dataset']['names']
NC = len(NAMES)


def reload_config():
    '''Re-read config.yaml after editing it in a running notebook.'''
    global CFG, PROF
    CFG = load_config()
    PROF = CFG['profiles'][PROFILE]
    return CFG


def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return rel(o)
    return str(o)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_text(path, text):
    # Write to a temp file first so an interruption never leaves a half-written file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def write_json(path, data):
    write_text(path, json.dumps(data, indent=2, default=_json_default))


def log(msg):
    line = f'[{now()}] [{PROFILE}] {msg}'
    print(line)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / 'log.txt', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def seed_everything(seed=None):
    seed = CFG['seed'] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if 'torch' in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if 'keras' in sys.modules:
        import keras
        keras.utils.set_random_seed(seed)
    return seed


def torch_device():
    try:
        import torch
    except ImportError:
        return 'cpu'
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, 'mps', None)
    return 'mps' if mps is not None and mps.is_available() else 'cpu'


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    if 'tensorflow' in sys.modules:
        tf = sys.modules['tensorflow']
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'GPU')
    return 'cpu'


def environment():
    vers = {'python': platform.python_version()}
    for m in ('numpy', 'tensorflow', 'keras', 'keras_hub', 'torch', 'ultralytics'):
        if m in sys.modules:
            vers[m] = getattr(sys.modules[m], '__version__', '?')
    return {'profile': PROFILE, 'os': platform.system(), 'gpu': gpu_name(), 'versions': vers, 'time': now()}


def fit_verbose():
    # Progress bars flood log files, so use one line per epoch outside notebooks
    return 1 if ('ipykernel' in sys.modules or sys.stdout.isatty()) else 2


# Progress tracking: one small json per step, so folders from different machines merge cleanly

def step_status(name):
    return read_json(PROGRESS / f'{name}.json', {})


def is_done(name):
    return step_status(name).get('status') == 'done'


def mark(name, status, **extra):
    d = step_status(name)
    if status == 'running' and d.get('status') != 'running':
        d['started'] = now()
        d.pop('error', None)
    d.update(extra)
    d.update(step=name, status=status, profile=PROFILE, updated=now())
    if status == 'done':
        d['finished'] = now()
    write_json(PROGRESS / f'{name}.json', d)
    if not extra or status != 'running':
        log(f'{name}: {status}')
    write_status()


def run_step(name, fn, *args, redo=False, **kwargs):
    '''Run fn once and record it. Skips steps already marked done unless redo=True (passed on to fn).'''
    if is_done(name) and not redo:
        print(f'Skip {name}: already done')
        return None
    mark(name, 'running')
    try:
        out = fn(*args, redo=redo, **kwargs)
    except KeyboardInterrupt:
        mark(name, 'interrupted')
        raise
    except Exception as e:
        mark(name, 'failed', error=f'{type(e).__name__}: {e}'[:500])
        raise
    mark(name, 'done')
    return out


def experiment(name):
    if name not in CFG['experiments']:
        raise KeyError(f'Unknown experiment {name}, check config.yaml')
    return {'name': name, **CFG['experiments'][name]}


def allowed_here(e):
    return PROFILE in e.get('run_on', list(CFG['profiles']))


def model_path(name):
    ext = '.pt' if experiment(name)['family'] == 'yolo' else '.weights.h5'
    return MODELS / f'{name}{ext}'


def eval_dir(name, split, sahi=False):
    return RESULTS / name / (f'{split}_sahi' if sahi else split)


def metrics_for(name, split, sahi=False):
    return read_json(eval_dir(name, split, sahi) / 'metrics.json')


def best_dfine():
    '''Name of the D-FINE setup with the best val mAP50-95, or None if any is missing.'''
    split = CFG['evaluation']['select_split']
    scores = {n: (metrics_for(n, split) or {}).get('map50_95') for n in CFG['dfine_setup_compare']}
    if any(v is None for v in scores.values()):
        return None
    return max(scores, key=scores.get)


def resolve_name(name):
    return best_dfine() if name == 'best' else name


def resolve_setup(e):
    if e.get('setup') != 'best':
        return e['setup'], None
    best = best_dfine()
    if best is None:
        need = ', '.join(CFG['dfine_setup_compare'])
        raise RuntimeError(f'Recommended: train and evaluate {need} first, {e["name"]} uses the best of them')
    return experiment(best)['setup'], best


def batch_for(family, imgsz):
    table = PROF[f'{family}_batch']
    return table.get(imgsz) or table.get(str(imgsz)) or min(table.values())


def save_run_config(name, params):
    path = RUNS / name / 'run_config.json'
    d = read_json(path, {'experiment': name, 'sessions': []})
    d['params'] = params
    d['config'] = CFG['experiments'][name]
    d['sessions'].append(environment())
    write_json(path, d)


# Status file for humans (and for picking up work on another machine)

def write_status():
    rows = ['| Experiment | Run on | Model saved | Val mAP50-95 | Test mAP50-95 | Test SAHI | Train step |',
            '|---|---|---|---|---|---|---|']

    def fmt(m):
        return f'{100 * m["map50_95"]:.2f}' if m else '-'

    for name, e in CFG['experiments'].items():
        st = step_status(f'train_{name}')
        train = st.get('status', '-')
        if st.get('epoch'):
            train += f' (epoch {st["epoch"]}/{st.get("epochs", "?")})'
        rows.append(f'| {name} | {", ".join(e.get("run_on", ["any"]))} | {"yes" if model_path(name).exists() else "no"} | '
                    f'{fmt(metrics_for(name, "val"))} | {fmt(metrics_for(name, "test"))} | '
                    f'{fmt(metrics_for(name, "test", True))} | {train} |')
    recent = sorted(PROGRESS.glob('*.json'), key=lambda p: p.stat().st_mtime)[-12:] if PROGRESS.exists() else []
    steps = [f'- {d.get("updated")} `{d.get("step")}` {d.get("status")} on {d.get("profile")}'
             + (f' ({d["error"]})' if d.get('error') else '') for d in (read_json(p, {}) for p in reversed(recent))]
    text = ('### SkyScout status\n\nAuto-generated by skyscout.py, do not edit. '
            f'Updated {now()} on profile `{PROFILE}`.\n\n'
            'A step left as `running` with an old date was cut off (for example a Colab disconnect). '
            'Run it again and it continues from the last saved epoch.\n\n'
            + '\n'.join(rows) + '\n\n#### Recent steps\n\n' + ('\n'.join(steps) or '- none yet') + '\n')
    write_text(RESULTS / 'STATUS.md', text)
    return text


def show_status():
    text = write_status()
    try:
        from IPython.display import Markdown, display
        if 'ipykernel' in sys.modules:
            display(Markdown(text))
            return None
    except ImportError:
        pass
    print(text)
    return None


# Downloads and dataset

def download(url, dest):
    '''Download once into downloads/. Partial files are resumed.'''
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
    print(f'Downloading {dest.name}')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if start and r.status != 206:
                start = 0
            total = int(r.headers.get('Content-Length') or 0) + start
            done, shown = start, start
            with open(part, 'ab' if start else 'wb') as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done - shown >= total / 10:
                        print(f'  {100 * done / total:.0f}% of {total / 1e6:.0f} MB')
                        shown = done
    except urllib.error.HTTPError as e:
        if e.code != 416:
            raise
    os.replace(part, dest)
    return dest


def data_dir():
    if PROFILE == 'colab':
        return Path(CFG['paths']['colab_fast_data_dir']) / 'VisDrone'
    return DOWNLOADS / 'VisDrone'


def split_dir(split):
    return data_dir() / CFG['dataset']['splits'][split]


def prepare_dataset():
    '''Download, extract and convert VisDrone. Safe to call many times, finished parts are skipped.'''
    base = CFG['dataset']['base_url']
    for split, folder in CFG['dataset']['splits'].items():
        sdir = split_dir(split)
        if not (sdir / 'coco.json').exists():
            if not (sdir / '.extracted').exists():
                zpath = download(f'{base}/{folder}.zip', DOWNLOADS / 'zips' / f'{folder}.zip')
                print(f'Extracting {zpath.name} to {sdir.parent}')
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(sdir.parent)
                if not (sdir / 'images').exists():
                    raise FileNotFoundError(f'{sdir / "images"} missing after extracting {zpath.name}')
                (sdir / '.extracted').write_text(now())
            convert_split(sdir)
    return data_yaml()


def convert_split(sdir):
    '''VisDrone txt annotations -> YOLO labels and one COCO json (used by D-FINE and the evaluator).'''
    from PIL import Image
    drop = set(CFG['dataset']['drop_categories'])
    (sdir / 'labels').mkdir(exist_ok=True)
    images, anns = [], []
    files = sorted((sdir / 'annotations').glob('*.txt'))
    print(f'Converting {len(files)} annotation files in {sdir.name}')
    for iid, ann_file in enumerate(files, 1):
        img_path = sdir / 'images' / f'{ann_file.stem}.jpg'
        with Image.open(img_path) as im:
            w, h = im.size
        lines = []
        for row in ann_file.read_text().strip().splitlines():
            v = [x for x in row.strip().split(',') if x != '']
            if len(v) < 6:
                continue
            x, y, bw, bh, score, cat = map(int, v[:6])
            x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
            if cat in drop or score == 0 or x2 <= x1 or y2 <= y1:
                continue
            c = cat - 1
            lines.append(f'{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': c + 1,
                         'bbox': [x1, y1, x2 - x1, y2 - y1], 'area': (x2 - x1) * (y2 - y1), 'iscrowd': 0})
        (sdir / 'labels' / f'{ann_file.stem}.txt').write_text('\n'.join(lines))
        images.append({'id': iid, 'file_name': img_path.name, 'width': w, 'height': h})
    cats = [{'id': i + 1, 'name': n} for i, n in enumerate(NAMES)]
    write_json(sdir / 'coco.json', {'images': images, 'annotations': anns, 'categories': cats})


def data_yaml():
    s = CFG['dataset']['splits']
    d = {'path': data_dir().as_posix(), 'names': dict(enumerate(NAMES))}
    d.update({k: f'{v}/images' for k, v in s.items()})
    path = data_dir() / 'skyscout_data.yaml'
    write_text(path, yaml.safe_dump(d, sort_keys=False))
    return path


_COCO = {}


def load_coco(split):
    if split not in _COCO:
        path = split_dir(split) / 'coco.json'
        if not path.exists():
            prepare_dataset()
        _COCO[split] = read_json(path)
    return _COCO[split]


def gt_by_image(coco):
    out = {im['id']: ([], []) for im in coco['images']}
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        out[a['image_id']][0].append([x, y, x + w, y + h])
        out[a['image_id']][1].append(a['category_id'] - 1)
    return {k: (np.array(b, 'float32').reshape(-1, 4), np.array(c, int)) for k, (b, c) in out.items()}


def read_rgb(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dataset_summary():
    '''Object counts and size mix per split, saved to results/dataset_summary.md.'''
    lines = ['| Split | Images | Objects | Objects per image | Small (<32 px) | Medium | Large |', '|---|---|---|---|---|---|---|']
    per_class = {}
    for split in CFG['dataset']['splits']:
        coco = load_coco(split)
        areas = np.array([a['area'] for a in coco['annotations']], float)
        n, k = len(coco['images']), len(areas)
        small, med = np.mean(areas < 32 ** 2), np.mean((areas >= 32 ** 2) & (areas < 96 ** 2))
        lines.append(f'| {split} | {n} | {k} | {k / max(n, 1):.1f} | {100 * small:.1f}% | {100 * med:.1f}% | {100 * (1 - small - med):.1f}% |')
        per_class[split] = np.bincount([a['category_id'] - 1 for a in coco['annotations']], minlength=NC)
    lines += ['', '| Class | ' + ' | '.join(per_class) + ' |', '|---|' + '---|' * len(per_class)]
    for k, n in enumerate(NAMES):
        lines.append(f'| {n} | ' + ' | '.join(str(v[k]) for v in per_class.values()) + ' |')
    text = '### VisDrone2019-DET summary\n\n' + '\n'.join(lines) + '\n'
    write_text(RESULTS / 'dataset_summary.md', text)
    print(text)
    return text


# YOLO (Ultralytics, PyTorch)

def ultralytics_setup():
    from ultralytics import settings
    (DOWNLOADS / 'weights').mkdir(parents=True, exist_ok=True)
    settings.update({'weights_dir': str(DOWNLOADS / 'weights'), 'runs_dir': str(RUNS), 'datasets_dir': str(DOWNLOADS)})


def yolo_weights(fname):
    '''COCO-pretrained YOLO weights, downloaded once into downloads/weights.'''
    from ultralytics.utils.downloads import attempt_download_asset
    ultralytics_setup()
    dest = DOWNLOADS / 'weights' / fname
    if not dest.exists():
        attempt_download_asset(str(dest))
    return dest


def yolo_ckpt_state(last):
    if not last.exists():
        return 'missing'
    import torch
    ck = torch.load(last, map_location='cpu', weights_only=False)
    return 'resumable' if ck.get('optimizer') is not None and ck.get('epoch', -1) >= 0 else 'finished'


def patch_yolo_ckpt(last, updates):
    # Checkpoints store absolute paths, rewrite them so a run can resume on a different machine
    import torch
    ck = torch.load(last, map_location='cpu', weights_only=False)
    args = ck.get('train_args') or {}
    if 'save_dir' in args:
        updates = {**updates, 'save_dir': str(last.parent.parent)}
    if all(args.get(k) == v for k, v in updates.items()):
        return
    args.update(updates)
    ck['train_args'] = args
    torch.save(ck, last)


def yolo_args(e):
    hp = {**CFG['yolo'], **e.get('overrides', {})}
    hp.update({k: e[k] for k in ('epochs', 'fraction') if k in e})
    imgsz = e['imgsz']
    return {**hp, 'data': str(prepare_dataset()), 'imgsz': imgsz, 'batch': batch_for('yolo', imgsz),
            'workers': PROF['workers'], 'device': torch_device(), 'project': str(RUNS), 'name': e['name'],
            'exist_ok': True, 'seed': CFG['seed']}


def train_yolo(name, redo=False):
    from ultralytics import YOLO
    e, final = experiment(name), model_path(name)
    if final.exists() and not redo:
        print(f'{name}: fine-tuned model already saved at {rel(final)}')
        return final
    run_dir = RUNS / name
    if redo and run_dir.exists():
        shutil.rmtree(run_dir)
    ultralytics_setup()
    seed_everything()
    args = yolo_args(e)
    save_run_config(name, args)
    last, best = run_dir / 'weights' / 'last.pt', run_dir / 'weights' / 'best.pt'
    state = yolo_ckpt_state(last)
    if state == 'resumable':
        log(f'{name}: resuming from {rel(last)}')
        patch_yolo_ckpt(last, {k: args[k] for k in ('data', 'project', 'name', 'workers')})
        YOLO(str(last)).train(resume=True, batch=args['batch'], device=args['device'], workers=args['workers'])
    elif state == 'missing':
        YOLO(str(yolo_weights(e['weights']))).train(**args)
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best if best.exists() else last, final)
    write_json(MODELS / f'{name}.json', {'experiment': name, 'family': 'yolo', 'imgsz': e['imgsz'], 'saved': now()})
    log(f'{name}: saved fine-tuned model to {rel(final)}')
    return final


def time_yolo_epoch(name, fraction=0.05):
    from ultralytics import YOLO
    e = experiment(name)
    args = {**yolo_args(e), 'epochs': 1, 'fraction': fraction, 'project': str(RUNS / '_timing'),
            'plots': False, 'val': False}
    t0 = time.time()
    YOLO(str(yolo_weights(e['weights']))).train(**args)
    sec = (time.time() - t0) / fraction
    shutil.rmtree(RUNS / '_timing', ignore_errors=True)
    return sec, e.get('epochs', CFG['yolo']['epochs'])


class YoloRunner:
    family = 'yolo'

    def __init__(self, name):
        from ultralytics import YOLO
        self.name, self.imgsz = name, experiment(name)['imgsz']
        self.model = YOLO(str(model_path(name)))
        self.device = torch_device()
        self.max_det = CFG['evaluation']['max_det']

    def predict(self, images, conf):
        '''images: list of RGB arrays. Returns [(xyxy, scores, classes)] and summed times in ms.'''
        bgr = [np.ascontiguousarray(im[..., ::-1]) for im in images]
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=conf, max_det=self.max_det, device=self.device, verbose=False)
        out = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)) for r in res]
        times = {k: float(sum(r.speed[k] for r in res)) for k in ('preprocess', 'inference', 'postprocess')}
        return out, times

    def params(self):
        return int(sum(p.numel() for p in self.model.model.parameters()))

    def gflops(self):
        from ultralytics.utils.torch_utils import get_flops
        return float(get_flops(self.model.model, self.imgsz))


# D-FINE (KerasHub, TensorFlow)

_TF = {}


def setup_tf():
    import tensorflow as tf
    if 'ready' not in _TF:
        for g in tf.config.list_physical_devices('GPU'):
            with contextlib.suppress(RuntimeError):
                tf.config.experimental.set_memory_growth(g, True)
        _TF['ready'] = True
    return tf


def dfine_classes():
    '''Builds the D-FINE subclass and optimizer on first use so YOLO-only runs never import TensorFlow.'''
    if 'classes' in _TF:
        return _TF['classes']
    tf = setup_tf()
    import keras
    import keras_hub
    from keras_hub.src.models.d_fine import d_fine_loss as L
    ops = keras.ops

    class SkyDFine(keras_hub.models.DFineObjectDetector):
        '''D-FINE with a loss that handles a different number of boxes per image.

        The stock KerasHub loss counts padded boxes as real objects and looks up matched targets
        without the per-image offset, which breaks batches of drone images. Labels of -1 mark padding.
        '''

        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, **kwargs):
            if 'dn_num_group' in y_pred:
                raise ValueError('Denoising queries are not supported in this loss')
            labels = ops.cast(y['labels'], 'int32')
            valid = labels >= 0
            counts = ops.sum(ops.cast(valid, 'int32'), axis=1)
            offsets = ops.expand_dims(ops.cumsum(counts) - counts, 1)
            targets = [{'labels': tf.boolean_mask(labels, valid),
                        'boxes': tf.boolean_mask(ops.cast(y['boxes'], 'float32'), valid)}]
            n_boxes = ops.maximum(ops.cast(ops.sum(counts), 'float32'), 1.0)
            cost = (self.use_focal_loss, self.matcher_alpha, self.matcher_gamma, self.matcher_bbox_cost,
                    self.matcher_class_cost, self.matcher_ciou_cost, self.backbone)
            w, total = self.weight_dict, []

            def match(out):
                rows, cols, ok = L.hungarian_matcher(out, targets, counts, *cost)
                return rows, cols + offsets, ok

            def add(out, ddf=None):
                idx = match(out)
                parts = {**L.compute_vfl_loss(out, targets, idx, n_boxes, self.num_classes, self.matcher_alpha, self.matcher_gamma),
                         **L.compute_box_losses(out, targets, idx, n_boxes)}
                if ddf is not None:
                    parts.update(L.compute_local_losses(out, targets, idx, n_boxes, self.backbone, self.ddf_temperature, compute_ddf=ddf))
                total.extend(v * w[k] for k, v in parts.items() if k in w)

            logits, refs = y_pred['intermediate_logits'], y_pred['intermediate_reference_points']
            corners, init = y_pred['intermediate_predicted_corners'], y_pred['initial_reference_points']
            add({'logits': logits[:, -1], 'pred_boxes': ops.clip(refs[:, -1], 0, 1), 'pred_corners': corners[:, -1],
                 'ref_points': init[:, -1], 'teacher_corners': ops.zeros_like(corners[:, -1]),
                 'teacher_logits': ops.zeros_like(logits[:, -1])}, ddf=False)
            n_layers = self.backbone.num_decoder_layers
            for i in range(n_layers):
                add({'logits': logits[:, i], 'pred_boxes': ops.clip(refs[:, i], 0, 1), 'pred_corners': corners[:, i],
                     'ref_points': init[:, i], 'teacher_corners': corners[:, -1], 'teacher_logits': logits[:, -1]},
                    ddf=i < n_layers - 1)
            add({'logits': y_pred['enc_topk_logits'], 'pred_boxes': ops.clip(y_pred['enc_topk_bboxes'], 0, 1)})
            return ops.sum(ops.stack(total))

    class LayerwiseAdamW(keras.optimizers.AdamW):
        '''AdamW where each variable can have its own learning-rate multiplier (used for LLRD).'''

        def __init__(self, lr_scales=None, **kwargs):
            super().__init__(**kwargs)
            self._scale_by_id = lr_scales or {}
            self._scales = []

        def build(self, variables):
            super().build(variables)
            self._scales = [self._scale_by_id.get(id(v), 1.0) for v in variables]

        def update_step(self, gradient, variable, learning_rate):
            idx = self._get_variable_index(variable)
            scale = self._scales[idx] if idx < len(self._scales) else 1.0
            return super().update_step(gradient, variable, learning_rate * scale if scale != 1.0 else learning_rate)

    _TF['classes'] = (SkyDFine, LayerwiseAdamW)
    return _TF['classes']


def dfine_converter():
    '''Pixel scale and offset the pretrained D-FINE expects.'''
    if 'conv' not in _TF:
        import keras_hub
        c = keras_hub.layers.DFineImageConverter.from_preset(CFG['dfine']['preset'])
        scale = 1 / 255 if c.scale is None else c.scale
        offset = 0.0 if c.offset is None else c.offset
        _TF['conv'] = (np.array(scale, 'float32').reshape(-1), np.array(offset, 'float32').reshape(-1))
    return _TF['conv']


def build_dfine(imgsz, pretrained=True):
    '''D-FINE for 10 classes at imgsz. Pretrained weights are copied over by shape.

    from_preset(num_classes=10) rebuilds the backbone from its config and silently drops the
    pretrained weights, so the copy is done here instead. Only the class heads stay random.
    '''
    SkyDFine, _ = dfine_classes()
    import keras_hub
    src = keras_hub.models.DFineBackbone.from_preset(CFG['dfine']['preset'], load_weights=pretrained)
    cfg = src.get_config()
    cfg.update(num_labels=NC, image_shape=(imgsz, imgsz, 3), anchor_image_size=(imgsz, imgsz))
    inner = cfg['backbone'].get('config', {})
    if inner.get('image_shape') and inner['image_shape'][0] is not None:
        inner['image_shape'] = [imgsz, imgsz, 3]
    bb = keras_hub.models.DFineBackbone.from_config(cfg)
    if pretrained:
        if len(src.weights) != len(bb.weights):
            raise RuntimeError('D-FINE weight lists differ, cannot copy pretrained weights')
        copied = 0
        for a, b in zip(src.weights, bb.weights):
            if tuple(a.shape) == tuple(b.shape):
                b.assign(a.numpy())
                copied += 1
        print(f'D-FINE: copied {copied}/{len(bb.weights)} pretrained tensors (the rest are the new class heads)')
    del src
    return SkyDFine(backbone=bb, num_classes=NC, preprocessor=None)


def dfine_arrays(split, imgsz, fraction=1.0, limit=None):
    '''Image paths plus padded boxes (normalized cx, cy, w, h) and labels (-1 = padding).'''
    hp = CFG['dfine']
    coco, sdir = load_coco(split), split_dir(split)
    rng = np.random.default_rng(CFG['seed'])
    by_img = defaultdict(list)
    for a in coco['annotations']:
        by_img[a['image_id']].append(a)
    ims = coco['images']
    if fraction < 1.0:
        keep = set(rng.choice(len(ims), int(round(len(ims) * fraction)), replace=False).tolist())
        ims = [im for i, im in enumerate(ims) if i in keep]
    if limit:
        ims = ims[:limit]
    m, paths, boxes, labels = hp['max_boxes'], [], [], []
    for im in ims:
        w, h = im['width'], im['height']
        bx, lb = [], []
        for a in by_img[im['id']]:
            x, y, bw, bh = a['bbox']
            if bw / w * imgsz < hp['min_box_px'] or bh / h * imgsz < hp['min_box_px']:
                continue
            bx.append([(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
            lb.append(a['category_id'] - 1)
        if not bx:
            continue
        bx, lb = np.array(bx, 'float32'), np.array(lb, 'int32')
        if len(lb) > m:
            pick = rng.choice(len(lb), m, replace=False)
            bx, lb = bx[pick], lb[pick]
        pb, pl = np.zeros((m, 4), 'float32'), -np.ones(m, 'int32')
        pb[:len(lb)], pl[:len(lb)] = bx, lb
        paths.append(str(sdir / 'images' / im['file_name']))
        boxes.append(pb)
        labels.append(pl)
    return paths, np.stack(boxes), np.stack(labels)


def dfine_dataset(split, imgsz, batch, training, fraction=1.0, limit=None):
    tf = setup_tf()
    paths, boxes, labels = dfine_arrays(split, imgsz, fraction, limit)
    scale, offset = dfine_converter()
    hflip = CFG['dfine']['hflip']
    ds = tf.data.Dataset.from_tensor_slices((paths, boxes, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=CFG['seed'], reshuffle_each_iteration=True)

    def load(path, b, lab):
        img = tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), (imgsz, imgsz))
        if training:
            flip = tf.random.uniform([]) < hflip
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            b = tf.cond(flip, lambda: tf.stack([1.0 - b[:, 0], b[:, 1], b[:, 2], b[:, 3]], axis=-1), lambda: b)
        return img * scale + offset, {'boxes': b, 'labels': lab}

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths) // batch


def make_dfine_optimizer(det, setup, hp, steps, epochs):
    import keras
    _, LayerwiseAdamW = dfine_classes()
    total = max(steps * epochs, 2)
    warm = min(int(steps * hp['warmup_epochs']), total - 1)
    lr = keras.optimizers.schedules.CosineDecay(initial_learning_rate=hp['lr'] * 0.01, decay_steps=total - warm,
                                                alpha=hp['min_lr_ratio'], warmup_target=hp['lr'], warmup_steps=warm)
    hgnet = det.backbone.backbone
    scales = {}
    if setup == 'frozen':
        hgnet.trainable = False
    elif setup == 'llrd':
        vs, k = hgnet.trainable_variables, hp['llrd_groups']
        for j, v in enumerate(vs):
            scales[id(v)] = hp['llrd_decay'] ** (k - min(j * k // len(vs), k - 1))
    elif setup != 'full':
        raise ValueError(f'Unknown D-FINE setup {setup}')
    return LayerwiseAdamW(lr_scales=scales, learning_rate=lr, weight_decay=hp['weight_decay'],
                          global_clipnorm=hp['clipnorm'], use_ema=hp['ema'], ema_momentum=hp['ema_momentum'])


def _dfine_prepare(name):
    '''Shared setup for training and timing: data, model, optimizer.'''
    import keras
    e = experiment(name)
    hp = {**CFG['dfine'], **e.get('overrides', {})}
    setup, source = resolve_setup(e)
    imgsz, epochs = e['imgsz'], e.get('epochs', hp['epochs'])
    batch = batch_for('dfine', imgsz)
    prepare_dataset()
    seed_everything()
    train_ds, steps = dfine_dataset('train', imgsz, batch, training=True, fraction=e.get('fraction', 1.0))
    val_ds, _ = dfine_dataset('val', imgsz, batch, training=False, limit=hp['val_images'])
    det = build_dfine(imgsz)
    init = e.get('init_from')
    if init:
        src = source if init == 'best' else init
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
            det.load_weights(str(model_path(src)))
        print(f'{name}: starting from fine-tuned {src}')
    opt = make_dfine_optimizer(det, setup, hp, steps, epochs)
    keras.Model.compile(det, optimizer=opt, jit_compile=hp['jit_compile'])
    params = {'setup': setup, 'setup_from': source, 'init_from': init, 'imgsz': imgsz, 'epochs': epochs,
              'batch': batch, 'steps_per_epoch': steps, 'fraction': e.get('fraction', 1.0), **hp}
    return det, train_ds, val_ds, params


def train_dfine(name, redo=False):
    final = model_path(name)
    if final.exists() and not redo:
        print(f'{name}: fine-tuned model already saved at {rel(final)}')
        return final
    import keras
    run_dir = RUNS / name
    if redo and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    det, train_ds, val_ds, params = _dfine_prepare(name)
    save_run_config(name, params)
    epochs = params['epochs']

    class EpochLog(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            mark(f'train_{name}', 'running', epoch=epoch + 1, epochs=epochs,
                 loss=float(logs.get('loss', 0)), val_loss=float(logs.get('val_loss', 0)))

    callbacks = [
        keras.callbacks.BackupAndRestore(str(run_dir / 'backup'), delete_checkpoint=False),
        keras.callbacks.ModelCheckpoint(str(run_dir / 'last.weights.h5'), save_weights_only=True),
        keras.callbacks.ModelCheckpoint(str(run_dir / 'best.weights.h5'), save_weights_only=True,
                                        save_best_only=True, monitor='val_loss'),
        keras.callbacks.CSVLogger(str(run_dir / 'history.csv'), append=True),
        EpochLog(),
    ]
    det.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=fit_verbose())
    MODELS.mkdir(parents=True, exist_ok=True)
    det.save_weights(str(final))
    write_json(MODELS / f'{name}.json', {'experiment': name, 'family': 'dfine', 'imgsz': params['imgsz'],
                                          'setup': params['setup'], 'setup_from': params['setup_from'],
                                          'preset': CFG['dfine']['preset'], 'saved': now()})
    log(f'{name}: saved fine-tuned model to {rel(final)}')
    return final


def time_dfine_epoch(name, steps=30):
    import keras
    det, train_ds, _, params = _dfine_prepare(name)
    stamps = []

    class Clock(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            stamps.append(time.perf_counter())

    det.fit(train_ds.take(steps), epochs=1, callbacks=[Clock()], verbose=0)
    per_step = float(np.median(np.diff(stamps[min(5, len(stamps) - 2):])))
    return per_step * params['steps_per_epoch'], params['epochs']


def decode_topk(logits, boxes, hw, k, conf):
    '''D-FINE post-processing: top-k over all query/class scores, no NMS.'''
    h, w = hw
    nq, nc = logits.shape
    prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    k = min(k, prob.size)
    idx = np.argpartition(-prob, k - 1)[:k]
    idx = idx[np.argsort(-prob[idx])]
    idx = idx[prob[idx] >= conf]
    cx, cy, bw, bh = boxes[idx // nc].T
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    return np.clip(xyxy, 0, [w, h, w, h]).astype('float32'), prob[idx].astype('float32'), (idx % nc).astype(int)


class DFineRunner:
    family = 'dfine'

    def __init__(self, name):
        tf = setup_tf()
        meta = read_json(MODELS / f'{name}.json', {})
        self.name, self.imgsz = name, meta.get('imgsz', experiment(name)['imgsz'])
        self.det = build_dfine(self.imgsz, pretrained=False)
        with warnings.catch_warnings():
            # The saved file also holds optimizer state, which inference does not need
            warnings.filterwarnings('ignore', message='Skipping variable loading for optimizer')
            self.det.load_weights(str(model_path(name)))
        self.scale, self.offset = dfine_converter()
        self.max_det = CFG['evaluation']['max_det']
        spec = tf.TensorSpec([None, self.imgsz, self.imgsz, 3], tf.float32)
        self.forward = tf.function(self._forward, input_signature=[spec])

    def _forward(self, x):
        out = self.det(x, training=False)
        return out['logits'], out['pred_boxes']

    def predict(self, images, conf):
        import cv2
        s = self.imgsz
        t0 = time.perf_counter()
        x = np.stack([cv2.resize(im, (s, s), interpolation=cv2.INTER_LINEAR) for im in images]).astype('float32')
        x = x * self.scale + self.offset
        t1 = time.perf_counter()
        logits, boxes = self.forward(x)
        logits, boxes = logits.numpy(), boxes.numpy()
        t2 = time.perf_counter()
        out = [decode_topk(logits[i], boxes[i], im.shape[:2], self.max_det, conf) for i, im in enumerate(images)]
        t3 = time.perf_counter()
        return out, {'preprocess': (t1 - t0) * 1e3, 'inference': (t2 - t1) * 1e3, 'postprocess': (t3 - t2) * 1e3}

    def params(self):
        return int(self.det.count_params())

    def gflops(self):
        '''FLOPs from the TensorFlow profiler. Returns None if the profiler cannot handle the graph.'''
        try:
            tf = setup_tf()
            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
            fn = tf.function(lambda x: self.det(x, training=False))
            cf = fn.get_concrete_function(tf.TensorSpec([1, self.imgsz, self.imgsz, 3], tf.float32))
            frozen, _ = convert_variables_to_constants_v2_as_graph(cf)
            builder = tf.compat.v1.profiler.ProfileOptionBuilder
            opts = builder(builder.float_operation()).with_empty_output().build()
            info = tf.compat.v1.profiler.profile(graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(), cmd='op', options=opts)
            return info.total_float_ops / 1e9
        except Exception as e:
            print(f'FLOPs not available for {self.name}: {type(e).__name__}')
            return None


def train(name, redo=False):
    fam = experiment(name)['family']
    return train_yolo(name, redo) if fam == 'yolo' else train_dfine(name, redo)


def time_epoch(name):
    '''Estimate one epoch before a full run. Saved to results/timing.json.'''
    fam = experiment(name)['family']
    sec, epochs = time_yolo_epoch(name) if fam == 'yolo' else time_dfine_epoch(name)
    epochs = epochs or CFG[fam]['epochs']
    timing = read_json(RESULTS / 'timing.json', {})
    timing[f'{name}@{PROFILE}'] = {'epoch_min': round(sec / 60, 1), 'epochs': epochs,
                                   'total_hours': round(sec * epochs / 3600, 1), 'gpu': gpu_name(), 'measured': now()}
    write_json(RESULTS / 'timing.json', timing)
    log(f'{name}: about {sec / 60:.1f} min per epoch, {sec * epochs / 3600:.1f} h for {epochs} epochs')
    return timing[f'{name}@{PROFILE}']


# Evaluation (same code for every model)

_RUNNERS = {}


def load_runner(name):
    if name not in _RUNNERS:
        if not model_path(name).exists():
            raise FileNotFoundError(f'No fine-tuned model for {name}. Recommended: train it first')
        cls = YoloRunner if experiment(name)['family'] == 'yolo' else DFineRunner
        _RUNNERS[name] = cls(name)
    return _RUNNERS[name]


def box_iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes, scores, classes, iou):
    '''Class-wise NMS, returns kept indices sorted by score.'''
    if not len(boxes):
        return np.zeros(0, int)
    try:
        import torch
        import torchvision
        keep = torchvision.ops.batched_nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(),
                                           torch.from_numpy(classes).long(), iou)
        return keep.numpy()
    except ImportError:
        pass
    shifted = boxes + classes[:, None] * (boxes.max() + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[box_iou(shifted[i:i + 1], shifted[rest])[0] <= iou]
    return np.array(keep, int)


def tile_windows(h, w, tile, overlap):
    step = max(int(tile * (1 - overlap)), 1)

    def starts(n):
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        return s if s[-1] == n - tile else s + [n - tile]

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]


def sahi_predict(runner, img, conf):
    '''SAHI-style tiled inference: detect on overlapping tiles (plus the full image), then merge with NMS.'''
    sc = CFG['sahi']
    h, w = img.shape[:2]
    wins = tile_windows(h, w, sc['tile'], sc['overlap'])
    crops = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]
    offs = [(x1, y1) for x1, y1, _, _ in wins]
    if sc['include_full_image']:
        crops.append(img)
        offs.append((0, 0))
    bs, ss, cs, times = [], [], [], defaultdict(float)
    for i in range(0, len(crops), sc['batch']):
        out, t = runner.predict(crops[i:i + sc['batch']], conf)
        for k, v in t.items():
            times[k] += v
        for (b, s, c), (ox, oy) in zip(out, offs[i:i + sc['batch']]):
            bs.append(b + np.array([ox, oy, ox, oy], 'float32'))
            ss.append(s)
            cs.append(c)
    b, s, c = np.concatenate(bs), np.concatenate(ss), np.concatenate(cs)
    with contextlib.suppress(ImportError):
        import torchvision  # noqa: F401  (import before timing so it is not counted as latency)
    t0 = time.perf_counter()
    keep = nms(b, s, c, sc['merge_iou'])[:CFG['evaluation']['max_det']]
    times['postprocess'] += (time.perf_counter() - t0) * 1e3
    return (b[keep], s[keep], c[keep]), dict(times)


def coco_metrics(coco, dets):
    '''COCO mAP, AP by object size and per-class AP via pycocotools.'''
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids = set(dets)
    gt_dict = {'images': [im for im in coco['images'] if im['id'] in ids],
               'annotations': [a for a in coco['annotations'] if a['image_id'] in ids],
               'categories': coco['categories']}
    rows = [np.column_stack([np.full(len(s), iid), b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1], s, c + 1])
            for iid, (b, s, c) in dets.items() if len(s)]
    zero = {k: 0.0 for k in ('map50_95', 'map50', 'map75', 'ap_small', 'ap_medium', 'ap_large')}
    if not rows:
        return zero, [{'class': n, 'ap50': 0.0, 'ap50_95': 0.0, 'ap_small': 0.0} for n in NAMES]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(np.concatenate(rows))
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.maxDets = [1, 10, CFG['evaluation']['max_det']]
        ev.evaluate()
        ev.accumulate()
    p = ev.eval['precision']

    def ap(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else 0.0

    m = {'map50_95': ap(p[:, :, :, 0, -1]), 'map50': ap(p[0, :, :, 0, -1]), 'map75': ap(p[5, :, :, 0, -1]),
         'ap_small': ap(p[:, :, :, 1, -1]), 'ap_medium': ap(p[:, :, :, 2, -1]), 'ap_large': ap(p[:, :, :, 3, -1])}
    per_class = [{'class': n, 'ap50': ap(p[0, :, k, 0, -1]), 'ap50_95': ap(p[:, :, k, 0, -1]),
                  'ap_small': ap(p[:, :, k, 1, -1])} for k, n in enumerate(NAMES)]
    return m, per_class


def precision_recall_f1(gts, dets):
    '''Precision and recall at IoU 0.5, taken at the confidence with the best mean F1 (as Ultralytics reports).'''
    thr = CFG['evaluation']['pr_iou']
    confs, tps, pcls, n_gt = [], [], [], np.zeros(NC)
    for iid, (gb, gc) in gts.items():
        n_gt += np.bincount(gc, minlength=NC)
        b, s, c = dets.get(iid, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        order = np.argsort(-s)
        b, s, c = b[order], s[order], c[order]
        tp = np.zeros(len(s), bool)
        if len(s) and len(gb):
            iou = box_iou(b, gb)
            iou[c[:, None] != gc[None, :]] = 0
            used = np.zeros(len(gb), bool)
            for i in np.where(iou.max(1) >= thr)[0]:
                row = np.where(used, 0, iou[i])
                j = row.argmax()
                if row[j] >= thr:
                    used[j], tp[i] = True, True
        confs.append(s)
        tps.append(tp)
        pcls.append(c)
    conf, tp, pc = np.concatenate(confs), np.concatenate(tps), np.concatenate(pcls)
    x = np.linspace(0, 1, 1000)
    p_curve, r_curve = np.zeros((NC, 1000)), np.zeros((NC, 1000))
    for k in range(NC):
        sel = pc == k
        if not sel.any() or n_gt[k] == 0:
            continue
        order = np.argsort(-conf[sel])
        ctp = np.cumsum(tp[sel][order])
        cfp = np.cumsum(~tp[sel][order])
        recall, precision = ctp / n_gt[k], ctp / (ctp + cfp)
        p_curve[k] = np.interp(-x, -conf[sel][order], precision, left=1)
        r_curve[k] = np.interp(-x, -conf[sel][order], recall, left=0)
    valid = n_gt > 0
    f1 = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    i = int(f1[valid].mean(0).argmax())
    p, r = float(p_curve[valid, i].mean()), float(r_curve[valid, i].mean())
    return {'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r + 1e-16), 'best_conf': float(x[i])}


def confusion_matrix(gts, dets):
    '''Rows = predicted, columns = true, last index = background (same layout as Ultralytics).'''
    ev = CFG['evaluation']
    m = np.zeros((NC + 1, NC + 1))
    for iid, (gb, gc) in gts.items():
        b, s, c = dets.get(iid, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        keep = s >= ev['cm_conf']
        b, c = b[keep], c[keep]
        if not len(c):
            np.add.at(m, (NC, gc), 1)
            continue
        if not len(gc):
            np.add.at(m, (c, NC), 1)
            continue
        iou = box_iou(gb, b)
        gi, pi = np.where(iou > ev['cm_iou'])
        if len(gi):
            order = np.argsort(-iou[gi, pi])
            gi, pi = gi[order], pi[order]
            _, first = np.unique(pi, return_index=True)
            gi, pi = gi[np.sort(first)], pi[np.sort(first)]
            _, first = np.unique(gi, return_index=True)
            gi, pi = gi[first], pi[first]
        np.add.at(m, (c[pi], gc[gi]), 1)
        np.add.at(m, (NC, np.delete(gc, gi)), 1)
        np.add.at(m, (np.delete(c, pi), NC), 1)
    return m


def sample_ids(split):
    '''Fixed test images with many objects, reused in every sample figure.'''
    coco = load_coco(split)
    counts = defaultdict(int)
    for a in coco['annotations']:
        counts[a['image_id']] += 1
    busy = sorted(i for i, n in counts.items() if n >= 50) or sorted(counts)
    rng = np.random.default_rng(CFG['seed'])
    k = min(CFG['evaluation']['sample_images'], len(busy))
    return sorted(rng.choice(busy, k, replace=False).tolist())


def evaluate(name, split=None, sahi=False, redo=False):
    '''Run one fine-tuned model on a split and save metrics, tables and plots in results/<name>/<split>.'''
    ev = CFG['evaluation']
    split = split or ev['report_split']
    out_dir = eval_dir(name, split, sahi)
    if (out_dir / 'metrics.json').exists() and not redo:
        return read_json(out_dir / 'metrics.json')
    coco = load_coco(split)
    img_dir = split_dir(split) / 'images'
    runner = load_runner(name)
    images = coco['images'][:ev['limit']] if ev.get('limit') else coco['images']
    dets, times = {}, []
    print(f'Evaluating {name} on {split}{" with SAHI" if sahi else ""} ({len(images)} images)')
    for i, im in enumerate(images):
        img = read_rgb(img_dir / im['file_name'])
        t0 = time.perf_counter()
        if sahi:
            dets[im['id']], t = sahi_predict(runner, img, ev['conf'])
        else:
            out, t = runner.predict([img], ev['conf'])
            dets[im['id']] = out[0]
        if i >= ev['warmup']:
            times.append({**t, 'total': (time.perf_counter() - t0) * 1e3})
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{len(images)}')
    gts = {iid: g for iid, g in gt_by_image(coco).items() if iid in dets}
    m, per_class = coco_metrics(coco, dets)
    m.update(precision_recall_f1(gts, dets))
    meta = read_json(MODELS / f'{name}.json', {})
    times = times or [{'total': 0.0}]
    m.update({
        'experiment': name, 'family': runner.family, 'split': split, 'sahi': sahi, 'imgsz': runner.imgsz,
        'setup': meta.get('setup', 'fine-tune'), 'n_images': len(images), 'params': runner.params(),
        'gflops': runner.gflops(), 'gpu': gpu_name(), 'profile': PROFILE, 'evaluated': now(),
        'latency_ms': {k: float(np.mean([t.get(k, 0.0) for t in times])) for k in ('preprocess', 'inference', 'postprocess', 'total')},
        'per_class': per_class, 'confusion': confusion_matrix(gts, dets),
    })
    m['samples'] = {str(i): [dets[i][0], dets[i][1], dets[i][2]] for i in sample_ids(split) if i in dets}
    write_json(out_dir / 'metrics.json', m)
    write_eval_report(m, out_dir)
    plot_confusion(np.array(m['confusion']), out_dir / 'confusion_matrix_normalized.png')
    log(f'{name} {split}{" sahi" if sahi else ""}: mAP50-95 {100 * m["map50_95"]:.2f}, mAP50 {100 * m["map50"]:.2f}, APs {100 * m["ap_small"]:.2f}')
    return m


def write_eval_report(m, out_dir):
    lat = m['latency_ms']
    flops = f'{m["gflops"]:.1f}B (at {m["imgsz"]} px)' if m.get('gflops') else 'n/a'
    post = 'NMS' if m['experiment'].startswith('yolo11') else 'post-process'
    lines = [f'### {m["experiment"]}', '',
             f'Evaluated on the VisDrone-DET **{m["split"]}** split ({m["n_images"]} images) with the shared SkyScout '
             f'evaluation code{" and SAHI tiled inference" if m["sahi"] else ""}.', '',
             '| Metric | Score (%) |', '|---|---|',
             f'| mAP@50 | {100 * m["map50"]:.2f} |', f'| mAP@50-95 | {100 * m["map50_95"]:.2f} |',
             f'| AP small (<32 px) | {100 * m["ap_small"]:.2f} |', f'| Precision | {100 * m["precision"]:.2f} |',
             f'| Recall | {100 * m["recall"]:.2f} |', f'| F1 Score | {100 * m["f1"]:.2f} |',
             f'| Parameters | {m["params"] / 1e6:.1f}M |', f'| FLOPs | {flops} |',
             f'| Latency | {lat["total"]:.1f} ms/img (inference {lat["inference"]:.1f}, {post} {lat["postprocess"]:.1f}) |',
             '', '#### Per-class performance', '', '| Class | mAP@50 | mAP@50-95 | AP small |', '|---|---|---|---|']
    lines += [f'| {r["class"]} | {100 * r["ap50"]:.2f} | {100 * r["ap50_95"]:.2f} | {100 * r["ap_small"]:.2f} |' for r in m['per_class']]
    lines += ['', '![Confusion matrix](confusion_matrix_normalized.png)', '']
    write_text(out_dir / 'report.md', '\n'.join(lines))
    with open(out_dir / 'per_class.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['class', 'ap50', 'ap50_95', 'ap_small'])
        wr.writeheader()
        wr.writerows(m['per_class'])


# Plots and the final report

def _plt():
    import matplotlib
    if 'ipykernel' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                         'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'legend.frameon': False})
    return plt


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_confusion(m, path):
    plt = _plt()
    norm = m / (m.sum(0, keepdims=True) + 1e-9)
    labels = NAMES + ['background']
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.where(norm < 0.005, np.nan, norm), cmap='Blues', vmin=0, vmax=max(norm.max(), 1e-3))
    for (r, c), v in np.ndenumerate(norm):
        if v >= 0.005:
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v > 0.5 else INK)
    ax.set_xticks(range(NC + 1), labels, rotation=90)
    ax.set_yticks(range(NC + 1), labels)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix Normalized')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


def draw_dets(img, boxes, scores, classes, conf=0.0):
    import cv2
    out = img.copy()
    t = max(1, round(max(img.shape[:2]) / 900))
    for b, s, c in zip(boxes, scores, classes):
        if s < conf:
            continue
        rgb = tuple(int(BOX_COLORS[c][i:i + 2], 16) for i in (1, 3, 5))
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), rgb, t)
    return out


def _bar(ax, labels, values, color, fmt='{:.1f}'):
    bars = ax.bar(labels, values, color=color, width=0.6, edgecolor=SURFACE, linewidth=2)
    for r, v in zip(bars, values):
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha='center', va='bottom', fontsize=8, color=INK2)
    return bars


def make_report():
    '''Comparison table and plots from every saved metrics.json. Safe to run at any time.'''
    plt = _plt()
    split = CFG['evaluation']['report_split']
    rows = []
    for name in CFG['experiments']:
        for sahi in (False, True):
            m = metrics_for(name, split, sahi)
            if m:
                rows.append(m)
    if not rows:
        print('No evaluated models yet. Recommended: run an experiment first')
        return None
    head = ['Model', 'Setup', 'Input', 'mAP@50', 'mAP@50-95', 'APs', 'Precision', 'Recall', 'F1',
            'Params (M)', 'GFLOPs', 'Latency (ms)', 'NMS/post (ms)']
    table = []
    for m in sorted(rows, key=lambda r: -r['map50_95']):
        table.append([m['experiment'] + (' + SAHI' if m['sahi'] else ''), m['setup'], m['imgsz'],
                      f'{100 * m["map50"]:.2f}', f'{100 * m["map50_95"]:.2f}', f'{100 * m["ap_small"]:.2f}',
                      f'{100 * m["precision"]:.2f}', f'{100 * m["recall"]:.2f}', f'{100 * m["f1"]:.2f}',
                      f'{m["params"] / 1e6:.1f}', f'{m["gflops"]:.1f}' if m.get('gflops') else 'n/a',
                      f'{m["latency_ms"]["total"]:.1f}', f'{m["latency_ms"]["postprocess"]:.1f}'])
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in table]
    with open(RESULTS / 'comparison.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([head] + table)
    gpus = sorted({m['gpu'] for m in rows})
    figs = [p for p in (plot_map_vs_latency(rows), plot_setups(), plot_input_size(), plot_sahi(),
                        plot_data_efficiency(), plot_samples(split)) if p]
    text = (f'### SkyScout results\n\nAll models scored on the VisDrone-DET **{split}** split with the same evaluation code. '
            f'Latency measured with batch 1 on: {", ".join(gpus)}.\n\n#### Comparison\n\n' + '\n'.join(md) + '\n\n'
            + ''.join(f'![{p.stem}](plots/{p.name})\n\n' for p in figs)
            + '#### Per-model details\n\n' + ''.join(f'- [{m["experiment"]}{" + SAHI" if m["sahi"] else ""}]'
                                                   f'({rel(eval_dir(m["experiment"], split, m["sahi"]) / "report.md")[len("results/"):]})\n' for m in rows))
    write_text(RESULTS / 'comparison.md', text)
    write_status()
    print(f'Report written to {rel(RESULTS / "comparison.md")}')
    return text


def _family_color(m):
    order = ['yolo11', 'yolo26', 'dfine']
    return PALETTE[next(i for i, k in enumerate(order) if m['experiment'].startswith(k))]


def plot_map_vs_latency(rows):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seen = set()
    # Data-efficiency runs have their own plot
    for m in [r for r in rows if 'fraction' not in CFG['experiments'][r['experiment']]]:
        fam = m['experiment'].split('_')[0]
        ax.scatter(m['latency_ms']['total'], 100 * m['map50_95'], s=64, color=_family_color(m), edgecolor=SURFACE,
                   linewidth=2, marker='s' if m['sahi'] else 'o', label=None if fam in seen else fam, zorder=3)
        seen.add(fam)
        ax.annotate(m['experiment'] + (' +SAHI' if m['sahi'] else ''), (m['latency_ms']['total'], 100 * m['map50_95']),
                    xytext=(5, 4), textcoords='offset points', fontsize=7, color=INK2)
    ax.set_xlabel('Latency per image (ms, lower is better)')
    ax.set_ylabel('mAP@50-95 (%)')
    ax.set_title('Accuracy vs speed (squares = SAHI)', loc='left')
    ax.legend(loc='best')
    return _save(fig, PLOTS / 'map_vs_latency.png')


def plot_setups():
    names = [n for n in CFG['dfine_setup_compare'] if metrics_for(n, 'val')]
    if not names:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, split, color in zip(axes, ('val', 'test'), PALETTE[:2]):
        vals = [100 * (metrics_for(n, split) or {}).get('map50_95', 0) for n in names]
        _bar(ax, [experiment(n)['setup'] for n in names], vals, color)
        ax.set_title(f'{split} mAP@50-95 (%)', loc='left')
    fig.suptitle('D-FINE fine-tuning setups', x=0.02, ha='left')
    return _save(fig, PLOTS / 'dfine_setups.png')


def plot_input_size():
    split = CFG['evaluation']['report_split']
    pairs = [(resolve_name(a), resolve_name(b)) for a, b in CFG['report']['input_size_pairs']]
    pairs = [(a, b) for a, b in pairs if a and metrics_for(a, split) and metrics_for(b, split)]
    if not pairs:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pairs))
    for j, (label, color) in enumerate([('640', PALETTE[0]), ('1024', PALETTE[1])]):
        vals = [100 * metrics_for(p[j], split)['ap_small'] for p in pairs]
        bars = ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, color=color, edgecolor=SURFACE, linewidth=2, label=f'{label} px')
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(x, [a.rsplit('_', 1)[0] for a, _ in pairs])
    ax.set_ylabel('AP small (%)')
    ax.set_title('Input size 640 vs 1024', loc='left')
    ax.legend()
    return _save(fig, PLOTS / 'input_size.png')


def plot_sahi():
    split = CFG['evaluation']['report_split']
    names = [n for n in map(resolve_name, CFG['sahi']['experiments']) if n and metrics_for(n, split, True)]
    if not names:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ('map50_95', 'ap_small'), ('mAP@50-95 (%)', 'AP small (%)')):
        x = np.arange(len(names))
        for j, (sahi, label) in enumerate([(False, 'full image'), (True, 'SAHI tiles')]):
            vals = [100 * metrics_for(n, split, sahi)[key] for n in names]
            ax.bar(x + (j - 0.5) * 0.36, vals, 0.36, color=PALETTE[j], edgecolor=SURFACE, linewidth=2, label=label)
        ax.set_xticks(x, names, rotation=15)
        ax.set_title(title, loc='left')
    axes[0].legend()
    return _save(fig, PLOTS / 'sahi.png')


def plot_data_efficiency():
    split = CFG['evaluation']['report_split']
    pts = sorted((int(k), resolve_name(v)) for k, v in CFG['report']['data_efficiency'].items())
    pts = [(k, metrics_for(n, split)) for k, n in pts if n]
    pts = [(k, m) for k, m in pts if m]
    if len(pts) < 2:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label, color in (('map50_95', 'mAP@50-95', PALETTE[0]), ('ap_small', 'AP small', PALETTE[1])):
        xs, ys = [k for k, _ in pts], [100 * m[key] for _, m in pts]
        ax.plot(xs, ys, color=color, linewidth=2, marker='o', markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(f'{ys[-1]:.1f}', (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8, color=INK2, va='center')
    ax.set_xticks([k for k, _ in pts], [f'{k}%' for k, _ in pts])
    ax.set_xlabel('Share of the training set')
    ax.set_ylabel('Score (%)')
    ax.set_title('D-FINE data efficiency', loc='left')
    ax.legend()
    return _save(fig, PLOTS / 'data_efficiency.png')


def plot_samples(split):
    names = [n for n in CFG['experiments'] if metrics_for(n, split)]
    dfine = 'dfine_best_long_640' if 'dfine_best_long_640' in names else best_dfine()
    names = [n for n in names if n in ('yolo11s_640', 'yolo26s_640', dfine)] or names[:3]
    ids = sample_ids(split)
    if not names or not ids:
        return None
    plt = _plt()
    coco = load_coco(split)
    files = {im['id']: im['file_name'] for im in coco['images']}
    gts = gt_by_image(coco)
    conf = CFG['demo']['conf']
    fig, axes = plt.subplots(len(ids), len(names) + 1, figsize=(4 * (len(names) + 1), 3 * len(ids)), squeeze=False)
    for r, iid in enumerate(ids):
        img = read_rgb(split_dir(split) / 'images' / files[iid])
        gb, gc = gts[iid]
        panels = [('ground truth', draw_dets(img, gb, np.ones(len(gc)), gc))]
        for n in names:
            d = metrics_for(n, split)['samples'].get(str(iid))
            if d:
                panels.append((n, draw_dets(img, np.array(d[0]).reshape(-1, 4), np.array(d[1]), np.array(d[2], int), conf)))
        for c, ax in enumerate(axes[r]):
            ax.axis('off')
            if c < len(panels):
                ax.imshow(panels[c][1])
                if r == 0:
                    ax.set_title(panels[c][0], fontsize=10)
    fig.suptitle(f'Sample predictions (confidence >= {conf})', x=0.02, ha='left')
    return _save(fig, PLOTS / 'sample_predictions.png')


# Pipeline

def run_experiment(name, redo=False, anywhere=False):
    '''Train (or resume), then evaluate on every split in config. Each part is recorded.'''
    e = experiment(name)
    if not allowed_here(e) and not anywhere:
        print(f'Recommended: run {name} on {", ".join(e["run_on"])} (this machine is {PROFILE}). Use anywhere=True to override')
        return None
    run_step(f'train_{name}', train, name, redo=redo)
    out = {}
    for split in CFG['evaluation']['splits']:
        run_step(f'eval_{name}_{split}', evaluate, name, split, redo=redo)
        out[split] = metrics_for(name, split)
    return out


def run_sahi(redo=False):
    split = CFG['evaluation']['report_split']
    for name in map(resolve_name, CFG['sahi']['experiments']):
        if name and model_path(name).exists():
            run_step(f'sahi_{name}', evaluate, name, split, sahi=True, redo=redo)


def run_all():
    '''Everything this machine is allowed to run, in config order. Re-run after any interruption.'''
    log(f'run_all started on {PROFILE} ({gpu_name()})')
    prepare_dataset()
    if not (RESULTS / 'dataset_summary.md').exists():
        dataset_summary()
    for name, e in CFG['experiments'].items():
        if e.get('setup') == 'best' and best_dfine() is None:
            print(f'Skip {name}: the D-FINE setups are not all evaluated yet')
            continue
        run_experiment(name)
    run_sahi()
    make_report()
    show_status()


def launch_demo(name=None, share=None):
    '''Gradio app: upload a drone image, get boxes and a count per class.'''
    import gradio as gr
    choices = [n for n in CFG['experiments'] if model_path(n).exists()]
    if not choices:
        raise FileNotFoundError('No fine-tuned models yet. Recommended: train at least one experiment first')
    split = CFG['evaluation']['report_split']
    scored = {n: (metrics_for(n, split) or {}).get('map50_95', -1) for n in choices}
    default = name or max(scored, key=scored.get)

    def infer(image, model_name, conf, tiles):
        if image is None:
            return None, []
        runner = load_runner(model_name)
        if tiles:
            (b, s, c), _ = sahi_predict(runner, image, conf)
        else:
            out, _ = runner.predict([image], conf)
            b, s, c = out[0]
        counts = np.bincount(c, minlength=NC)
        table = [[NAMES[k], int(counts[k])] for k in np.argsort(-counts) if counts[k]]
        return draw_dets(image, b, s, c), table + [['total', int(counts.sum())]]

    examples = []
    with contextlib.suppress(Exception):
        coco = load_coco(split)
        files = {im['id']: im['file_name'] for im in coco['images']}
        examples = [[str(split_dir(split) / 'images' / files[i]), default, CFG['demo']['conf'], False] for i in sample_ids(split)]
    legend = ', '.join(f'<span style="color:{BOX_COLORS[k]}">&#9632;</span> {n}' for k, n in enumerate(NAMES))
    app = gr.Interface(
        fn=infer,
        inputs=[gr.Image(type='numpy', label='Drone image'), gr.Dropdown(choices, value=default, label='Model'),
                gr.Slider(0.05, 0.95, value=CFG['demo']['conf'], step=0.05, label='Confidence threshold'),
                gr.Checkbox(label='Tiled inference (SAHI), slower but finds smaller objects')],
        outputs=[gr.Image(label='Detections'), gr.Dataframe(headers=['Class', 'Count'], label='Objects per class')],
        examples=examples or None, title='SkyScout: small-object detection in drone images',
        description=f'Upload a drone photo. Box colors: {legend}', flagging_mode='never')
    return app.launch(share=(PROFILE == 'colab') if share is None else share)


def main(argv=None):
    p = argparse.ArgumentParser(description='SkyScout pipeline (settings in config.yaml)')
    p.add_argument('command', choices=['status', 'prepare', 'summary', 'time', 'train', 'eval', 'run', 'sahi', 'all', 'report', 'demo'])
    p.add_argument('names', nargs='*', help='experiment names from config.yaml')
    p.add_argument('--split', help='val or test (default from config)')
    p.add_argument('--sahi', action='store_true', help='evaluate with tiled inference')
    p.add_argument('--redo', action='store_true', help='start again instead of resuming or skipping')
    p.add_argument('--anywhere', action='store_true', help='ignore run_on in config.yaml')
    a = p.parse_args(argv)
    print(f'SkyScout | profile: {PROFILE} | project: {ROOT}')
    if a.command == 'status':
        show_status()
    elif a.command == 'prepare':
        prepare_dataset()
    elif a.command == 'summary':
        dataset_summary()
    elif a.command == 'all':
        run_all()
    elif a.command == 'sahi':
        run_sahi(a.redo)
    elif a.command == 'report':
        make_report()
    elif a.command == 'demo':
        launch_demo(a.names[0] if a.names else None)
    else:
        if not a.names:
            p.error(f'{a.command} needs at least one experiment name')
        for n in a.names:
            if a.command == 'time':
                time_epoch(n)
            elif a.command == 'train':
                run_step(f'train_{n}', train, n, redo=a.redo)
            elif a.command == 'eval':
                split = a.split or CFG['evaluation']['report_split']
                step = f'{"sahi" if a.sahi else "eval"}_{n}' + ('' if a.sahi else f'_{split}')
                run_step(step, evaluate, n, split, sahi=a.sahi, redo=a.redo)
            elif a.command == 'run':
                run_experiment(n, redo=a.redo, anywhere=a.anywhere)


if __name__ == '__main__':
    main()
    # TensorFlow and PyTorch threads can block a normal exit, everything is saved by now
    sys.stdout.flush()
    os._exit(0)
