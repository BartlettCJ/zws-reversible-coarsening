from __future__ import annotations

import base64
import csv
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .atomic import atomic_write_text, sha256_file
from .history_erasure import state_histogram, state_sha256
from .history_media import load_exact_npz, verify_sidecar


RULE_SHA256 = "d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c"
STATE_NAMES = ("empty", "positive", "negative", "bound-neutral")
STATE_HEX = ("#05070a", "#f0e442", "#0072b2", "#cc79a7")
STATE_RGB = np.asarray(
    [tuple(int(value[index : index + 2], 16) for index in (1, 3, 5)) for value in STATE_HEX],
    dtype=np.uint8,
)
REQUIRED_PANEL_FIELDS = (
    "source_task",
    "seed",
    "density",
    "branch_point",
    "intervention",
    "phase",
    "state_hash",
    "rule_hash",
    "crop",
    "rendering_parameters",
)


def _write_sidecar(path: Path) -> None:
    atomic_write_text(
        path.with_suffix(path.suffix + ".sha256"),
        f"{sha256_file(path)}  {path.name}\n",
    )


def read_verified_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    verify_sidecar(target)
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_json_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{\"":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def validate_panel_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_PANEL_FIELDS if field not in record]
    if missing:
        raise RuntimeError(f"figure panel metadata lacks fields: {', '.join(missing)}")
    if record["rule_hash"] != RULE_SHA256:
        raise RuntimeError("figure panel metadata does not identify the frozen winner")
    crop = record["crop"]
    if not isinstance(crop, dict) or not {"x", "y", "width", "height"}.issubset(crop):
        raise RuntimeError("figure panel crop metadata is incomplete")
    rendering = record["rendering_parameters"]
    if not isinstance(rendering, dict) or rendering.get("interpolation") != "nearest":
        raise RuntimeError("automaton state panels must use nearest-neighbour sampling")


def make_panel_record(
    *,
    source_task: str,
    seed: int | str | None,
    density: float | str | None,
    branch_point: str | None,
    intervention: str,
    phase: int | str,
    state_hash: str,
    crop: tuple[int, int, int, int],
    source_path: str | Path | None,
    elapsed: int | None = None,
    requested_elapsed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x, y, width, height = (int(value) for value in crop)
    record: dict[str, Any] = {
        "source_task": source_task,
        "seed": seed,
        "density": density,
        "branch_point": branch_point,
        "intervention": intervention,
        "phase": phase,
        "elapsed": elapsed,
        "requested_elapsed": requested_elapsed,
        "state_hash": state_hash,
        "rule_hash": RULE_SHA256,
        "source_path": str(Path(source_path).resolve()) if source_path is not None else None,
        "crop": {"x": x, "y": y, "width": width, "height": height},
        "rendering_parameters": {
            "interpolation": "nearest",
            "palette": dict(zip(STATE_NAMES, STATE_HEX, strict=True)),
            "antialias_state_cells": False,
        },
    }
    if extra:
        record.update(extra)
    validate_panel_record(record)
    return record


def aligned_densest_crop(
    state: np.ndarray,
    size: int,
    *,
    bounds: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Choose the densest aligned square, tie-breaking by smallest y then x."""

    array = np.asarray(state, dtype=np.uint8)
    if array.ndim != 2 or np.any(array > 3):
        raise ValueError("state must be a two-dimensional four-state uint8 array")
    height, width = array.shape
    if bounds is None:
        left, top, bound_width, bound_height = 0, 0, width, height
    else:
        left, top, bound_width, bound_height = (int(value) for value in bounds)
    if size <= 0 or left < 0 or top < 0 or left + bound_width > width or top + bound_height > height:
        raise ValueError("invalid aligned crop bounds")
    if bound_width < size or bound_height < size:
        raise ValueError("crop size exceeds bounds")
    candidates: list[tuple[int, int, int]] = []
    for y in range(top, top + bound_height - size + 1, size):
        for x in range(left, left + bound_width - size + 1, size):
            occupied = int(np.count_nonzero(array[y : y + size, x : x + size]))
            candidates.append((-occupied, y, x))
    if not candidates:
        raise ValueError("bounds do not contain an aligned crop")
    _, y, x = min(candidates)
    return (x, y, size, size)


def crop_state(state: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = (int(value) for value in crop)
    array = np.asarray(state, dtype=np.uint8)
    if x < 0 or y < 0 or width <= 0 or height <= 0 or y + height > array.shape[0] or x + width > array.shape[1]:
        raise ValueError("crop lies outside state")
    return np.ascontiguousarray(array[y : y + height, x : x + width])


def state_png_bytes(state: np.ndarray) -> bytes:
    """Encode an exact state as a lossless RGB PNG using the frozen palette."""

    array = np.asarray(state, dtype=np.uint8)
    if array.ndim != 2 or np.any(array > 3):
        raise ValueError("invalid four-state array")
    rgb = STATE_RGB[array]
    stream = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def state_data_uri(state: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(state_png_bytes(state)).decode("ascii")


def state_record_from_npz(
    path: str | Path,
    *,
    source_task: str,
    seed: int,
    density: float,
    branch_point: str,
    intervention: str,
    requested_elapsed: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = Path(path)
    state, phase, elapsed = load_exact_npz(target)
    if crop is None:
        crop = (0, 0, int(state.shape[1]), int(state.shape[0]))
    record = make_panel_record(
        source_task=source_task,
        seed=seed,
        density=density,
        branch_point=branch_point,
        intervention=intervention,
        phase=phase,
        elapsed=elapsed,
        requested_elapsed=requested_elapsed,
        state_hash=state_sha256(state),
        crop=crop,
        source_path=target,
    )
    return state, record


def explorer_frame(
    state: np.ndarray,
    panel: dict[str, Any],
    *,
    metrics: dict[str, Any] | None = None,
    events: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_panel_record(panel)
    histogram = state_histogram(state).tolist()
    return {
        "panel": panel,
        "image": state_data_uri(state),
        "width": int(state.shape[1]),
        "height": int(state.shape[0]),
        "histogram": histogram,
        "Q": int(histogram[1] - histogram[2]),
        "M": int(sum(histogram[1:])),
        "metrics": metrics or {},
        "events": list(events or []),
    }


def _safe_script_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")


def build_history_explorer(
    output_path: str | Path,
    *,
    frames_by_density: dict[str, dict[str, list[dict[str, Any]]]],
    fixed_crops: dict[str, dict[str, int]],
    title: str,
) -> dict[str, Any]:
    """Write a self-contained, network-free synchronized exact-state explorer."""

    if not frames_by_density:
        raise ValueError("explorer requires at least one density")
    for branches in frames_by_density.values():
        if not branches:
            raise ValueError("each density requires at least one branch")
        for frames in branches.values():
            if not frames:
                raise ValueError("each branch requires at least one exact frame")
            for frame in frames:
                validate_panel_record(frame["panel"])
                if not str(frame.get("image", "")).startswith("data:image/png;base64,"):
                    raise ValueError("explorer frames must embed lossless PNG data")

    payload = {
        "schema": "zws-reversible-null-phase-v032-explorer-v1",
        "title": title,
        "rule_hash": RULE_SHA256,
        "palette": dict(zip(STATE_NAMES, STATE_HEX, strict=True)),
        "densities": frames_by_density,
        "fixed_crops": fixed_crops,
    }
    data = _safe_script_json(payload)
    html_text = _EXPLORER_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
    target = Path(output_path)
    atomic_write_text(target, html_text)
    _write_sidecar(target)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "densities": sorted(frames_by_density),
        "self_contained": True,
        "network_requests": False,
    }


def build_history_dashboard(
    output_path: str | Path,
    *,
    decision: dict[str, Any],
    results_rows: list[dict[str, Any]],
    survival_rows: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "zws-reversible-null-phase-v032-dashboard-v1",
        "rule_hash": RULE_SHA256,
        "decision": decision,
        "results": results_rows,
        "survival": survival_rows,
        "integrity": integrity,
    }
    data = _safe_script_json(payload)
    title = "ZWSLab v0.3.2 — State-matched history erasure"
    html_text = _DASHBOARD_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)
    target = Path(output_path)
    atomic_write_text(target, html_text)
    _write_sidecar(target)
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "self_contained": True,
        "network_requests": False,
    }


_COMMON_CSS = """
:root{color-scheme:dark;--bg:#071018;--panel:#101d29;--ink:#f6f8fa;--muted:#9fb0bf;--line:#263746;--accent:#f0e442;--blue:#56b4e9;--pink:#cc79a7}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#14283a 0,#071018 48%);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:22px 28px;border-bottom:1px solid var(--line);background:#071018e8;position:sticky;top:0;z-index:5}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 12px}.sub{color:var(--muted)}main{padding:22px;max-width:1600px;margin:auto}.card{background:linear-gradient(150deg,#132331,#0b1721);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 12px 30px #0005}.grid{display:grid;gap:14px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button,select,input{accent-color:var(--accent);background:#122433;color:var(--ink);border:1px solid #385066;border-radius:6px;padding:7px}button{cursor:pointer}.pill{padding:4px 9px;border-radius:999px;background:#172a3a;border:1px solid #365066;color:var(--muted)}.legend{display:flex;gap:12px;flex-wrap:wrap}.swatch{width:12px;height:12px;display:inline-block;border:1px solid #fff5;margin-right:5px}.small{font-size:12px;color:var(--muted)}code{color:#d7e9f6;word-break:break-all}canvas{background:#05070a;border:1px solid #2b4051;border-radius:8px;image-rendering:pixelated;max-width:100%}
"""


_EXPLORER_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>""" + _COMMON_CSS + """
#worlds{grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}.world canvas{width:100%;aspect-ratio:1}.timeline{min-width:280px;flex:1}.metric{height:190px;width:100%}.hidden{display:none}
</style></head><body><header><h1>__TITLE__</h1><div class="sub">Exact saved states only · local and offline · no upload</div></header><main class="grid">
<section class="card controls"><label>Density <select id="density"></select></label><label class="timeline">Exact time <input id="time" type="range" min="0" max="0" step="1" style="width:100%"></label><span id="timeLabel" class="pill"></span><label>View <select id="view"><option value="full">Full world</option><option value="crop">Frozen crop</option></select></label><button id="reset">Reset camera</button><button id="fullExport">Export full frame</button><button id="viewExport">Export viewport</button><button id="bookmark">Bookmark JSON</button></section>
<section class="card grid"><div><strong>Branches</strong><div class="legend" id="branchToggles"></div></div><div><strong>States</strong><div class="legend" id="stateToggles"></div></div><div class="small">Wheel to zoom; drag to pan. Camera is synchronized across panels. Click a cell for exact state and coordinate.</div></section>
<section id="worlds" class="grid"></section>
<section class="card"><h2>Low-k and finite-domain history</h2><canvas id="metric" class="metric" width="1400" height="360"></canvas><div id="markers" class="small"></div></section>
<section class="card"><h2>Exact metadata</h2><pre id="meta" class="small"></pre></section>
</main><script id="zws-data" type="application/json">__DATA__</script><script>
'use strict';const DATA=JSON.parse(document.getElementById('zws-data').textContent);const palette=DATA.palette;const names=Object.keys(palette);const rgb=Object.fromEntries(names.map((n,i)=>[palette[n].toLowerCase(),i]));let density=Object.keys(DATA.densities)[0],idx=0,scale=1,panX=0,panY=0,drag=null;const visible=[true,true,true,true];const $=s=>document.querySelector(s);const densitySel=$('#density'),time=$('#time'),worlds=$('#worlds');Object.keys(DATA.densities).forEach(d=>densitySel.add(new Option(d,d)));names.forEach((n,i)=>{const l=document.createElement('label');l.innerHTML=`<input type="checkbox" checked data-state="${i}"><span class="swatch" style="background:${palette[n]}"></span>${n}`;$('#stateToggles').append(l)});function branches(){return DATA.densities[density]}function maxFrames(){return Math.max(...Object.values(branches()).map(x=>x.length))}function frameAt(frames){return frames[Math.min(idx,frames.length-1)]}function rebuild(){worlds.innerHTML='';Object.entries(branches()).forEach(([name,frames])=>{const el=document.createElement('article');el.className='card world';el.innerHTML=`<h2>${name}</h2><canvas width="720" height="720"></canvas><div class="small readout"></div>`;worlds.append(el);const c=el.querySelector('canvas');c._frames=frames;c._image=new Image();c._image.onload=()=>drawOne(c);wireCanvas(c)});time.max=String(maxFrames()-1);idx=Math.min(idx,+time.max);loadFrames()}function loadFrames(){document.querySelectorAll('.world canvas').forEach(c=>{const f=frameAt(c._frames);c._frame=f;c._image.src=f.image});const first=frameAt(Object.values(branches())[0]);$('#timeLabel').textContent=`elapsed ${first.panel.elapsed??'—'} · phase ${first.panel.phase}`;$('#meta').textContent=JSON.stringify(Object.fromEntries(Object.entries(branches()).map(([n,f])=>[n,frameAt(f).panel])),null,2);drawMetric();drawMarkers()}function cameraFor(frame){if($('#view').value==='crop'){const c=DATA.fixed_crops[density];return {x:c.x,y:c.y,w:c.width,h:c.height}}return {x:0,y:0,w:frame.width,h:frame.height}}function drawOne(c){const f=c._frame;if(!f||!c._image.complete)return;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.imageSmoothingEnabled=false;ctx.fillStyle=palette.empty;ctx.fillRect(0,0,c.width,c.height);const base=cameraFor(f),w=base.w/scale,h=base.h/scale,x=base.x+panX+(base.w-w)/2,y=base.y+panY+(base.h-h)/2;ctx.drawImage(c._image,x,y,w,h,0,0,c.width,c.height);if(visible.some(v=>!v)){const im=ctx.getImageData(0,0,c.width,c.height),d=im.data;for(let p=0;p<d.length;p+=4){const key='#'+[d[p],d[p+1],d[p+2]].map(v=>v.toString(16).padStart(2,'0')).join('');const s=rgb[key];if(s!==undefined&&!visible[s])d[p+3]=0}ctx.clearRect(0,0,c.width,c.height);ctx.putImageData(im,0,0)}c._camera={x,y,w,h};c.parentElement.querySelector('.readout').textContent=`Q=${f.Q.toLocaleString()} · M=${f.M.toLocaleString()} · exact phase ${f.panel.phase} · ${f.panel.state_hash.slice(0,12)}…`}function drawAll(){document.querySelectorAll('.world canvas').forEach(drawOne)}function wireCanvas(c){c.addEventListener('wheel',e=>{e.preventDefault();scale=Math.max(1,Math.min(32,scale*Math.exp(-e.deltaY*.001)));drawAll()},{passive:false});c.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,px:panX,py:panY};c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(drag){const cam=c._camera;panX=drag.px-(e.clientX-drag.x)/c.clientWidth*cam.w;panY=drag.py-(e.clientY-drag.y)/c.clientHeight*cam.h;drawAll()}});c.addEventListener('pointerup',e=>{if(drag&&Math.hypot(e.clientX-drag.x,e.clientY-drag.y)<4)inspect(c,e);drag=null});}function inspect(c,e){const r=c.getBoundingClientRect(),cam=c._camera,x=Math.floor(cam.x+(e.clientX-r.left)/r.width*cam.w),y=Math.floor(cam.y+(e.clientY-r.top)/r.height*cam.h),tmp=document.createElement('canvas');tmp.width=c._frame.width;tmp.height=c._frame.height;const q=tmp.getContext('2d');q.drawImage(c._image,0,0);const p=q.getImageData(Math.max(0,Math.min(tmp.width-1,x)),Math.max(0,Math.min(tmp.height-1,y)),1,1).data,key='#'+[p[0],p[1],p[2]].map(v=>v.toString(16).padStart(2,'0')).join(''),s=rgb[key];c.parentElement.querySelector('.readout').textContent+=` · cell (${x},${y})=${names[s]??'unknown'}`;}function drawMetric(){const c=$('#metric'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#08121b';ctx.fillRect(0,0,c.width,c.height);const all=Object.entries(branches()).map(([n,fs])=>[n,fs.map(f=>f.metrics).filter(m=>Number.isFinite(+m.low_k_power))]);const vals=all.flatMap(x=>x[1].map(m=>+m.low_k_power));if(!vals.length){ctx.fillStyle='#9fb0bf';ctx.fillText('No low-k series in retained exact-frame metadata.',30,50);return}const ymax=Math.max(...vals,1),colors=['#f0e442','#56b4e9','#cc79a7','#e69f00'];ctx.lineWidth=3;all.forEach(([n,ms],j)=>{ctx.beginPath();ms.forEach((m,i)=>{const x=50+i/Math.max(1,ms.length-1)*(c.width-90),y=c.height-35-(+m.low_k_power/ymax)*(c.height-70);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=colors[j%colors.length];ctx.stroke();ctx.fillStyle=colors[j%colors.length];ctx.fillText(n,70+j*220,22)});}function drawMarkers(){const events=Object.entries(branches()).flatMap(([n,fs])=>fs.flatMap(f=>(f.events||[]).map(e=>`${n}: ${e.label??'event'} @ ${e.elapsed??e.phase}`)));$('#markers').textContent=events.length?'Event markers: '+[...new Set(events)].join(' · '):'No event marker at this exact frame.'}function download(name,blob){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}function exportCanvas(c,name){c.toBlob(b=>download(name,b),'image/png')}densitySel.onchange=()=>{density=densitySel.value;idx=0;resetCamera();rebuild()};time.oninput=()=>{idx=+time.value;loadFrames()};$('#view').onchange=()=>{resetCamera();drawAll()};$('#reset').onclick=resetCamera;function resetCamera(){scale=1;panX=panY=0;drawAll()}document.querySelectorAll('[data-state]').forEach(x=>x.onchange=()=>{visible[+x.dataset.state]=x.checked;drawAll()});$('#fullExport').onclick=()=>{const c=document.querySelector('.world canvas'),old=$('#view').value;$('#view').value='full';resetCamera();drawOne(c);exportCanvas(c,'zws-full-'+density+'.png');$('#view').value=old;drawAll()};$('#viewExport').onclick=()=>exportCanvas(document.querySelector('.world canvas'),'zws-viewport-'+density+'.png');$('#bookmark').onclick=()=>download('zws-bookmark.json',new Blob([JSON.stringify({density,index:idx,view:$('#view').value,scale,panX,panY,frames:Object.fromEntries(Object.entries(branches()).map(([n,f])=>[n,frameAt(f).panel]))},null,2)],{type:'application/json'}));rebuild();
</script><script id="zws-branch-controls">
'use strict';
const branchVisibleV032=Object.create(null);
const branchEntriesV032=()=>Object.entries(branches()).filter(([name])=>branchVisibleV032[name]!==false);
function rebuildBranchTogglesV032(){
  const host=$('#branchToggles');host.innerHTML='';
  Object.keys(branches()).forEach(name=>{
    if(branchVisibleV032[name]===undefined)branchVisibleV032[name]=true;
    const label=document.createElement('label'),input=document.createElement('input');
    input.type='checkbox';input.checked=branchVisibleV032[name];input.setAttribute('data-branch',name);
    input.onchange=()=>{
      const anotherVisible=Object.keys(branches()).some(other=>other!==name&&branchVisibleV032[other]!==false);
      if(!input.checked&&!anotherVisible){input.checked=true;return}
      branchVisibleV032[name]=input.checked;rebuildVisibleWorldsV032();
    };
    label.append(input,document.createTextNode(name));host.append(label);
  });
}
function rebuildVisibleWorldsV032(){
  worlds.innerHTML='';
  branchEntriesV032().forEach(([name,frames])=>{
    const el=document.createElement('article');el.className='card world';el.setAttribute('data-branch',name);
    el.innerHTML=`<h2>${name}</h2><canvas width="720" height="720"></canvas><div class="small readout"></div>`;
    worlds.append(el);const c=el.querySelector('canvas');c._frames=frames;c._image=new Image();c._image.onload=()=>drawOne(c);wireCanvas(c);
  });
  time.max=String(maxFrames()-1);idx=Math.min(idx,+time.max);loadFrames();
}
maxFrames=()=>Math.max(1,...branchEntriesV032().map(([,frames])=>frames.length));
loadFrames=()=>{
  document.querySelectorAll('.world canvas').forEach(c=>{const f=frameAt(c._frames);c._frame=f;c._image.src=f.image});
  const entries=branchEntriesV032(),first=entries.length?frameAt(entries[0][1]):null;
  $('#timeLabel').textContent=first?`elapsed ${first.panel.elapsed??'-'} · phase ${first.panel.phase}`:'no branches visible';
  $('#meta').textContent=JSON.stringify({visible_branches:entries.map(([name])=>name),frames:Object.fromEntries(entries.map(([name,frames])=>[name,frameAt(frames).panel]))},null,2);
  drawMetric();drawMarkers();
};
drawMetric=()=>{
  const c=$('#metric'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#08121b';ctx.fillRect(0,0,c.width,c.height);
  const all=branchEntriesV032().map(([name,frames])=>[name,frames.map(frame=>frame.metrics).filter(metric=>Number.isFinite(+metric.low_k_power))]);
  const vals=all.flatMap(([,metrics])=>metrics.map(metric=>+metric.low_k_power));
  if(!vals.length){ctx.fillStyle='#9fb0bf';ctx.fillText('No low-k series in retained exact-frame metadata.',30,50);return}
  const ymax=Math.max(...vals,1),colors=['#f0e442','#56b4e9','#cc79a7','#e69f00'],branchOrder=Object.keys(branches());ctx.lineWidth=3;
  all.forEach(([name,metrics])=>{const j=branchOrder.indexOf(name),color=colors[j%colors.length];ctx.beginPath();metrics.forEach((metric,i)=>{const x=50+i/Math.max(1,metrics.length-1)*(c.width-90),y=c.height-35-(+metric.low_k_power/ymax)*(c.height-70);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=color;ctx.stroke();ctx.fillStyle=color;ctx.fillText(name,70+j*220,22)});
};
drawMarkers=()=>{
  const events=branchEntriesV032().flatMap(([name,frames])=>frames.flatMap(frame=>(frame.events||[]).map(event=>`${name}: ${event.label??'event'} @ ${event.elapsed??event.phase}`)));
  $('#markers').textContent=events.length?'Event markers: '+[...new Set(events)].join(' · '):'No event marker at this exact frame.';
};
rebuild=()=>{rebuildBranchTogglesV032();rebuildVisibleWorldsV032()};
$('#bookmark').onclick=()=>{const entries=branchEntriesV032();download('zws-bookmark.json',new Blob([JSON.stringify({density,index:idx,view:$('#view').value,scale,panX,panY,visible_branches:entries.map(([name])=>name),frames:Object.fromEntries(entries.map(([name,frames])=>[name,frameAt(frames).panel]))},null,2)],{type:'application/json'}))};
rebuild();
</script></body></html>"""


_DASHBOARD_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>""" + _COMMON_CSS + """
.hero{grid-template-columns:2fr 1fr 1fr}.big{font-size:24px;font-weight:700;color:var(--accent)}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid var(--line);text-align:left}.chart{width:100%;height:340px}@media(max-width:900px){.hero{grid-template-columns:1fr}}
</style></head><body><header><h1>__TITLE__</h1><div class="sub">Frozen rule · preregistered finite campaign · exact inventory-matched interventions</div></header><main class="grid"><section class="grid hero"><article class="card"><h2>Combined mechanism</h2><div id="mechanism" class="big"></div></article><article class="card"><h2>Branch A</h2><div id="a" class="big"></div></article><article class="card"><h2>Branch B</h2><div id="b" class="big"></div></article></section><section class="card"><h2>Restricted-mean paired delay</h2><canvas id="delay" class="chart" width="1400" height="420"></canvas><div class="small">Positive values mean the state-matched intervention delayed the frozen strong-domain crossing. Shaded guides mark ±8,192 phases.</div></section><section class="card"><h2>Fixed-time and survival evidence</h2><canvas id="survival" class="chart" width="1400" height="420"></canvas></section><section class="card"><h2>Integrity gates</h2><table><tbody id="checks"></tbody></table></section><section class="card"><h2>Interpretation boundary</h2><p>Q = N<sub>positive</sub> − N<sub>negative</sub> remains exactly zero. M = N<sub>positive</sub> + N<sub>negative</sub> + N<sub>bound</sub> is occupancy and is not conserved by the winner. The result is a formal computational finding for this frozen reversible automaton, not evidence for a physical cosmology or an equilibrium phase transition.</p><code id="rule"></code></section></main><script id="zws-data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('zws-data').textContent),dec=D.decision;document.getElementById('mechanism').textContent=dec.combined_mechanism;document.getElementById('a').textContent=dec.branch_A.classification;document.getElementById('b').textContent=dec.branch_B.classification;document.getElementById('rule').textContent=D.rule_hash;const checks=document.getElementById('checks');Object.entries(D.integrity.checks||{}).forEach(([k,v])=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${k.replaceAll('_',' ')}</td><td style="color:${v?'#70d6a1':'#ff6b6b'}">${v?'PASS':'FAIL'}</td>`;checks.append(tr)});function axes(ctx,c,min,max){ctx.strokeStyle='#385066';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(80,20);ctx.lineTo(80,c.height-55);ctx.lineTo(c.width-30,c.height-55);ctx.stroke();const y=v=>20+(max-v)/(max-min)*(c.height-75);[-8192,0,8192].forEach(v=>{ctx.strokeStyle=v===0?'#9fb0bf':'#f0e44266';ctx.beginPath();ctx.moveTo(80,y(v));ctx.lineTo(c.width-30,y(v));ctx.stroke();ctx.fillStyle='#9fb0bf';ctx.fillText(v.toLocaleString(),8,y(v)+4)});return y}const dc=document.getElementById('delay'),dx=dc.getContext('2d');dx.fillStyle='#08121b';dx.fillRect(0,0,dc.width,dc.height);const rows=D.results.filter(r=>r.rmst_delay!==''&&Number.isFinite(+r.rmst_delay)),vals=rows.flatMap(r=>[+r.ci95_low,+r.ci95_high,+r.rmst_delay]).filter(Number.isFinite),lim=Math.max(16384,...vals.map(Math.abs))*1.12,y=axes(dx,dc,-lim,lim),colors=['#f0e442','#56b4e9','#cc79a7','#e69f00'];rows.forEach((r,i)=>{const x=150+i*(dc.width-220)/Math.max(1,rows.length-1);dx.strokeStyle=colors[i%colors.length];dx.lineWidth=8;dx.beginPath();dx.moveTo(x,y(+r.ci95_low));dx.lineTo(x,y(+r.ci95_high));dx.stroke();dx.fillStyle=colors[i%colors.length];dx.beginPath();dx.arc(x,y(+r.rmst_delay),10,0,Math.PI*2);dx.fill();dx.save();dx.translate(x,dc.height-38);dx.rotate(-.35);dx.fillStyle='#d7e9f6';dx.fillText(r.comparison,0,0);dx.restore()});const sc=document.getElementById('survival'),sx=sc.getContext('2d');sx.fillStyle='#08121b';sx.fillRect(0,0,sc.width,sc.height);const surv=D.survival.filter(r=>r.density==='pooled'),groups=Object.groupBy?Object.groupBy(surv,r=>r.branch_point+' · '+r.intervention):surv.reduce((a,r)=>((a[r.branch_point+' · '+r.intervention]??=[]).push(r),a),{});sx.strokeStyle='#385066';sx.strokeRect(70,20,sc.width-100,sc.height-70);Object.entries(groups).forEach(([n,rs],j)=>{rs.sort((a,b)=>+a.time-+b.time);sx.strokeStyle=colors[j%colors.length];sx.lineWidth=3;sx.beginPath();rs.forEach((r,i)=>{const x=70+(+r.time/262144)*(sc.width-100),yy=20+(1-(+r.survival))*(sc.height-70);i?sx.lineTo(x,yy):sx.moveTo(x,yy)});sx.stroke();sx.fillStyle=colors[j%colors.length];sx.fillText(n,90+j*230,18)});
</script></body></html>"""
