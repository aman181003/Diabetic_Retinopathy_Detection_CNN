# Retina AI — Frontend Dashboard (Static)

This is a lightweight, production-ready frontend for an AI Diabetic Retinopathy detector.
It expects the backend `/predict` endpoint to return JSON (see `json-schema.json`).

## Files
- `index.html` — main UI
- `styles.css` — theme + animations
- `app.js` — client-side logic (softmax, temperature scaling, UI updates)
- `json-schema.json` — expected schema for backend response

## How to connect your model backend
The UI calls `POST /predict` with `FormData` (key `file`). Your backend should return JSON like:

```json
{
  "machine_report": {
    "predictions":[{"label":"No DR","score":0.032},{"label":"Mild","score":0.124},{"label":"Moderate","score":0.278},{"label":"Severe","score":0.341},{"label":"Proliferative","score":0.225}],
    "gradcam_image_url":"/tmp/overlay_123.png",
    "calibration":{"method":"temperature","temp":1.2}
  },
  "patient":{"id":"P-001","age":62,"notes":"screening"},
  "probs":{"No DR":0.032,"Mild":0.124, "Moderate":0.278, "Severe":0.341, "Proliferative":0.225}
}




// app.js — Retina AI dashboard client JS (no frameworks)


// Author: delivered as requested. Keep in sync with backend JSON schema.

/* ====== Configuration ====== */
const backendCalibrated = false;     // set true if backend returns final calibrated probabilities
const defaultTemperature = 1.0;      // used for temperature scaling (can be updated from dev panel)
const lowConfidenceThreshold = 60.0; // percent (UI shows warning when max prob < this)
const gaugeArcRadius = 50;

/* ====== Utils: softmax, temp scaling, platt/isotonic placeholders ====== */
function softmax(logits, temp=1.0){
  if(!Array.isArray(logits)) return [];
  const scaled = logits.map(x => x / Math.max(1e-6, temp));
  const max = Math.max(...scaled);
  const exps = scaled.map(v => Math.exp(v - max));
  const sum = exps.reduce((s,x)=>s+x,0);
  return exps.map(e => e / sum);
}

// Platt scaling (logistic regression on logits) — placeholder uses simple sigmoid with learned params
function plattScale(logits, a=1.0, b=0.0){
  // logits: array -> apply elementwise
  return logits.map(l => 1/(1+Math.exp(-(a*l + b))));
}

// isotonic calibration placeholder: in production provide a mapping table or model
function isotonicCalibrate(probs, mapping=null){
  // mapping: function or lookup. For now return probs unchanged
  return probs;
}

/* ====== DOM references ====== */
const fileInput = document.getElementById('fileInput');
const origImage = document.getElementById('origImage');
const overlayCanvas = document.getElementById('overlayCanvas');
const opacityRange = document.getElementById('opacity');
const fadeToggle = document.getElementById('fadeToggle');
const compareBtn = document.getElementById('compareBtn');
const predLabelEl = document.getElementById('predLabel');
const probsList = document.querySelector('.prob-list');
const gaugeArc = document.getElementById('gaugeArc');
const gaugeText = document.getElementById('gaugeText');
const confState = document.getElementById('confState');
const heatFocus = document.getElementById('heatFocus');
const alertsArea = document.getElementById('alertsArea');
const devToggle = document.getElementById('devToggle');
const devPanel = document.getElementById('devPanel');
const diagLogits = document.getElementById('diagLogits');
const diagProbs = document.getElementById('diagProbs');
const diagTemp = document.getElementById('diagTemp');
const diagCal = document.getElementById('diagCal');
const downloadReport = document.getElementById('downloadReport');
const onboardBtn = document.getElementById('onboardBtn');
const tourCard = document.getElementById('tourCard');

/* ====== Canvas setup ====== */
const ctx = overlayCanvas.getContext('2d');

/* ====== Internal state ====== */
let lastMachineJson = null;
let currentHeatmap = null;    // expected 2D array or image
let currentOverlayImage = null;
let currentTemp = defaultTemperature;
let rawLogits = null;

/* ====== Accessibility: keyboard shortcuts ====== */
document.addEventListener('keydown', (e) => {
  if(e.key === 't') toggleTheme();
  if(e.key === 'd') toggleDev();
});

/* ====== Theme toggle ====== */
const themeBtn = document.getElementById('themeToggle');
themeBtn.addEventListener('click', toggleTheme);
function toggleTheme(){
  const body = document.body;
  const isLight = body.classList.contains('theme-light');
  if(isLight){
    body.classList.remove('theme-light');
    body.classList.add('theme-dark');
    themeBtn.textContent = '☀️';
    themeBtn.setAttribute('aria-pressed','true');
  } else {
    body.classList.remove('theme-dark');
    body.classList.add('theme-light');
    themeBtn.textContent = '🌙';
    themeBtn.setAttribute('aria-pressed','false');
  }
}

/* ====== Dev panel toggle ====== */
devToggle.addEventListener('click', toggleDev);
function toggleDev(){
  const hidden = devPanel.getAttribute('aria-hidden') === 'true';
  devPanel.setAttribute('aria-hidden', hidden ? 'false' : 'true');
}

/* ====== Onboard tour ====== */
onboardBtn.addEventListener('click', startTour);
function startTour(){
  tourCard.style.display = '';
  tourCard.setAttribute('aria-hidden','false');
  setTimeout(()=>{ tourCard.style.display='none'; tourCard.setAttribute('aria-hidden','true'); }, 7000);
}

/* ====== File handling & preview ====== */
fileInput.addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if(!f) return;
  const url = URL.createObjectURL(f);
  origImage.src = url;
  origImage.onload = () => {
    fitCanvasToImage();
    // POST to backend
    sendToBackend(f);
  };
});

function fitCanvasToImage(){
  const w = origImage.clientWidth;
  const h = origImage.clientHeight;
  overlayCanvas.width = origImage.naturalWidth;
  overlayCanvas.height = origImage.naturalHeight;
  overlayCanvas.style.width = origImage.clientWidth + 'px';
  overlayCanvas.style.height = origImage.clientHeight + 'px';
  overlayCanvas.getContext('2d').clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
}

/* ====== Backend comms: send file to /predict (adjust to your backend route) ====== */
async function sendToBackend(file){
  const fd = new FormData();
  fd.append('file', file);
  // optional: add metadata fields to form
  fd.append('patient_id','demo-123');
  try {
    showLoading(true);
    const resp = await fetch('/predict', { method:'POST', body: fd });
    if(!resp.ok){
      const text = await resp.text();
      showAlert('error', 'Server error: ' + text);
      showLoading(false);
      return;
    }
    const j = await resp.json();
    // j should contain machine_report (server still includes) and patient/clinician
    // For UI we will consume machine_report or fallback to j.probs
    let machine = j.machine_report || { predictions: j.probs ? Object.entries(j.probs).map(([k,v])=>({label:k,score:v})) : [] };
    // normalize shape: server might send predictions array or a dict
    if(machine && machine.predictions) {
      updateFromJson(machine, j);
    } else if(j.probs) {
      // convert dict->predictions
      const preds = Object.entries(j.probs).map(([k,v]) => ({label:k, score: v}));
      updateFromJson({predictions: preds, calibration: j.machine_report && j.machine_report.calibration ? j.machine_report.calibration : null}, j);
    } else {
      showAlert('error','Unexpected response from backend');
    }
  } catch(err){
    console.error(err);
    showAlert('error','Network or server error');
  } finally {
    showLoading(false);
  }
}

/* ====== Main UI update from machine JSON ======
   machine = { predictions: [{label,score | logits}], gradcam_image_url, calibration:{method,temp} }
   j = full backend response (optional fields patient, clinician)
*/
async function updateFromJson(machine, fullResponse=null){
  // detect if predictions are logits or probabilities
  let preds = machine.predictions || [];
  let scores = preds.map(p => (typeof p.score === 'number' ? p.score : 0));
  const labels = preds.map(p => p.label || '—');

  // decide: raw logits if any score > 1 or any negative or not summing to ~1
  let isProb = scores.every(s => s >= 0 && s <= 1);
  let sum = scores.reduce((a,b)=>a+b,0);
  if(isProb && Math.abs(sum-1) < 1e-3){
    // already probs
    rawLogits = null;
    var calibrated = scores.slice();
  } else {
    rawLogits = scores.slice();
    const temp = (machine.calibration && machine.calibration.temp) ? machine.calibration.temp : defaultTemperature;
    // do temperature-scaled softmax
    calibrated = softmax(scores, temp);
    diagTemp.textContent = temp.toFixed(2);
  }

  // optional: if you have isotonic/platt mapping model, call here: isotonicCalibrate([...])
  // calibrated = isotonicCalibrate(calibrated);

  // ensure numeric rounding & no exact 0 unless real
  calibrated = calibrated.map(p => {
    if(p === 0) return 0; // don't artificially inflate 0.0
    return Math.max(0, Math.min(1, Number(p)));
  });

  // normalize again just in case rounding changed sum
  const total = calibrated.reduce((s,x)=>s+x,0) || 1;
  calibrated = calibrated.map(x => x/total);

  // store lastMachineJson
  lastMachineJson = {
    predictions: labels.map((lab,i) => ({label:lab, score:calibrated[i]})),
    calibration: machine.calibration || {method: rawLogits? 'temperature': 'backend', temp: defaultTemperature}
  };

  // Dev diagnostics
  diagLogits.textContent = JSON.stringify(rawLogits || [], null, 2);
  diagProbs.textContent = JSON.stringify(lastMachineJson.predictions.map(p => ({label:p.label, score: (p.score*100).toFixed(1)+'%'})), null, 2);
  diagCal.textContent = lastMachineJson.calibration.method || 'none';

  // UI: show predictions sorted
  const sorted = lastMachineJson.predictions.slice().sort((a,b)=>b.score-a.score);
  renderProbList(sorted);
  // label: top prediction
  const top = sorted[0] || {label:'—', score:0};
  predLabelEl.textContent = `${top.label} (${(top.score*100).toFixed(1)}%)`;
  // gauge: max prob
  const conf = Math.round(top.score*100);
  animateGauge(conf);

  // conf state
  if(conf < lowConfidenceThreshold){
    showLowConfidenceWarning(conf);
  } else {
    clearAlerts();
    if(top.label === 'No DR' && top.score < 0.05){
      // if very low combined risk, subtle celebration
      celebration();
    }
  }

  // load gradcam image if provided in fullResponse or machine.gradcam_image_url
  const gradUrl = (fullResponse && fullResponse.overlay_url) || machine.gradcam_image_url || null;
  if(gradUrl){
    await loadHeatmapFromUrl(gradUrl);
  } else {
    // clear overlay
    clearOverlay();
  }

  // fill patient metadata if provided
  if(fullResponse && fullResponse.patient){
    document.getElementById('metaId').textContent = fullResponse.patient.id || '—';
    document.getElementById('metaAge').textContent = fullResponse.patient.age || '—';
    document.getElementById('metaLater').textContent = fullResponse.patient.laterality || '—';
    document.getElementById('metaTime').textContent = new Date().toLocaleString();
  }
}

/* ====== render probability list with animated bars ====== */
function renderProbList(sorted){
  probsList.innerHTML = '';
  sorted.forEach(p => {
    const row = document.createElement('div');
    row.className = 'prob-item';
    row.setAttribute('role','listitem');
    row.tabIndex = 0;
    row.innerHTML = `
      <div class="label" aria-hidden="true"><strong>${escapeHtml(p.label)}</strong></div>
      <div class="prob-bar" aria-hidden="true"><div class="prob-fill" style="width:0%"></div></div>
      <div class="pct" aria-label="${(p.score*100).toFixed(1)} percent">${(p.score*100).toFixed(1)}%</div>
    `;
    const fill = row.querySelector('.prob-fill');
    probsList.appendChild(row);
    // animate after insertion
    requestAnimationFrame(()=> {
      fill.style.width = `${(p.score*100).toFixed(1)}%`;
    });
    // hover microinteraction: show tooltip like reason (placeholder)
    row.addEventListener('mouseenter', ()=> {
      row.style.transform = 'translateY(-3px)';
      row.style.boxShadow = '0 8px 24px rgba(11,92,221,0.06)';
    });
    row.addEventListener('mouseleave', ()=> {
      row.style.transform = '';
      row.style.boxShadow = '';
    });
  });
}

/* ===== Gauge animation ===== */
function animateGauge(percent){
  const start = 0;
  const end = Math.max(0, Math.min(100, percent));
  let startTime = null;
  const duration = 800;
  function step(ts){
    if(!startTime) startTime = ts;
    const progress = Math.min(1, (ts - startTime)/duration);
    const cur = start + (end - start) * easeOutCubic(progress);
    drawGauge(cur);
    if(progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);

  // conf state label
  if(end < 60) { confState.textContent = 'Low confidence'; confState.style.color = 'var(--warn)'; }
  else if(end < 85) { confState.textContent = 'Medium confidence'; confState.style.color = '#ffb020'; }
  else { confState.textContent = 'High confidence'; confState.style.color = 'var(--success)'; }
}

function drawGauge(percent){
  const startAngle = -Math.PI; // left
  const endAngle = -Math.PI + (Math.PI) * (percent/100); // to right
  // build an arc path approximated for SVG path 'd' generation (semi-circle)
  // We'll approximate using SVG path already in index; set stroke-dasharray approach
  const circleCirc = Math.PI * 2 * 50; // approximate
  const visible = (percent/100) * (Math.PI * 50); // not used exactly — instead use strokeDasharray
  const arcLen = (percent/100) * (Math.PI * 2 * 50) / 2; // semi circle fraction
  const svgArc = document.getElementById('gaugeArc');
  if(svgArc){
    svgArc.style.strokeDasharray = `${arcLen} ${1000}`;
  }
  const text = document.getElementById('gaugeText');
  if(text) text.textContent = `${percent}%`;
}

/* ====== Heatmap loading & overlay drawing ====== */
async function loadHeatmapFromUrl(url){
  try {
    const resp = await fetch(url);
    const blob = await resp.blob();
    const img = await createImageBitmap(blob);
    // draw into canvas sized to displayed image
    overlayCanvas.width = img.width;
    overlayCanvas.height = img.height;
    overlayCanvas.style.width = origImage.clientWidth + 'px';
    overlayCanvas.style.height = origImage.clientHeight + 'px';
    ctx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
    ctx.globalAlpha = 0.6;
    // stretch heatmap to canvas
    ctx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
    currentOverlayImage = img;
    // estimate focus: compute center of mass on heatmap to describe region
    const focusText = estimateHeatmapFocus();
    heatFocus.textContent = focusText;
  } catch(err){
    console.warn('failed load heatmap', err);
    clearOverlay();
  }
}
function clearOverlay(){
  ctx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height);
  heatFocus.textContent = '—';
  currentOverlayImage = null;
}

/* crude focus estimator — placeholder */
function estimateHeatmapFocus(){
  try{
    // sample canvas pixels and find brightest region
    const w = overlayCanvas.width, h = overlayCanvas.height;
    const imgData = ctx.getImageData(0,0,w,h).data;
    let maxVal = 0, maxX=0, maxY=0;
    for(let y=0;y<h;y+=8){
      for(let x=0;x<w;x+=8){
        const i = (y*w + x) * 4;
        const v = imgData[i] + imgData[i+1] + imgData[i+2];
        if(v > maxVal){ maxVal = v; maxX = x; maxY = y; }
      }
    }
    if(maxVal <= 10) return 'no strong focus';
    const horiz = (maxX < w/3) ? 'nasal' : ((maxX > 2*w/3) ? 'temporal' : 'central');
    const vert = (maxY < h/3) ? 'superior' : ((maxY > 2*h/3) ? 'inferior' : 'central');
    return `${vert}-${horiz} quadrant`;
  }catch(e){ return 'unknown'; }
}

/* ====== Alerts & warnings ====== */
function showLowConfidenceWarning(conf){
  alertsArea.innerHTML = '';
  const el = document.createElement('div');
  el.className = 'alert-card alert-low';
  el.innerHTML = `<svg width=20 height=20 style="color:var(--warn)"><use href="#icon-warning"></use></svg>
    <div><strong>Low confidence</strong><div style="font-size:13px;color:var(--muted)">(${conf}%) — consider re-scan or specialist review</div></div>`;
  alertsArea.appendChild(el);
  // small animated attention microinteraction
  el.animate([{transform:'translateY(0)'},{transform:'translateY(-6px)'},{transform:'translateY(0)'}],{duration:900,iterations:2});
}
function showAlert(type, text){
  alertsArea.innerHTML = '';
  const el = document.createElement('div');
  el.className = 'alert-card ' + (type==='error' ? 'alert-low' : 'alert-high');
  el.innerHTML = `<div>${escapeHtml(text)}</div>`;
  alertsArea.appendChild(el);
}
function clearAlerts(){ alertsArea.innerHTML = ''; }

/* ====== helper: easing ====== */
function easeOutCubic(t){ return 1 - Math.pow(1 - t, 3); }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, function(m){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]); }); }

/* ====== Loading UI ====== */
function showLoading(loading=true){
  const card = document.getElementById('viewerCard');
  if(loading) card.classList.add('loading');
  else card.classList.remove('loading');
}

/* ====== clear overlay fade/compare handlers ====== */
opacityRange.addEventListener('input', ()=> {
  const v = Number(opacityRange.value)/100;
  overlayCanvas.style.opacity = String(v);
});
fadeToggle.addEventListener('click', ()=>{
  const pressed = fadeToggle.getAttribute('aria-pressed') === 'true';
  fadeToggle.setAttribute('aria-pressed', String(!pressed));
  if(!pressed){
    // fade effect: crossfade original <-> overlay by animating opacity
    overlayCanvas.style.transition = 'opacity .6s ease';
    overlayCanvas.style.opacity = String(Number(opacityRange.value)/100);
  } else {
    overlayCanvas.style.opacity = '0';
  }
});
compareBtn.addEventListener('click', ()=> {
  const pressed = compareBtn.getAttribute('aria-pressed') === 'true';
  compareBtn.setAttribute('aria-pressed', String(!pressed));
  if(!pressed){
    // show side-by-side by resizing layout (simple)
    document.body.classList.add('compare-mode');
  } else {
    document.body.classList.remove('compare-mode');
  }
});

/* ====== Celebration microinteraction for low risk ====== */
function celebration(){
  const el = document.createElement('div');
  el.style.position = 'fixed';
  el.style.left = '50%';
  el.style.top = '12%';
  el.style.transform = 'translateX(-50%)';
  el.style.background = 'linear-gradient(90deg,#effff6,#f0fff9)';
  el.style.padding = '10px 18px';
  el.style.borderRadius = '10px';
  el.style.boxShadow = '0 16px 48px rgba(12,20,40,0.06)';
  el.innerHTML = '<strong style="color:var(--success)">Low combined risk</strong> • No DR likely';
  document.body.appendChild(el);
  el.animate([{transform:'translateY(-10px)', opacity:0},{transform:'translateY(0)',opacity:1}],{duration:400});
  setTimeout(()=> {
    el.animate([{opacity:1},{opacity:0}],{duration:500}).onfinish = ()=> el.remove();
  }, 1800);
}

/* ====== Download report action: open backend URL if provided ====== */
downloadReport.addEventListener('click', ()=> {
  // use lastMachineJson to open report URL if backend provided it; fallback to alert
  // We stored full response in updateFromJson's fullResponse param earlier; here we assume server included report_url
  // For demo: open /download/report_latest.pdf
  window.open('/download/report_latest.pdf','_blank');
});

/* ====== Exported functions for dev / testing ====== */
window.ui = {
  updateFromJson,
  softmax,
  plattScale,
  isotonicCalibrate
};

// small initialization
document.addEventListener('DOMContentLoaded', ()=> {
  // initial gauge blank
  drawGauge(0);
  // hide tour initially
  tourCard.style.display = 'none';
});



*/

