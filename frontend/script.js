const conversation = document.getElementById('conversation');
const userInput = document.getElementById('user-input');
const chatForm = document.getElementById('chat-form');
const recordBtn = document.getElementById('record-btn');
const voiceTranscript = document.getElementById('voice-transcript');
const voiceStatus = document.getElementById('voice-status');
const cameraPreview = document.getElementById('camera-preview');
const cameraStatus = document.getElementById('camera-status');
const cameraEmptyState = document.getElementById('camera-empty-state');
const startCameraBtn = document.getElementById('start-camera');
const stopCameraBtn = document.getElementById('stop-camera');
const visionOverlay = document.getElementById('vision-overlay');
const visionDetailsPanel = document.getElementById('vision-details-panel');
const toggleVisionDetailsBtn = document.getElementById('toggle-vision-details');

const FRAME_INTERVAL_MS = 700;
const eventState = { face: null, attention: null, posture: null, decision: null };

let mediaRecorder = null;
let audioChunks = [];
let speechRecognition = null;
let speechFinalTranscript = '';
let speechInterimTranscript = '';
let speechRecognitionSupported = false;
let cameraStream = null;
let frameTimer = null;
let visionRequestInFlight = false;
let frameCaptureCanvas = null;
let analyzedFrames = 0;
let visionStartedAt = 0;
let latestVisionState = null;

function $(id) {
  return document.getElementById(id);
}

function appendMessage(sender, html) {
  if (sender === 'user') {
    conversation.innerHTML = '';
  }
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('flex', 'mb-2');
  if (sender === 'user') msgDiv.classList.add('justify-end');
  const bubble = document.createElement('div');
  bubble.classList.add('rounded-lg', 'p-3', 'max-w-md', 'break-words', 'shadow');
  bubble.classList.add(sender === 'user' ? 'bg-indigo-600' : 'bg-gray-200', sender === 'user' ? 'text-white' : 'text-gray-800');
  bubble.innerHTML = html;
  msgDiv.appendChild(bubble);
  conversation.appendChild(msgDiv);
  conversation.scrollTop = conversation.scrollHeight;
  return bubble;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderMarkdown(text) {
  let safe = escapeHtml(text || '').replace(/\r\n/g, '\n');
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/^#{1,3}\s+(.+)$/gm, '<strong>$1</strong>');
  safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/\*(.+?)\*/g, '<em>$1</em>');
  safe = safe.replace(/(^|\s)(\d+)\.\s+/g, (match, prefix, number, offset) => {
    const separator = offset === 0 ? '' : '<br>';
    return `${separator}<strong>${number}.</strong> `;
  });
  safe = safe.replace(/(^|\n)\s*[-*]\s+/g, '<br>&bull; ');
  safe = safe.replace(/\n/g, '<br>');
  return safe.replace(/^(<br>)+/, '');
}

function renderAssistantReply(data) {
  let reply = renderMarkdown(data.response_text || '');
  if (data.decision_type === 'action' && data.action) {
    reply += `<br><em>Action:</em> ${escapeHtml(data.action.name)}`;
    if (data.action.args && Object.keys(data.action.args).length) {
      reply += `<br><em>Args:</em> <code>${escapeHtml(JSON.stringify(data.action.args))}</code>`;
    }
  }
  if (data.action_result) {
    reply += `<br><em>Result:</em> <code>${escapeHtml(JSON.stringify(data.action_result))}</code>`;
  }
  const confPercent = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : '-';
  reply += `<br><small class="text-gray-500">Model: ${escapeHtml(data.model)}, Intent: ${escapeHtml(data.intent)}, Confidence: ${confPercent}</small>`;
  return reply;
}

function addEvent(message, level = 'info') {
  const timeline = $('event-timeline');
  const item = document.createElement('div');
  item.className = `event-item ${level}`;
  item.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
  timeline.prepend(item);
  while (timeline.children.length > 80) timeline.removeChild(timeline.lastChild);
}

function setVoiceStatus(text, active = false) {
  voiceStatus.textContent = text;
  voiceStatus.classList.toggle('is-active', active);
}

function setTranscript(text) {
  voiceTranscript.textContent = text || 'No speech recognized yet.';
}

function clearTranscriptAfterSend() {
  setTranscript('Mic transcript is hidden after sending.');
  setVoiceStatus('Idle');
}

function createSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    speechRecognitionSupported = false;
    return null;
  }
  speechRecognitionSupported = true;
  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  recognition.onresult = (event) => {
    speechInterimTranscript = '';
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript.trim();
      if (event.results[index].isFinal) speechFinalTranscript = `${speechFinalTranscript} ${transcript}`.trim();
      else speechInterimTranscript = `${speechInterimTranscript} ${transcript}`.trim();
    }
    setTranscript(speechInterimTranscript || speechFinalTranscript || 'Listening...');
  };
  recognition.onerror = (event) => setVoiceStatus(`Speech error: ${event.error}`);
  return recognition;
}

function setCameraStatus(text, active = false) {
  cameraStatus.textContent = text;
  cameraStatus.classList.toggle('is-active', active);
}

function setCameraPreviewVisible(visible) {
  cameraEmptyState.classList.toggle('hidden', visible);
  cameraPreview.style.opacity = visible ? '1' : '0';
  visionOverlay.style.opacity = visible ? '1' : '0';
}

function resetVisionStats(message = 'Start the camera to begin analysis.') {
  latestVisionState = null;
  setText('face-status', 'Not detected');
  setText('posture-status', 'Unknown');
  setText('tracking-status', 'Center');
  setText('fps-status', '0.0');
  setText('latency-status', '-');
  setText('analysis-time-status', '-');
  setText('vision-decision', message);
  clearOverlay();
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setCameraStatus('Camera unavailable');
    setText('vision-decision', 'This browser does not support camera access.');
    addEvent('Camera API unavailable', 'warning');
    return;
  }
  stopCamera();
  setCameraStatus('Requesting access...');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    setCameraPreviewVisible(true);
    setCameraStatus('Camera active', true);
    startCameraBtn.disabled = true;
    stopCameraBtn.disabled = false;
    analyzedFrames = 0;
    visionStartedAt = performance.now();
    setText('vision-decision', 'Analyzing camera frames...');
    addEvent('Camera started', 'info');
    startFrameLoop();
  } catch (err) {
    cameraStream = null;
    cameraPreview.srcObject = null;
    setCameraPreviewVisible(false);
    setCameraStatus('Camera stopped');
    setText('vision-decision', `Camera permission denied or unavailable: ${err}`);
    addEvent('Camera permission denied or unavailable', 'warning');
  }
}

function stopCamera() {
  stopFrameLoop();
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.pause();
  cameraPreview.srcObject = null;
  cameraPreview.removeAttribute('src');
  cameraPreview.load();
  setCameraPreviewVisible(false);
  setCameraStatus('Camera stopped');
  startCameraBtn.disabled = false;
  stopCameraBtn.disabled = true;
  resetVisionStats('Camera stopped');
}

function startFrameLoop() {
  stopFrameLoop();
  frameTimer = window.setInterval(captureAndAnalyzeFrame, FRAME_INTERVAL_MS);
  captureAndAnalyzeFrame();
}

function stopFrameLoop() {
  if (frameTimer) window.clearInterval(frameTimer);
  frameTimer = null;
  visionRequestInFlight = false;
}

async function captureAndAnalyzeFrame() {
  if (!cameraStream || visionRequestInFlight || !cameraPreview.videoWidth || !cameraPreview.videoHeight) return;
  visionRequestInFlight = true;
  try {
    const canvas = getFrameCaptureCanvas(cameraPreview.videoWidth, cameraPreview.videoHeight);
    canvas.getContext('2d').drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
    const res = await fetch('/vision/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: canvas.toDataURL('image/jpeg', 0.75),
        timestamp: new Date().toISOString(),
        include_decision: true,
        context: {
          sitting_minutes: Number(localStorage.getItem('sitting_minutes') || 0),
          sensor_data: {},
        },
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderVisionResult(await res.json(), canvas.width, canvas.height);
  } catch (err) {
    setText('vision-decision', 'Vision backend unavailable');
    setText('latency-status', '-');
    addEvent('Vision backend unavailable', 'warning');
  } finally {
    visionRequestInFlight = false;
  }
}

function getFrameCaptureCanvas(width, height) {
  if (!frameCaptureCanvas) frameCaptureCanvas = document.createElement('canvas');
  if (frameCaptureCanvas.width !== width || frameCaptureCanvas.height !== height) {
    frameCaptureCanvas.width = width;
    frameCaptureCanvas.height = height;
  }
  return frameCaptureCanvas;
}

function renderVisionResult(result, sourceWidth, sourceHeight) {
  latestVisionState = result;
  analyzedFrames += 1;
  const fps = analyzedFrames / Math.max(0.001, (performance.now() - visionStartedAt) / 1000);
  const face = result.face || {};
  const attention = result.eyes_attention || {};
  const body = result.body_posture || {};
  const tracking = result.tracking || {};
  const health = result.health_behavior || {};
  const sensors = result.sensors || {};
  const decision = result.decision || {};

  setText('face-status', face.detected ? `Detected (${face.count})` : 'Not detected');
  setText('posture-status', `${body.posture || 'unknown'} ${pct(body.confidence)}`);
  setText('tracking-status', tracking.direction || 'center');
  setText('fps-status', fps.toFixed(1));
  setText('latency-status', `${result.latency_ms} ms`);
  setText('analysis-time-status', formatTime(result.timestamp));

  setText('face-detected-value', String(Boolean(face.detected)));
  setText('face-count-value', String(face.count || 0));
  setText('face-confidence-value', pct(face.confidence));
  setText('gender-value', estimateText(face.apparent_gender_estimate));
  setText('age-value', estimateText(face.estimated_age_range));
  setText('expression-value', estimateText(face.expression));
  setText('face-center-value', coords(face.center_x, face.center_y));

  setText('eyes-visible-value', String(Boolean(attention.eyes_visible)));
  setText('eye-contact-value', attention.eye_contact || 'unknown');
  setText('attention-value', attention.attention_state || 'unknown');
  setText('blink-value', String(Boolean(attention.blink_detected)));
  setText('ear-value', attention.eye_aspect_ratio ?? '-');
  setText('distraction-value', attention.distraction_reason || 'none');

  setText('person-value', String(Boolean(body.person_detected)));
  setText('body-posture-value', body.posture || 'unknown');
  setText('posture-confidence-value', pct(body.confidence));
  setText('body-position-value', body.body_position || 'unknown');
  setText('sitting-minutes-value', String(body.sitting_minutes || 0));
  setText('posture-warning-value', body.posture_warning || 'None');

  setText('target-x-value', valueOrDash(tracking.target_x));
  setText('target-y-value', valueOrDash(tracking.target_y));
  setText('direction-value', tracking.direction || 'center');
  setText('motor-action-value', tracking.recommended_motor_action || 'hold_position');
  setText('motor-ready-value', String(Boolean(tracking.motor_ready)));
  setText('tracking-quality-value', tracking.tracking_quality || 'lost');

  setText('sitting-warning-value', String(Boolean(health.sitting_time_warning)));
  setText('water-value', health.water_reminder_status || 'ok');
  setText('medicine-value', health.medicine_reminder_status || 'ok');
  setText('fatigue-value', String(Boolean(health.fatigue_warning)));
  setText('drowsiness-value', String(Boolean(health.drowsiness_warning)));
  setText('attention-warning-value', String(Boolean(health.attention_warning)));
  setText('behavior-summary-value', health.behavior_summary || 'No alert.');

  setText('temperature-value', sensorValue(sensors.temperature, ' C'));
  setText('humidity-value', sensorValue(sensors.humidity, '%'));
  setText('distance-value', sensorValue(sensors.distance_cm, ' cm'));
  setText('light-value', sensors.light_level ?? '-');
  setText('battery-value', sensorValue(sensors.battery_percent, '%'));
  setText('motion-value', String(sensors.motion_detected ?? '-'));
  setText('edge-value', `${Boolean(sensors.edge_device_connected)} (${sensors.source || 'mock'})`);

  setText('vision-decision', decision.message || 'Analysis complete.');
  const alertLevel = decision.alert_level || 'normal';
  const alertEl = $('alert-level-value');
  alertEl.textContent = alertLevel;
  alertEl.className = `badge ${alertLevel}`;
  setText('decision-flags-value', `speak:${Boolean(decision.should_speak)} log:${Boolean(decision.should_log_event)} move:${Boolean(decision.should_move_robot)}`);

  drawOverlays(result.overlays || {}, sourceWidth, sourceHeight);
  emitVisionEvents(face, attention, body, decision);
}

function emitVisionEvents(face, attention, body, decision) {
  const faceState = face.detected ? 'detected' : 'lost';
  if (eventState.face !== faceState) {
    addEvent(face.detected ? 'Face detected' : 'Face lost', face.detected ? 'info' : 'warning');
    eventState.face = faceState;
  }
  if (eventState.attention !== attention.attention_state) {
    addEvent(`Attention changed: ${attention.attention_state || 'unknown'}`, attention.attention_state === 'distracted' ? 'warning' : 'info');
    eventState.attention = attention.attention_state;
  }
  if (eventState.posture !== body.posture) {
    addEvent(`Posture changed: ${body.posture || 'unknown'}`, 'info');
    eventState.posture = body.posture;
  }
  if (eventState.decision !== decision.message) {
    addEvent(`Decision: ${decision.message || 'none'}`, decision.alert_level || 'info');
    eventState.decision = decision.message;
  }
}

function drawOverlays(overlays, sourceWidth, sourceHeight) {
  const rect = visionOverlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  visionOverlay.width = Math.max(1, Math.round(rect.width * dpr));
  visionOverlay.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = visionOverlay.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const sx = rect.width / Math.max(1, sourceWidth);
  const sy = rect.height / Math.max(1, sourceHeight);

  drawBoxes(ctx, overlays.body_boxes || [], sx, sy, '#f59e0b', 'Person');
  drawBoxes(ctx, overlays.face_boxes || [], sx, sy, '#20c997', 'Face');
  drawPoints(ctx, overlays.eye_landmarks || [], rect.width, rect.height, '#38bdf8');
  drawPose(ctx, overlays.pose_points || [], rect.width, rect.height);
  drawGaze(ctx, overlays.gaze || {}, rect.width, rect.height, overlays.attention_indicator);
}

function drawBoxes(ctx, boxes, sx, sy, color, label) {
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.font = '700 13px system-ui, sans-serif';
  boxes.forEach((box) => {
    const w = box.width * sx;
    const h = box.height * sy;
    const x = ctx.canvas.clientWidth - ((box.x * sx) + w);
    const y = box.y * sy;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillText(`${label} ${pct(box.confidence)}`, x + 8, Math.max(18, y - 8));
  });
}

function drawPoints(ctx, points, width, height, color) {
  ctx.fillStyle = color;
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(width - (point.x * width), point.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawPose(ctx, points, width, height) {
  ctx.strokeStyle = '#a78bfa';
  ctx.fillStyle = '#a78bfa';
  points.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x * width, point.y * height, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawGaze(ctx, gaze, width, height, attention) {
  if (typeof gaze.from_x !== 'number' || typeof gaze.from_y !== 'number') return;
  const startX = width - (gaze.from_x * width);
  const startY = gaze.from_y * height;
  const delta = { left: [-42, 0], right: [42, 0], up: [0, -36], down: [0, 36], center: [0, -28] }[gaze.direction] || [0, -28];
  ctx.strokeStyle = attention === 'distracted' ? '#f59e0b' : '#38bdf8';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(startX + delta[0], startY + delta[1]);
  ctx.stroke();
}

function clearOverlay() {
  const ctx = visionOverlay.getContext('2d');
  ctx.clearRect(0, 0, visionOverlay.width, visionOverlay.height);
}

function latestVisionContext() {
  if (!latestVisionState) return {};
  const face = latestVisionState.face || {};
  const attention = latestVisionState.eyes_attention || {};
  const body = latestVisionState.body_posture || {};
  const tracking = latestVisionState.tracking || {};
  const health = latestVisionState.health_behavior || {};
  const alerts = [];
  if (health.sitting_time_warning) alerts.push('sitting_time_warning');
  if (health.attention_warning) alerts.push('attention_warning');
  if (health.drowsiness_warning) alerts.push('drowsiness_warning');
  return {
    vision: {
      face_detected: Boolean(face.detected),
      apparent_gender_estimate: face.apparent_gender_estimate || { label: 'unknown', confidence: 0 },
      estimated_age_range: face.estimated_age_range || { label: 'unknown', confidence: 0 },
      expression: face.expression || { label: 'unknown', confidence: 0 },
      attention_state: attention.attention_state || 'unknown',
      eye_contact: attention.eye_contact || 'unknown',
      posture: body.posture || 'unknown',
      tracking_direction: tracking.direction || 'center',
      health_alerts: alerts,
    },
  };
}

async function sendMessage(message) {
  appendMessage('user', message);
  clearTranscriptAfterSend();
  const placeholder = appendMessage('assistant', '<span class="italic text-gray-500">Thinking...</span>');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_text: message,
        memory: null,
        context: latestVisionContext(),
        retrieve_memory: true,
        memory_top_k: 3,
        memory_recent_k: 8,
        store_log: true,
      }),
    });
    if (!res.ok) {
      placeholder.innerHTML = `<span class="text-red-500">Error: ${await res.text()}</span>`;
      return;
    }
    const data = await res.json();
    placeholder.innerHTML = renderAssistantReply(data);
  } catch (err) {
    placeholder.innerHTML = `<span class="text-red-500">Error: ${err}</span>`;
  }
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = userInput.value.trim();
  if (text) {
    sendMessage(text);
    userInput.value = '';
  }
});

recordBtn.addEventListener('click', async () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    if (speechRecognition) speechRecognition.stop();
    recordBtn.classList.remove('is-recording');
    setVoiceStatus('Transcribing...');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Audio recording is not supported in this browser.');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    speechFinalTranscript = '';
    speechInterimTranscript = '';
    setTranscript('Listening...');
    setVoiceStatus('Listening', true);
    speechRecognition = createSpeechRecognition();
    if (speechRecognition) {
      try { speechRecognition.start(); } catch (err) { speechRecognition = null; }
    }
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) audioChunks.push(event.data);
    };
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      if (speechRecognitionSupported) await new Promise((resolve) => window.setTimeout(resolve, 350));
      const browserTranscript = speechFinalTranscript.trim();
      if (browserTranscript) {
        clearTranscriptAfterSend();
        sendMessage(browserTranscript);
        return;
      }
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const base64Audio = arrayBufferToBase64(await blob.arrayBuffer());
      setVoiceStatus('Backend STT');
      try {
        const sttRes = await fetch('/stt', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio_base64: base64Audio, filename: 'audio.webm' }),
        });
        if (!sttRes.ok) {
          appendMessage('assistant', `<span class="text-red-500">STT error: ${await sttRes.text()}</span>`);
          return;
        }
        const sttData = await sttRes.json();
        if (sttData.text) {
          clearTranscriptAfterSend();
          sendMessage(sttData.text);
        } else {
          setTranscript('No speech recognized.');
          setVoiceStatus('Idle');
        }
      } catch (err) {
        setVoiceStatus('STT error');
        appendMessage('assistant', `<span class="text-red-500">STT error: ${err}</span>`);
      }
    };
    mediaRecorder.start();
    recordBtn.classList.add('is-recording');
  } catch (err) {
    setVoiceStatus('Mic blocked');
    alert(`Unable to access microphone: ${err}`);
  }
});

async function fetchSystemInfo() {
  const container = $('system-info');
  try {
    const res = await fetch('/api/v1/system/info');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    container.innerHTML = '';
    Object.entries(data).forEach(([key, value]) => {
      const p = document.createElement('p');
      p.innerHTML = `<strong>${key}</strong>: ${value}`;
      container.appendChild(p);
    });
  } catch (err) {
    container.innerHTML = `<span class="text-red-500">Failed to load system info: ${err}</span>`;
  }
}

function setText(id, value) { $(id).textContent = value; }
function valueOrDash(value) { return value === null || value === undefined ? '-' : String(value); }
function sensorValue(value, unit) { return value === null || value === undefined ? '-' : `${value}${unit}`; }
function pct(value) { return typeof value === 'number' ? `${Math.round(value * 100)}%` : '0%'; }
function coords(x, y) { return typeof x === 'number' && typeof y === 'number' ? `${x}, ${y}` : '-'; }
function estimateText(item) {
  if (!item || item.label === 'unknown' || (item.confidence || 0) < 0.55) return `Unknown ${pct(item?.confidence || 0)}`;
  return `${item.label} ${pct(item.confidence)}`;
}
function formatTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '-' : date.toLocaleTimeString();
}

startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', () => {
  stopCamera();
  addEvent('Camera stopped', 'info');
});
toggleVisionDetailsBtn.addEventListener('click', () => {
  const collapsed = visionDetailsPanel.classList.toggle('is-collapsed');
  toggleVisionDetailsBtn.textContent = collapsed ? 'Expand' : 'Collapse';
});
$('toggle-info').addEventListener('click', () => {
  fetchSystemInfo();
  $('info-panel').classList.remove('hidden');
});
$('close-info').addEventListener('click', () => $('info-panel').classList.add('hidden'));
window.addEventListener('beforeunload', () => stopCamera());

addEvent('Dashboard loaded', 'info');
stopCamera();
