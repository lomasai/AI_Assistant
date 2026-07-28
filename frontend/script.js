const conversation = document.getElementById('conversation');
const userInput = document.getElementById('user-input');
const chatForm = document.getElementById('chat-form');
const recordBtn = document.getElementById('record-btn');
const voiceTranscript = document.getElementById('voice-transcript');
const voiceStatus = document.getElementById('voice-status');
const cameraPreview = document.getElementById('camera-preview');
const backendCameraPreview = document.getElementById('backend-camera-preview');
const cameraStatus = document.getElementById('camera-status');
const cameraEmptyState = document.getElementById('camera-empty-state');
const startCameraBtn = document.getElementById('start-camera');
const stopCameraBtn = document.getElementById('stop-camera');
const visionOverlay = document.getElementById('vision-overlay');
const visionDetailsPanel = document.getElementById('vision-details-panel');
const toggleVisionDetailsBtn = document.getElementById('toggle-vision-details');
const studentApp = document.getElementById('student-app');
const debugDashboard = document.getElementById('debug-dashboard');
const lessonSetupForm = document.getElementById('lesson-setup-form');
const answerForm = document.getElementById('answer-form');
const answerInput = document.getElementById('answer-input');
const studentBrowserCamera = document.getElementById('student-browser-camera');
const studentBackendCamera = document.getElementById('student-backend-camera');
const manualStudentSelect = document.getElementById('manual-student-select');

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
let cameraRuntimeMode = 'browser';
let backendCameraEvents = null;
let activeTeachingSessionId = localStorage.getItem('active_teaching_session_id') || null;
let studentCameraStream = null;
let activeRegistrationId = null;
let activeRegistrationSeed = null;
let studentAudioMuted = false;
let engagementPollTimer = null;

function $(id) {
  return document.getElementById(id);
}

function requireElement(id) {
  const element = $(id);
  if (!element) throw new Error(`Missing required UI element: #${id}`);
  return element;
}

function showDebugDashboard() {
  studentApp.classList.add('hidden');
  debugDashboard.classList.remove('hidden');
  loadStudentProfiles();
  refreshEngagementAdmin();
  refreshHardwareAdmin();
}

function showStudentApp() {
  debugDashboard.classList.add('hidden');
  studentApp.classList.remove('hidden');
}

function showStudentView(viewId) {
  ['startup-view', 'setup-view', 'live-teaching-view', 'summary-view'].forEach((id) => {
    const el = $(id);
    if (el) el.classList.toggle('hidden', id !== viewId);
  });
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
  if (!timeline) return;
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
  if (cameraEmptyState) cameraEmptyState.classList.toggle('hidden', visible);
  if (cameraPreview) cameraPreview.style.opacity = visible && cameraRuntimeMode === 'browser' ? '1' : '0';
  if (backendCameraPreview) backendCameraPreview.style.opacity = visible && cameraRuntimeMode !== 'browser' ? '1' : '0';
  if (visionOverlay) visionOverlay.style.opacity = visible ? '1' : '0';
}

async function loadCameraRuntimeMode() {
  try {
    const res = await fetch('/camera/status');
    if (!res.ok) return;
    const status = await res.json();
    cameraRuntimeMode = status.provider || 'browser';
  } catch (err) {
    cameraRuntimeMode = 'browser';
  }
}

async function initializeStudentCameraPreview() {
  await loadCameraRuntimeMode();
  const msg = $('student-camera-message');
  if (cameraRuntimeMode !== 'browser') {
    if (studentBackendCamera) {
      studentBackendCamera.src = `/camera/stream.mjpg?student=${Date.now()}`;
      studentBackendCamera.classList.remove('hidden');
    }
    if (studentBrowserCamera) studentBrowserCamera.classList.add('hidden');
    if (msg) msg.textContent = cameraRuntimeMode === 'disabled' ? 'Camera unavailable' : 'Camera preview';
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !studentBrowserCamera) {
    if (msg) msg.textContent = 'Camera preview unavailable';
    return;
  }
  try {
    studentCameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    studentBrowserCamera.srcObject = studentCameraStream;
    await studentBrowserCamera.play();
    if (studentBackendCamera) studentBackendCamera.classList.add('hidden');
    studentBrowserCamera.classList.remove('hidden');
    if (msg) msg.textContent = 'Camera preview';
  } catch (err) {
    if (msg) msg.textContent = 'Camera permission needed';
  }
}

function stopStudentCameraPreview() {
  if (studentCameraStream) studentCameraStream.getTracks().forEach((track) => track.stop());
  studentCameraStream = null;
  if (studentBackendCamera) {
    studentBackendCamera.removeAttribute('src');
    studentBackendCamera.classList.add('hidden');
  }
}

async function runSelfTest() {
  try {
    const health = await fetch('/api/v1/health');
    const camera = await fetch('/camera/status');
    setText('self-test-system', health.ok ? 'Ready' : 'Unavailable');
    if (camera.ok) {
      const status = await camera.json();
      setText('self-test-camera', status.state || 'Ready');
    } else {
      setText('self-test-camera', 'Optional');
    }
  } catch (err) {
    setText('self-test-system', 'Offline');
    setText('self-test-camera', 'Optional');
  }
}

function adminHeaders(extra = {}) {
  const token = localStorage.getItem('admin_api_token') || '';
  return { 'Content-Type': 'application/json', ...(token ? { 'X-Admin-Token': token } : {}), ...extra };
}

async function adminFetch(url, options = {}) {
  let res = await fetch(url, { ...options, headers: adminHeaders(options.headers || {}) });
  if (res.status === 403 && !localStorage.getItem('admin_api_token')) {
    const token = window.prompt('Admin token');
    if (token) {
      localStorage.setItem('admin_api_token', token);
      res = await fetch(url, { ...options, headers: adminHeaders(options.headers || {}) });
    }
  }
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function loadStudentProfiles() {
  const list = $('student-profile-list');
  try {
    const data = await adminFetch('/api/v1/admin/students');
    if (list) {
      list.innerHTML = '';
      data.students.forEach((student) => {
        const row = document.createElement('div');
        row.className = 'profile-row';
        const label = document.createElement('span');
        label.textContent = `${student.display_name} - ${student.registration_status}`;
        const button = document.createElement('button');
        button.className = 'secondary-button';
        button.type = 'button';
        button.textContent = 'Delete';
        button.addEventListener('click', () => deleteStudentProfile(student.id, student.display_name));
        row.append(label, button);
        list.appendChild(row);
      });
    }
  } catch (err) {
    if (list) list.textContent = 'Profiles unavailable';
  }
}

async function loadStudentOptions() {
  if (!manualStudentSelect) return;
  try {
    const res = await fetch('/api/v1/student/profiles');
    if (!res.ok) throw new Error('profiles unavailable');
    const data = await res.json();
    manualStudentSelect.innerHTML = '<option value="">Guest</option>';
    data.students.forEach((student) => {
      const option = document.createElement('option');
      option.value = student.id;
      option.textContent = student.display_name;
      manualStudentSelect.appendChild(option);
    });
  } catch (err) {
    manualStudentSelect.innerHTML = '<option value="">Guest</option>';
  }
}

async function recognizeStudentForHome() {
  try {
    const res = await fetch('/api/v1/student/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quality_override: 'ok' }),
    });
    if (!res.ok) throw new Error('recognition unavailable');
    const result = await res.json();
    setText('recognized-student-name', result.recognized ? result.display_name : 'Guest');
  } catch (err) {
    setText('recognized-student-name', 'Guest');
  }
}

async function startRegistration(event) {
  event.preventDefault();
  try {
    const payload = {
      display_name: $('registration-name').value.trim(),
      grade_level: $('registration-grade').value.trim() || null,
      language: $('registration-language').value.trim() || null,
      consent_given: $('registration-consent').checked,
    };
    const result = await adminFetch('/api/v1/admin/registrations', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    activeRegistrationId = result.registration.id;
    activeRegistrationSeed = `student:${result.student.id}`;
    setText('registration-status', 'Capturing');
    setText('registration-guidance', guidanceText(result.next_guidance));
    setText('registration-feedback', 'Capture centre, left and right samples.');
  } catch (err) {
    setText('registration-status', 'Error');
    setText('registration-feedback', 'Registration could not start.');
  }
}

async function captureRegistrationSample() {
  if (!activeRegistrationId) {
    setText('registration-feedback', 'Start registration first.');
    return;
  }
  try {
    const result = await adminFetch(`/api/v1/admin/registrations/${activeRegistrationId}/samples`, {
      method: 'POST',
      body: JSON.stringify({
        image_base64: captureRegistrationImage(),
        embedding_seed: activeRegistrationSeed,
        quality_override: 'ok',
      }),
    });
    setText('registration-guidance', guidanceText(result.next_guidance));
    setText('registration-feedback', result.accepted ? `Accepted ${result.registration.accepted_samples}/${result.registration.required_samples}` : qualityText(result.reason));
    if (result.registration.status === 'ready_to_verify') setText('registration-status', 'Ready to verify');
  } catch (err) {
    setText('registration-feedback', 'Sample was rejected.');
  }
}

function captureRegistrationImage() {
  const source = cameraPreview && cameraPreview.videoWidth ? cameraPreview : studentBrowserCamera;
  if (!source || !source.videoWidth || !source.videoHeight) return '';
  const canvas = getFrameCaptureCanvas(source.videoWidth, source.videoHeight);
  canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.75);
}

async function completeRegistration() {
  if (!activeRegistrationId) return;
  try {
    const result = await adminFetch(`/api/v1/admin/registrations/${activeRegistrationId}/complete`, { method: 'POST' });
    setText('registration-status', result.verified ? 'Registered' : 'Retry');
    setText('registration-feedback', result.verified ? 'Verification passed.' : 'Verification failed.');
    activeRegistrationId = null;
    await loadStudentProfiles();
    await loadStudentOptions();
  } catch (err) {
    setText('registration-feedback', 'Need more accepted samples before completion.');
  }
}

async function cancelRegistration() {
  if (!activeRegistrationId) return;
  try {
    await adminFetch(`/api/v1/admin/registrations/${activeRegistrationId}/cancel`, { method: 'POST' });
  } catch (err) {
    // Keep cancellation quiet in the UI; server state remains authoritative.
  }
  activeRegistrationId = null;
  setText('registration-status', 'Cancelled');
  setText('registration-feedback', 'Registration cancelled.');
  await loadStudentProfiles();
  await loadStudentOptions();
}

async function deleteStudentProfile(studentId, name) {
  if (!window.confirm(`Delete ${name}?`)) return;
  try {
    await adminFetch(`/api/v1/admin/students/${studentId}?confirm=true`, { method: 'DELETE' });
    await loadStudentProfiles();
    await loadStudentOptions();
  } catch (err) {
    setText('registration-feedback', 'Profile deletion failed.');
  }
}

function guidanceText(pose) {
  const labels = { center: 'Look straight at the camera.', left: 'Turn slightly left.', right: 'Turn slightly right.' };
  return labels[pose] || 'Hold still for capture.';
}

function qualityText(reason) {
  const labels = {
    dark_or_overexposed: 'Lighting is not suitable.',
    blurry: 'Frame is blurry.',
    no_face: 'No face detected.',
    multi_face: 'Only one face should be visible.',
  };
  return labels[reason] || 'Sample rejected.';
}

async function createTeachingSession(event) {
  event.preventDefault();
  const payload = {
    student_display_name: $('student-name-input').value.trim(),
    grade_level: $('student-level-input').value.trim(),
    topic: $('lesson-topic-input').value.trim(),
    language: $('lesson-language-input').value.trim(),
    objective: $('lesson-objective-input').value.trim(),
  };
  const created = await teachingFetch('/api/v1/teaching/sessions', { method: 'POST', body: JSON.stringify(payload) });
  activeTeachingSessionId = created.session.id;
  localStorage.setItem('active_teaching_session_id', activeTeachingSessionId);
  const started = await teachingFetch(`/api/v1/teaching/sessions/${activeTeachingSessionId}/start`, { method: 'POST' });
  renderTeachingSession(started.session);
  showStudentView('live-teaching-view');
  initializeStudentCameraPreview();
  startEngagementPolling();
}

async function submitTeachingAnswer(event) {
  event.preventDefault();
  if (!activeTeachingSessionId || !answerInput.value.trim()) return;
  const submit = $('answer-submit');
  submit.disabled = true;
  try {
    const result = await teachingFetch(`/api/v1/teaching/sessions/${activeTeachingSessionId}/answer`, {
      method: 'POST',
      body: JSON.stringify({ answer_text: answerInput.value.trim() }),
    });
    answerInput.value = '';
    renderTeachingSession(result.session);
  } finally {
    submit.disabled = false;
  }
}

async function teachingCommand(command) {
  if (!activeTeachingSessionId) return;
  if (command === 'pause' || command === 'stop') await cancelStudentAudio();
  const result = await teachingFetch(`/api/v1/teaching/sessions/${activeTeachingSessionId}/${command}`, { method: 'POST' });
  renderTeachingSession(result.session);
  if (command === 'stop') stopEngagementPolling();
}

async function teachingFetch(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function renderTeachingSession(session) {
  const latestTutor = [...(session.turns || [])].reverse().find((turn) => turn.role === 'tutor' && turn.tutor_output);
  const latestQuestion = [...(session.turns || [])].reverse().find((turn) => turn.state === 'asking_question');
  const summary = session.summary;
  setText('lesson-student-name', session.config.student_display_name);
  setText('lesson-state-label', stateLabel(session.state));
  setText('lesson-title', latestTutor?.tutor_output?.screen_title || session.config.topic);
  setText('lesson-objective', session.config.objective);
  const points = $('lesson-points');
  if (points) {
    points.innerHTML = '';
    (latestTutor?.tutor_output?.screen_points || [session.config.objective]).forEach((point) => {
      const li = document.createElement('li');
      li.textContent = point;
      points.appendChild(li);
    });
  }
  setText('lesson-question', latestQuestion?.text || 'The lesson is complete.');
  const progress = $('lesson-progress');
  if (progress) {
    progress.max = session.progress.max_turns;
    progress.value = session.progress.completed_turns;
  }
  if (answerInput) answerInput.disabled = session.state !== 'waiting_for_answer';
  if ($('answer-submit')) $('answer-submit').disabled = session.state !== 'waiting_for_answer';
  if (summary || session.state === 'session_complete') {
    setText('summary-text', summary?.recap || 'Session complete.');
    stopEngagementPolling();
    showStudentView('summary-view');
  } else {
    startEngagementPolling();
  }
}

async function startStudentVoiceTurn() {
  if (!activeTeachingSessionId) {
    setText('student-audio-message', 'Start a lesson first.');
    return;
  }
  setText('student-audio-state', 'Listening');
  setText('student-audio-message', 'Speak your answer.');
  try {
    const result = await teachingFetch('/api/v1/audio/push-to-talk/start', {
      method: 'POST',
      body: JSON.stringify({ session_id: activeTeachingSessionId }),
    });
    if (result.transcript) setText('student-transcript-preview', result.transcript);
    if (result.session) renderTeachingSession(result.session);
    setText('student-audio-state', result.ok ? 'Ready' : 'Audio unavailable');
    setText('student-audio-message', audioStatusText(result.status));
    if (result.ok && !studentAudioMuted) {
      // Backend TTS is queued by the voice turn; the UI only reflects state.
      pollStudentAudioState();
    }
  } catch (err) {
    setText('student-audio-state', 'Audio unavailable');
    setText('student-audio-message', 'Use text input instead.');
  }
}

async function cancelStudentAudio() {
  try {
    await fetch('/api/v1/audio/push-to-talk/cancel', { method: 'POST' });
  } catch (err) {
    // Text input remains available if cancellation cannot reach the backend.
  }
  setText('student-audio-state', 'Ready');
}

async function pollStudentAudioState() {
  try {
    const res = await fetch('/api/v1/audio/state');
    if (!res.ok) return;
    const state = await res.json();
    setText('student-audio-state', state.speaking ? 'Speaking' : state.listening ? 'Listening' : state.state);
  } catch (err) {
    setText('student-audio-state', 'Audio unavailable');
  }
}

function audioStatusText(status) {
  const labels = {
    submitted: 'Answer submitted.',
    transcribed: 'Transcript ready.',
    stt_failed: 'Speech recognition failed. Use text input.',
    timeout: 'No speech detected.',
    too_short: 'Speech was too short.',
    speaking: 'Wait until speaking finishes.',
    duplicate_or_invalid: 'Answer already submitted or lesson not ready.',
  };
  return labels[status] || 'Text input remains available.';
}

function stateLabel(state) {
  const labels = {
    waiting_for_answer: 'Listening',
    evaluating: 'Thinking',
    explaining: 'Explaining',
    paused: 'Paused',
    session_complete: 'Complete',
  };
  return labels[state] || state.replace(/_/g, ' ');
}

async function recoverTeachingSession() {
  await runSelfTest();
  if (!activeTeachingSessionId) {
    showStudentView('setup-view');
    return;
  }
  try {
    const result = await teachingFetch(`/api/v1/teaching/sessions/${activeTeachingSessionId}`);
    renderTeachingSession(result.session);
    if (result.session.state !== 'session_complete') {
      showStudentView('live-teaching-view');
      initializeStudentCameraPreview();
      startEngagementPolling();
    }
  } catch (err) {
    localStorage.removeItem('active_teaching_session_id');
    activeTeachingSessionId = null;
    showStudentView('setup-view');
  }
}

function startEngagementPolling() {
  if (!activeTeachingSessionId || engagementPollTimer) return;
  pollEngagementState();
  engagementPollTimer = window.setInterval(pollEngagementState, 2500);
}

function stopEngagementPolling() {
  if (engagementPollTimer) window.clearInterval(engagementPollTimer);
  engagementPollTimer = null;
  const panel = $('engagement-panel');
  if (panel) panel.classList.add('hidden');
}

async function pollEngagementState() {
  if (!activeTeachingSessionId) return;
  try {
    const res = await fetch(`/api/v1/engagement/sessions/${activeTeachingSessionId}/state`);
    if (!res.ok) return;
    renderEngagementState(await res.json());
  } catch (err) {
    const panel = $('engagement-panel');
    if (panel) panel.classList.add('hidden');
  }
}

function renderEngagementState(state) {
  const panel = $('engagement-panel');
  if (!panel) return;
  const show = Boolean(state.enabled && state.message && state.state !== 'normal' && state.state !== 'disabled');
  panel.classList.toggle('hidden', !show);
  if (show) setText('engagement-message', state.message);
}

async function sendEngagementChoice(choice) {
  if (!activeTeachingSessionId) return;
  try {
    const res = await fetch(`/api/v1/engagement/sessions/${activeTeachingSessionId}/choice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ choice }),
    });
    if (res.ok) renderEngagementState(await res.json());
    if (choice === 'pause') {
      const result = await teachingFetch(`/api/v1/teaching/sessions/${activeTeachingSessionId}`);
      renderTeachingSession(result.session);
    }
    if (choice === 'use_text' && answerInput) answerInput.focus();
  } catch (err) {
    renderEngagementState({ enabled: false });
  }
}

async function refreshEngagementAdmin() {
  const status = $('engagement-admin-status');
  const history = $('engagement-admin-history');
  try {
    const health = await fetch('/api/v1/engagement/health');
    if (health.ok) {
      const data = await health.json();
      if (status) status.textContent = data.enabled ? (data.running ? 'Running' : 'Configured') : 'Disabled';
    }
    if (history && activeTeachingSessionId) {
      const data = await adminFetch(`/api/v1/engagement/sessions/${activeTeachingSessionId}/history`);
      history.innerHTML = '';
      (data.events || []).slice(-6).forEach((item) => {
        const row = document.createElement('div');
        row.className = 'engagement-history-row';
        row.textContent = `${formatTime(item.timestamp_utc)} - ${item.message}`;
        history.appendChild(row);
      });
      if (!history.children.length) history.textContent = 'No support events.';
    }
  } catch (err) {
    if (status) status.textContent = 'Unavailable';
    if (history) history.textContent = 'Engagement support unavailable.';
  }
}

async function refreshHardwareAdmin() {
  const status = $('hardware-admin-status');
  const mode = $('hardware-mode');
  const limits = $('hardware-limits');
  const history = $('hardware-history');
  try {
    const health = await adminFetch('/api/v1/hardware/health', { method: 'GET' });
    if (status) status.textContent = health.state || 'Unavailable';
    if (mode) {
      mode.textContent = `${health.provider || 'mock'} mode. Physical output ${health.physical_output_enabled ? 'enabled' : 'disabled'}.`;
    }
    const safeLimits = health.limits || {};
    if (limits) {
      limits.textContent = `Servo ${safeLimits.servo_min_angle_deg}..${safeLimits.servo_max_angle_deg} deg, cooldown ${safeLimits.motion_cooldown_seconds}s, max motion ${safeLimits.max_continuous_motion_seconds}s.`;
    }
    if (history) {
      const result = await adminFetch('/api/v1/hardware/history', { method: 'GET' });
      history.innerHTML = '';
      (result.history || []).slice(-6).forEach((item) => {
        const row = document.createElement('div');
        row.className = 'engagement-history-row';
        row.textContent = `${formatTime(item.timestamp_utc)} - ${item.action}: ${item.status}`;
        history.appendChild(row);
      });
      if (!history.children.length) history.textContent = 'No commands.';
    }
  } catch (err) {
    if (status) status.textContent = 'Unavailable';
    if (mode) mode.textContent = 'Hardware controls unavailable.';
    if (limits) limits.textContent = 'Safety limits unavailable.';
  }
}

async function submitHardwareAction(action) {
  try {
    await adminFetch('/api/v1/hardware/actions', {
      method: 'POST',
      body: JSON.stringify({ action, params: {} }),
    });
  } catch (err) {
    // The backend owns safety decisions; rejected commands are reflected in health/history.
  }
  await refreshHardwareAdmin();
}

async function cancelHardwareMotion() {
  try {
    await adminFetch('/api/v1/hardware/cancel', { method: 'POST' });
  } catch (err) {
    // Keep the UI quiet; the backend state remains authoritative.
  }
  await refreshHardwareAdmin();
}

async function emergencyStopHardware() {
  try {
    await adminFetch('/api/v1/hardware/emergency-stop', { method: 'POST' });
  } catch (err) {
    // Emergency stop failures are surfaced through backend health.
  }
  await refreshHardwareAdmin();
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
  await loadCameraRuntimeMode();
  if (cameraRuntimeMode !== 'browser') {
    startBackendCameraPreview();
    return;
  }
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

function startBackendCameraPreview() {
  stopCamera();
  if (cameraRuntimeMode === 'disabled') {
    setCameraStatus('Camera disabled');
    setText('vision-decision', 'Backend camera is disabled by configuration.');
    addEvent('Backend camera disabled', 'warning');
    return;
  }
  if (!backendCameraPreview) return;
  backendCameraPreview.src = `/camera/stream.mjpg?ts=${Date.now()}`;
  backendCameraPreview.classList.remove('hidden');
  cameraPreview.classList.add('hidden');
  setCameraPreviewVisible(true);
  setCameraStatus('Backend camera active', true);
  startCameraBtn.disabled = true;
  stopCameraBtn.disabled = false;
  setText('vision-decision', 'Backend camera preview is active.');
  addEvent('Backend camera preview started', 'info');
  bindBackendCameraEvents();
}

function stopCamera() {
  if (backendCameraEvents) {
    backendCameraEvents.close();
    backendCameraEvents = null;
  }
  if (backendCameraPreview) {
    backendCameraPreview.removeAttribute('src');
    backendCameraPreview.classList.add('hidden');
  }
  if (cameraPreview) cameraPreview.classList.remove('hidden');
  stopFrameLoop();
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  if (cameraPreview) {
    cameraPreview.pause();
    cameraPreview.srcObject = null;
    cameraPreview.removeAttribute('src');
    try {
      cameraPreview.load();
    } catch (err) {
      addEvent(`Camera reset warning: ${err}`, 'warning');
    }
  }
  setCameraPreviewVisible(false);
  setCameraStatus('Camera stopped');
  if (startCameraBtn) startCameraBtn.disabled = false;
  if (stopCameraBtn) stopCameraBtn.disabled = true;
  resetVisionStats('Camera stopped');
}

function bindBackendCameraEvents() {
  if (!window.EventSource || backendCameraEvents) return;
  backendCameraEvents = new EventSource('/camera/events');
  backendCameraEvents.addEventListener('camera_status', (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.state === 'error' || data.state === 'disabled') {
        setCameraStatus(data.state === 'disabled' ? 'Camera disabled' : 'Camera unavailable');
        setText('vision-decision', data.error || 'Backend camera unavailable.');
        addEvent('Backend camera unavailable', 'warning');
      }
    } catch (err) {
      addEvent(`Camera status event warning: ${err}`, 'warning');
    }
  });
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
  setText('gender-value', 'Not used');
  setText('age-value', 'Not used');
  setText('expression-value', 'Not used');
  setText('face-center-value', coords(face.center_x, face.center_y));

  setText('eyes-visible-value', String(Boolean(attention.eyes_visible)));
  setText('eye-contact-value', attention.eye_contact || 'unknown');
  setText('attention-value', 'observable only');
  setText('blink-value', String(Boolean(attention.blink_detected)));
  setText('ear-value', attention.eye_aspect_ratio ?? '-');
  setText('distraction-value', 'No diagnostic label');

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
  setText('fatigue-value', 'Not used');
  setText('drowsiness-value', 'Not used');
  setText('attention-warning-value', String(Boolean(health.attention_warning)));
  setText('behavior-summary-value', 'No support cue.');

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
  if (eventState.attention !== attention.eyes_visible) {
    addEvent(`Eyes visible: ${Boolean(attention.eyes_visible)}`, 'info');
    eventState.attention = attention.eyes_visible;
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
  if (!visionOverlay) return;
  const ctx = visionOverlay.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, visionOverlay.width, visionOverlay.height);
}

function latestVisionContext() {
  if (!latestVisionState) return {};
  const face = latestVisionState.face || {};
  const body = latestVisionState.body_posture || {};
  const tracking = latestVisionState.tracking || {};
  return {
    vision: {
      face_detected: Boolean(face.detected),
      face_count: Number(face.count || 0),
      tracking_direction: tracking.direction || 'center',
      body_position: body.body_position || 'unknown',
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

async function handleRecordClick() {
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
}

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

function bindSettingsControls() {
  const opacityControl = $('hud-opacity-control');
  const contrastControl = $('hud-contrast-control');
  if (opacityControl) {
    opacityControl.addEventListener('input', () => {
      document.documentElement.style.setProperty('--hud-opacity', String(Number(opacityControl.value) / 100));
    });
  }
  if (contrastControl) {
    contrastControl.addEventListener('input', () => {
      document.documentElement.style.setProperty('--hud-strong-opacity', String(Number(contrastControl.value) / 100));
    });
  }
}

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}
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

function bindUiEvents() {
  const debugRequested = window.location.search.includes('admin') || window.location.hash === '#admin';
  if (debugRequested) showDebugDashboard();
  else showStudentApp();

  const openDebug = $('open-debug-view');
  if (openDebug) openDebug.addEventListener('click', showDebugDashboard);
  if (lessonSetupForm) lessonSetupForm.addEventListener('submit', createTeachingSession);
  if (manualStudentSelect) {
    manualStudentSelect.addEventListener('change', () => {
      const selected = manualStudentSelect.options[manualStudentSelect.selectedIndex];
      const name = selected && selected.value ? selected.textContent : 'Guest';
      setText('recognized-student-name', name);
      if ($('student-name-input')) $('student-name-input').value = name;
    });
  }
  if (answerForm) answerForm.addEventListener('submit', submitTeachingAnswer);
  if ($('student-mic-button')) $('student-mic-button').addEventListener('click', startStudentVoiceTurn);
  if ($('student-audio-cancel')) $('student-audio-cancel').addEventListener('click', cancelStudentAudio);
  if ($('student-audio-retry')) $('student-audio-retry').addEventListener('click', startStudentVoiceTurn);
  if ($('student-mute-tts')) {
    $('student-mute-tts').addEventListener('change', () => {
      studentAudioMuted = $('student-mute-tts').checked;
      if (studentAudioMuted) cancelStudentAudio();
    });
  }
  if ($('registration-form')) $('registration-form').addEventListener('submit', startRegistration);
  if ($('capture-registration-sample')) $('capture-registration-sample').addEventListener('click', captureRegistrationSample);
  if ($('complete-registration')) $('complete-registration').addEventListener('click', completeRegistration);
  if ($('cancel-registration')) $('cancel-registration').addEventListener('click', cancelRegistration);
  if ($('pause-lesson')) $('pause-lesson').addEventListener('click', () => teachingCommand('pause'));
  if ($('resume-lesson')) $('resume-lesson').addEventListener('click', () => teachingCommand('resume'));
  if ($('stop-lesson')) $('stop-lesson').addEventListener('click', () => teachingCommand('stop'));
  if ($('engagement-continue')) $('engagement-continue').addEventListener('click', () => sendEngagementChoice('continue'));
  if ($('engagement-repeat')) $('engagement-repeat').addEventListener('click', () => sendEngagementChoice('repeat'));
  if ($('engagement-pause')) $('engagement-pause').addEventListener('click', () => sendEngagementChoice('pause'));
  if ($('engagement-use-text')) $('engagement-use-text').addEventListener('click', () => sendEngagementChoice('use_text'));
  if ($('hardware-neutral')) $('hardware-neutral').addEventListener('click', () => submitHardwareAction('neutral'));
  if ($('hardware-nod')) $('hardware-nod').addEventListener('click', () => submitHardwareAction('small_nod'));
  if ($('hardware-turn')) $('hardware-turn').addEventListener('click', () => submitHardwareAction('small_head_turn'));
  if ($('hardware-reset')) $('hardware-reset').addEventListener('click', () => submitHardwareAction('reset_position'));
  if ($('hardware-cancel')) $('hardware-cancel').addEventListener('click', cancelHardwareMotion);
  if ($('hardware-estop')) $('hardware-estop').addEventListener('click', emergencyStopHardware);
  if ($('new-lesson')) {
    $('new-lesson').addEventListener('click', () => {
      localStorage.removeItem('active_teaching_session_id');
      activeTeachingSessionId = null;
      stopStudentCameraPreview();
      stopEngagementPolling();
      showStudentView('setup-view');
    });
  }

  requireElement('chat-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const text = userInput.value.trim();
    if (text) {
      sendMessage(text);
      userInput.value = '';
    }
  });

  requireElement('record-btn').addEventListener('click', handleRecordClick);
  requireElement('start-camera').addEventListener('click', startCamera);
  requireElement('stop-camera').addEventListener('click', () => {
    stopCamera();
    addEvent('Camera stopped', 'info');
  });
  requireElement('toggle-vision-details').addEventListener('click', () => {
    const collapsed = visionDetailsPanel.classList.toggle('is-collapsed');
    toggleVisionDetailsBtn.textContent = collapsed ? 'Expand' : 'Collapse';
  });
  requireElement('toggle-info').addEventListener('click', () => {
    fetchSystemInfo();
    requireElement('info-panel').classList.remove('hidden');
  });
  requireElement('close-info').addEventListener('click', () => requireElement('info-panel').classList.add('hidden'));
  requireElement('toggle-settings').addEventListener('click', () => requireElement('settings-panel').classList.remove('hidden'));
  requireElement('close-settings').addEventListener('click', () => requireElement('settings-panel').classList.add('hidden'));
  bindSettingsControls();
  window.addEventListener('beforeunload', () => stopCamera());
}

function initializeDashboard() {
  bindUiEvents();
  loadCameraRuntimeMode();
  loadStudentOptions();
  recognizeStudentForHome();
  recoverTeachingSession();
  addEvent('Dashboard loaded', 'info');
  stopCamera();
}

try {
  initializeDashboard();
} catch (err) {
  console.error('Dashboard failed to initialize:', err);
  setText('vision-decision', `Dashboard error: ${err.message || err}`);
}
