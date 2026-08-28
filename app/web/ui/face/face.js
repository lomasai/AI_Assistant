// The robot's face. It subscribes and nothing else: there is no route from
// this file to a step or to the orchestrator.
//
// No framework, no canvas, no redraw loop. Chromium is already the largest
// CPU consumer on the Pi, and every frame this file does not paint is a frame
// the detector gets.

(() => {
  const stage = document.getElementById('stage');
  const line = document.getElementById('line');
  const badge = document.getElementById('badge');
  const options = document.getElementById('options');
  const ribbon = document.getElementById('ribbon');
  const boxes = document.getElementById('boxes');
  const camera = document.getElementById('camera');
  const feed = document.getElementById('feed');

  const RECONNECT_MS = 1500;
  const SPEAK_HOLD_MS = 400;

  // Fetched, not written here. The engaged/drifting boundary is one number
  // and it belongs in config with the rest of them.
  let engagedAt = 0.45;

  const chips = new Map();     // student id -> element
  const seats = new Map();     // track id -> student id
  let base = 'sleeping';
  let settle = 0;

  if (new URLSearchParams(location.search).get('camera') === '1') {
    camera.hidden = false;
    feed.src = '/camera.mjpeg';
  }

  fetch('/api/display').then((r) => r.json()).then((cfg) => {
    engagedAt = cfg.attention_threshold;
    document.documentElement.style.setProperty('--base', cfg.base_font_px + 'px');
  }).catch(() => {});

  // --- state ---------------------------------------------------------------

  const show = (state) => { stage.dataset.state = state; };

  const rest = () => {
    clearTimeout(settle);
    settle = setTimeout(() => show(base), SPEAK_HOLD_MS);
  };

  const say = (text, who) => {
    line.textContent = text;
    badge.textContent = who || '';
    badge.hidden = !who;
    options.hidden = true;
    show('speaking');
  };

  // --- the class ribbon ----------------------------------------------------
  // Name and one dot. No number reaches this element by any path.

  const seat = (id, name) => {
    if (!id || chips.has(id)) return chips.get(id);
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.dataset.mood = 'away';
    const dot = document.createElement('i');
    const label = document.createElement('span');
    label.textContent = (name || '').split(' ')[0];
    chip.append(dot, label);
    ribbon.append(chip);
    chips.set(id, chip);
    return chip;
  };

  const mood = (id, value) => {
    const chip = chips.get(id);
    if (chip) chip.dataset.mood = value;
  };

  const addressing = (name) => {
    const first = name ? name.split(' ')[0] : '';
    for (const [id, chip] of chips) {
      if (first && chip.lastChild.textContent === first) mood(id, 'speaking');
      else if (chip.dataset.mood === 'speaking') mood(id, 'engaged');
    }
  };

  // --- boxes over the video ------------------------------------------------

  const drawBoxes = (tracks, width, height) => {
    if (camera.hidden || !width || !height) return;
    const live = new Set();

    for (const track of tracks) {
      const id = 't' + track.track_id;
      live.add(id);
      let box = document.getElementById(id);
      if (!box) {
        box = document.createElement('div');
        box.id = id;
        box.className = 'box';
        box.append(document.createElement('b'));
        boxes.append(box);
      }
      box.style.left = (track.x / width * 100) + '%';
      box.style.top = (track.y / height * 100) + '%';
      box.style.width = (track.w / width * 100) + '%';
      box.style.height = (track.h / height * 100) + '%';

      const chip = chips.get(track.student_id);
      const name = chip ? chip.lastChild.textContent : '';
      box.classList.toggle('named', Boolean(name));
      box.firstChild.textContent = name;
    }

    for (const box of [...boxes.children]) {
      if (!live.has(box.id)) box.remove();
    }
  };

  // --- events --------------------------------------------------------------

  const handlers = {
    'session.opened': () => { base = 'listening'; show(base); },
    'session.closed': () => { base = 'sleeping'; show(base); },

    'attendance.marked': (p) => seat(p.student_id, p.name),
    'student.identified': (p) => { seats.set(p.track_id, p.student_id); mood(p.student_id, 'engaged'); },
    'student.left': (p) => { seats.delete(p.track_id); mood(p.student_id, 'away'); },
    'student.disengaged': (p) => mood(p.student_id, 'drifting'),

    'robot.say': (p) => { say(p.text, p.student_name); addressing(p.student_name); },
    'robot.spoke': () => rest(),
    'robot.state': (p) => { if (p.state === 'idle') rest(); },

    'question.asked': (p) => {
      line.textContent = p.text;
      badge.hidden = true;
      show('thinking');
    },

    'lesson.segment': (p) => say(p.say, ''),

    'quiz.posed': (p) => {
      line.textContent = p.text;
      badge.hidden = true;
      options.replaceChildren(...(p.options || []).map((text, index) => {
        const item = document.createElement('li');
        const key = document.createElement('span');
        key.textContent = index + 1;
        item.append(key, document.createTextNode(text));
        return item;
      }));
      options.hidden = !(p.options || []).length;
      show('asking');
    },

    'vision.tracks': (p) => {
      const tracks = p.tracks || [];
      drawBoxes(tracks, p.width, p.height);

      const visible = new Set();
      for (const track of tracks) {
        if (!track.student_id) continue;
        visible.add(track.student_id);
        mood(track.student_id, track.attention >= engagedAt ? 'engaged' : 'drifting');
      }
      for (const id of chips.keys()) {
        if (!visible.has(id)) mood(id, 'away');
      }
    },
  };

  // --- the socket ----------------------------------------------------------

  const connect = () => {
    const socket = new WebSocket(`ws://${location.host}/events`);
    socket.onmessage = (message) => {
      const { event, payload } = JSON.parse(message.data);
      const handler = handlers[event];
      if (handler && payload) handler(payload);
    };
    socket.onclose = () => setTimeout(connect, RECONNECT_MS);
  };

  connect();
})();
