// The teacher's tablet. Three panels, one socket, no framework.
//
// Everything destructive goes through the API and comes back as an event, so
// this file never holds an opinion about what the robot is doing - it reads
// the same stream the face does.

(() => {
  const $ = (id) => document.getElementById(id);
  const post = (path, body) =>
    fetch('/api' + path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body || {}),
    }).then((r) => r.json());

  const RECONNECT_MS = 1500;
  const SWEEP_TICK_MS = 180;
  const RING = 327;

  let engagedAt = 0.45;
  let speaker = null;
  let paused = false;
  let enrolling = null;
  let sweeping = 0;

  fetch('/api/display').then((r) => r.json()).then((cfg) => { engagedAt = cfg.attention_threshold; });

  // --- panels --------------------------------------------------------------

  for (const tab of document.querySelectorAll('.tab')) {
    tab.onclick = () => {
      for (const other of document.querySelectorAll('.tab')) other.classList.toggle('on', other === tab);
      for (const panel of document.querySelectorAll('.panel')) {
        panel.classList.toggle('on', panel.id === tab.dataset.panel);
      }
      if (tab.dataset.panel === 'enrol') { $('enrolFeed').src = '/camera.mjpeg'; loadEnrolled(); }
      if (tab.dataset.panel === 'report') loadSessions();
    };
  }

  $('feed').src = '/camera.mjpeg';

  // --- starting and ending a class -----------------------------------------

  let teaching = false;

  fetch('/api/topics').then((r) => r.json()).then((body) => {
    $('topic').replaceChildren(...body.topics.map((lesson) => {
      const option = document.createElement('option');
      option.value = lesson.id;
      option.textContent = `${lesson.title} (${lesson.segments} parts)`;
      return option;
    }));
  });

  const showTeaching = (on) => {
    teaching = on;
    $('start').textContent = on ? 'End class' : 'Start class';
    $('start').classList.toggle('teaching', on);
    $('topic').disabled = on;
  };

  $('start').onclick = async () => {
    const body = teaching
      ? await post('/session/stop')
      : await post('/session/start', { topic: $('topic').value });
    if (body.error) { $('step').textContent = body.error; return; }
    showTeaching(!teaching);
  };

  // --- live control --------------------------------------------------------

  $('pause').onclick = async () => {
    await post(paused ? '/resume' : '/pause');
    paused = !paused;
    $('pause').textContent = paused ? 'Resume' : 'Pause';
    $('pause').classList.toggle('paused', paused);
  };

  $('skip').onclick = () => post('/skip');
  $('halt').onclick = () => post('/halt', { reason: 'teacher' });

  $('nudging').querySelector('input').onchange = (event) =>
    post('/nudging', { enabled: event.target.checked });

  $('ask').onsubmit = (event) => {
    event.preventDefault();
    const field = event.target.text;
    post('/ask', {
      text: field.value,
      student_id: speaker ? speaker.id : '',
      student_name: speaker ? speaker.name : '',
    });
    field.value = '';
  };

  $('answer').onsubmit = (event) => {
    event.preventDefault();
    const field = event.target.response;
    post('/answer', { response: field.value }).then((body) => {
      if (body.error) $('step').textContent = body.error;
    });
    field.value = '';
  };

  // --- hearing a child -----------------------------------------------------
  // Press to talk. The robot cannot tell who spoke in a room of forty, so the
  // tapped name is the attribution.

  const hear = async (button, asAnswer) => {
    if (!speaker) { $('step').textContent = 'Tap a name first, so the answer has an owner.'; return; }

    button.classList.add('hearing');
    button.disabled = true;
    const was = button.textContent;
    button.textContent = 'Listening…';
    try {
      const heard = await post('/listen', {
        student_id: speaker.id, student_name: speaker.name, as_answer: asAnswer,
      });
      if (heard.error) $('step').textContent = heard.error;
      else if (!heard.text) $('step').textContent = heard.reason || 'nothing was said';
      else $(asAnswer ? 'answer' : 'ask').querySelector('input').value = heard.text;
    } finally {
      button.classList.remove('hearing');
      button.disabled = false;
      button.textContent = was;
    }
  };

  $('listenAsk').onclick = (event) => hear(event.target, false);
  $('listenAnswer').onclick = (event) => hear(event.target, true);

  // --- the roster, and who is speaking -------------------------------------

  const chips = new Map();

  const drawRoster = (roster) => {
    if (chips.size === roster.length) return;
    $('roster').replaceChildren(...roster.map((student) => {
      const row = document.createElement('li');
      row.dataset.mood = 'away';
      const dot = document.createElement('i');
      const name = document.createElement('span');
      name.textContent = student.name;
      row.append(dot, name);
      row.onclick = () => choose(student, row);
      chips.set(student.id, row);
      return row;
    }));
  };

  const choose = (student, row) => {
    speaker = { id: student.id, name: student.name };
    for (const other of $('roster').children) other.classList.toggle('on', other === row);
    post('/speaker', { student_id: student.id, student_name: student.name });
  };

  const mood = (id, value) => {
    const row = chips.get(id);
    if (row) row.dataset.mood = value;
  };

  // --- boxes over the stream -----------------------------------------------

  const names = new Map();

  const drawBoxes = (tracks, width, height) => {
    if (!width || !height) return;
    const live = new Set();

    for (const track of tracks) {
      const id = 'b' + track.track_id;
      live.add(id);
      let box = document.getElementById(id);
      if (!box) {
        box = document.createElement('div');
        box.id = id;
        box.className = 'box';
        box.append(document.createElement('b'));
        $('boxes').append(box);
      }
      box.style.left = (track.x / width * 100) + '%';
      box.style.top = (track.y / height * 100) + '%';
      box.style.width = (track.w / width * 100) + '%';
      box.style.height = (track.h / height * 100) + '%';

      const name = names.get(track.student_id) || '';
      box.classList.toggle('named', Boolean(name));
      box.firstChild.textContent = name;
    }

    for (const box of [...$('boxes').children]) if (!live.has(box.id)) box.remove();
  };

  // --- enrolment -----------------------------------------------------------

  const ring = (done, needed) => {
    const share = needed ? Math.min(1, done / needed) : 0;
    document.querySelector('#ring .fill').style.strokeDashoffset = RING * (1 - share);
  };

  const loadEnrolled = () =>
    fetch('/api/enrol/students').then((r) => r.json()).then((body) => {
      $('enrolled').replaceChildren(...body.students.map((student) => {
        const row = document.createElement('li');
        const name = document.createElement('span');
        name.textContent = student.name;
        const count = document.createElement('small');
        count.textContent = student.vectors ? `${student.vectors} stored` : 'not enrolled';
        row.append(name, count);
        row.onclick = () => {
          if (!student.vectors) return;
          if (!confirm(`Remove ${student.name}'s face data?`)) return;
          fetch('/api/enrol/students/' + student.id, { method: 'DELETE' }).then(loadEnrolled);
        };
        return row;
      }));
    });

  $('enrolForm').onsubmit = async (event) => {
    event.preventDefault();
    const form = event.target;

    // The server refuses without a name against the consent anyway. This is
    // only so the teacher finds out before the child sits down.
    if (!form.agreed.checked || !form.granted_by.value.trim()) {
      $('coach').textContent = 'A parent or guardian has to be named before enrolling.';
      return;
    }

    const started = await post('/enrol/start', {
      name: form.name.value,
      roll_no: form.roll_no.value,
      consent: { granted_by: form.granted_by.value, document_ref: form.document_ref.value },
    });
    if (started.error) { $('coach').textContent = started.error; return; }

    enrolling = started;
    $('coach').textContent = 'Turn your head slowly, left to right.';
    sweeping = setInterval(sweepTick, SWEEP_TICK_MS);
    setTimeout(endSweep, started.sweep_seconds * 1000);
  };

  const sweepTick = async () => {
    if (!enrolling) return;
    const feedback = await post(`/enrol/${enrolling.enrolment_id}/frame`);
    if (feedback.error) { $('coach').textContent = feedback.error; return; }
    $('coach').textContent = feedback.reason;
    ring(feedback.collected, feedback.needed);
  };

  const endSweep = async () => {
    clearInterval(sweeping);
    if (!enrolling) return;
    const done = await post(`/enrol/${enrolling.enrolment_id}/finish`);
    enrolling = null;
    ring(0, 1);
    $('coach').textContent = done.error
      ? done.error
      : `${done.name}: ${done.vectors} vectors stored, no image kept.`;
    loadEnrolled();
  };

  // --- the report ----------------------------------------------------------

  const loadSessions = () =>
    fetch('/api/sessions').then((r) => r.json()).then((body) => {
      $('sessions').replaceChildren(...body.sessions.map((session) => {
        const option = document.createElement('option');
        option.value = session.id;
        option.textContent = `${session.topic} — ${new Date(session.started_at * 1000).toLocaleString()}`;
        return option;
      }));
      if (body.sessions.length) loadReport(body.sessions[0].id);
    });

  $('sessions').onchange = (event) => loadReport(event.target.value);

  const card = (title, ...children) => {
    const box = document.createElement('div');
    box.className = 'card';
    const heading = document.createElement('h3');
    heading.textContent = title;
    box.append(heading, ...children);
    return box;
  };

  const figure = (text) => {
    const value = document.createElement('div');
    value.className = 'figure';
    value.textContent = text;
    return value;
  };

  const loadReport = (id) =>
    fetch('/api/report/' + id).then((r) => r.json()).then((body) => {
      if (body.error) { $('reportBody').textContent = body.error; return; }

      const questions = document.createElement('div');
      questions.append(...body.questions.map((asked) => {
        const item = document.createElement('div');
        item.className = 'q';
        const text = document.createElement('b');
        text.textContent = asked.asked_by ? `${asked.asked_by}: ${asked.text}` : asked.text;
        const answer = document.createElement('span');
        answer.textContent = asked.answered;
        item.append(text, answer);
        return item;
      }));

      const table = document.createElement('table');
      const head = document.createElement('tr');
      for (const label of ['Roll', 'Name', 'Answered', 'Correct']) {
        const cell = document.createElement('th');
        cell.textContent = label;
        head.append(cell);
      }
      table.append(head);

      // Roll order, exactly as the server sent it. Sorting this table by
      // result would turn a record into a league table.
      for (const student of body.quiz.students) {
        const row = document.createElement('tr');
        for (const [value, numeric] of [[student.roll_no, false], [student.name, false],
                                        [student.answered, true], [student.correct, true]]) {
          const cell = document.createElement('td');
          cell.textContent = value;
          if (numeric) cell.className = 'num';
          row.append(cell);
        }
        table.append(row);
      }

      $('reportBody').replaceChildren(
        card('Attendance', figure(`${body.attendance.count} present`)),
        card('Lesson covered', figure(`${body.coverage.taught} of ${body.coverage.total}`)),
        card('What the class asked', questions),
        card('Quiz', table),
      );
    });

  // --- state and the stream ------------------------------------------------

  const refresh = () =>
    fetch('/api/state').then((r) => r.json()).then((body) => {
      drawRoster(body.roster);
      for (const student of body.roster) names.set(student.id, student.name);
      $('step').textContent = body.step ? `${body.state} — ${body.step}` : body.state;
      showTeaching(body.teaching);

      // No microphone is not a broken button, it is an absent one.
      const deaf = !body.microphone || body.microphone === 'none';
      for (const id of ['listenAsk', 'listenAnswer']) {
        $(id).disabled = deaf;
        $(id).title = deaf ? 'no microphone on this machine' : 'press and speak';
      }
    });

  const handlers = {
    'step.entered': (p) => { $('step').textContent = 'running — ' + p.step; },
    'step.skipped': (p) => { $('step').textContent = p.step + ' skipped'; },
    'session.opened': () => refresh(),
    'session.closed': () => { $('step').textContent = 'class finished'; showTeaching(false); },
    'student.identified': (p) => mood(p.student_id, 'engaged'),
    'student.left': (p) => mood(p.student_id, 'away'),
    'student.disengaged': (p) => mood(p.student_id, 'drifting'),
    'student.enrolled': () => loadEnrolled(),

    'vision.tracks': (p) => {
      const tracks = p.tracks || [];
      drawBoxes(tracks, p.width, p.height);
      const visible = new Set();
      for (const track of tracks) {
        if (!track.student_id) continue;
        visible.add(track.student_id);
        mood(track.student_id, track.attention >= engagedAt ? 'engaged' : 'drifting');
      }
      for (const id of chips.keys()) if (!visible.has(id)) mood(id, 'away');
    },
  };

  const connect = () => {
    const socket = new WebSocket(`ws://${location.host}/events`);
    socket.onopen = () => { $('status').textContent = 'live'; $('status').className = 'live'; };
    socket.onmessage = (message) => {
      const { event, payload } = JSON.parse(message.data);
      const handler = handlers[event];
      if (handler && payload) handler(payload);
    };
    socket.onclose = () => {
      $('status').textContent = 'reconnecting';
      $('status').className = '';
      setTimeout(connect, RECONNECT_MS);
    };
  };

  refresh();
  connect();
})();
