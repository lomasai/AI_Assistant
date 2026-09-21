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
  // How long the End button stays out of reach after a class starts.
  const STARTING_MS = 3000;

  fetch('/api/topics').then((r) => r.json()).then((body) => {
    // First and default: nothing chosen, so the robot asks the class and
    // teaches whatever they say. Picking a lesson here is the override.
    const ask = document.createElement('option');
    ask.value = '';
    ask.textContent = 'Ask the class what to learn';

    $('topic').replaceChildren(ask, ...body.topics.map((lesson) => {
      const option = document.createElement('option');
      option.value = lesson.id;
      option.textContent = `${lesson.title} (${lesson.segments} parts)`;
      return option;
    }));
  });

  const showTeaching = (on) => {
    teaching = on;
    $('start').textContent = on ? 'End class' : 'Start class';
    // A class takes a moment to open, and a teacher pressing the button
    // again in that moment meant to start it, not to end it.
    if (on) {
      $('start').disabled = true;
      setTimeout(() => { $('start').disabled = false; }, STARTING_MS);
    }
    $('start').classList.toggle('teaching', on);
    $('topic').disabled = on;
    $('anyTopic').disabled = on;
    $('listenTopic').disabled = on;
  };

  $('start').onclick = async () => {
    // The same button starts and ends, and a second click while a class was
    // starting ended it before the greeting. Ending is now asked about.
    if (teaching && !confirm('End the class now? The report is kept.')) return;

    // Typed wins over chosen: the list is what has been written, the box is
    // what a child just asked for.
    const asked = $('anyTopic').value.trim();
    if (!teaching && asked) $('step').textContent = `writing a lesson on ${asked}…`;
    const body = teaching
      ? await post('/session/stop')
      : await post('/session/start', { topic: asked || $('topic').value });
    if (body.error) { $('step').textContent = body.error; return; }
    showTeaching(!teaching);
  };

  // The topic, spoken. Nothing is attributed and nothing is asked of the
  // tutor: these words are the subject of the lesson, not a question.
  $('listenTopic').onclick = async (event) => {
    const button = event.target;
    button.disabled = true;
    const was = button.textContent;
    button.textContent = 'Listening…';
    try {
      const heard = await post('/listen', { as_topic: true });
      if (heard.text) $('anyTopic').value = heard.text;
      else $('step').textContent = heard.error || heard.reason || 'nothing was said';
    } finally {
      button.disabled = false;
      button.textContent = was;
    }
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

  // --- hearing a child -----------------------------------------------------
  // Press to talk. A tapped name still wins, but the robot works the speaker
  // out for itself when nobody has tapped - from the camera, from who spoke a
  // moment ago, or from "I am Akshay" at the front of the question. It says
  // which, so a wrong guess can be corrected by tapping.

  // What the class is waiting for. A quiz question on the board makes the
  // box an answer box; the rest of the time it asks Lomas something.
  let answering = false;

  const said = {
    tapped: 'you tapped the name',
    single_face: 'the only face in view',
    recent: 'still the same speaker',
    spoken_name: 'they said their name',
    mouth_motion: 'their mouth moved',
    unknown: 'nobody could be worked out - tap a name',
    caller: '',
  };

  const hear = async (button) => {

    button.classList.add('hearing');
    button.disabled = true;
    const was = button.textContent;
    button.textContent = 'Listening…';
    try {
      const heard = await post('/listen', {
        student_id: speaker ? speaker.id : '', student_name: speaker ? speaker.name : '',
        as_answer: answering,
      });
      if (heard.error) $('step').textContent = heard.error;
      else if (!heard.text) $('step').textContent = heard.reason || 'nothing was said';
      else {
        $('saidText').value = heard.text;
        const why = said[heard.how] === undefined ? heard.how : said[heard.how];
        $('who').textContent = heard.student_name
          ? `${heard.student_name} spoke — ${why}`
          : why || 'heard';
      }
    } finally {
      button.classList.remove('hearing');
      button.disabled = false;
      button.textContent = was;
    }
  };

  $('listen').onclick = (event) => hear(event.target);

  $('say').onsubmit = async (event) => {
    event.preventDefault();
    const field = $('saidText');
    const text = field.value.trim();
    if (!text) return;
    field.value = '';
    const body = answering
      ? await post('/answer', { response: text })
      : await post('/ask', {
        text,
        student_id: speaker ? speaker.id : '',
        student_name: speaker ? speaker.name : '',
      });
    if (body.error) $('step').textContent = body.error;
  };

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
      const drop = document.createElement('button');
      drop.className = 'drop';
      drop.textContent = '×';
      drop.title = `remove ${student.name} from this class`;
      drop.onclick = (event) => {
        event.stopPropagation();
        if (teaching || !confirm(`Remove ${student.name} from the class?`)) return;
        fetch(`/api/students/${student.id}`, { method: 'DELETE' })
          .then(() => { chips.clear(); refresh(); });
      };
      row.append(drop);
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
      for (const id of ['listen', 'listenTopic']) {
        $(id).disabled = deaf;
        $(id).title = deaf ? 'no microphone on this machine' : 'press and speak';
      }
    });

  const asking = (on) => {
    answering = on;
    $('send').textContent = on ? 'Record answer' : 'Ask Lomas';
    $('saidText').placeholder = on
      ? 'what the child answered'
      : 'ask Lomas something on behalf of the class';
  };

  const handlers = {
    'quiz.posed': () => asking(true),
    'quiz.recorded': () => asking(false),
    'step.exited': (p) => { if (p.step === 'quiz') asking(false); },
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
