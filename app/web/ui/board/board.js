// The classroom screen. Same event stream as the face, different job: the
// face shows who the robot is, this shows what the class is on.
//
// One idea at a time, and no child's name on it except the one being spoken
// to right now.

(() => {
  const title = document.getElementById('title');
  const progress = document.getElementById('progress');
  const say = document.getElementById('say');
  const options = document.getElementById('options');
  const who = document.getElementById('who');
  const count = document.getElementById('count');

  const RECONNECT_MS = 1500;
  const present = new Set();

  const write = (text, asQuestion) => {
    say.textContent = text;
    say.classList.toggle('question', Boolean(asQuestion));
    say.classList.add('on');
  };

  const dots = (total, index) => {
    if (progress.children.length !== total) {
      progress.replaceChildren(...Array.from({ length: total }, () => document.createElement('i')));
    }
    [...progress.children].forEach((dot, at) => {
      dot.className = at < index ? 'done' : at === index ? 'now' : '';
    });
  };

  const handlers = {
    'session.opened': (p) => {
      title.textContent = p.topic;
      present.clear();
      count.textContent = '';
      write('', false);
      say.classList.remove('on');
    },

    'session.closed': () => {
      progress.replaceChildren();
      options.hidden = true;
    },

    'attendance.marked': (p) => {
      present.add(p.student_id);
      count.textContent = `${present.size} present`;
    },

    'lesson.segment': (p) => {
      title.textContent = p.lesson_id.replace(/-/g, ' ');
      dots(p.total, p.index);
      options.hidden = true;
      // display when the pack gives one, otherwise what is being said. A
      // segment written for the board reads differently from one read aloud.
      write(p.display || p.say, false);
    },

    'question.asked': (p) => write(p.text, true),
    'question.answered': (p) => write(p.answer, false),

    'quiz.posed': (p) => {
      write(p.text, true);
      options.replaceChildren(...(p.options || []).map((text, index) => {
        const item = document.createElement('li');
        const key = document.createElement('b');
        key.textContent = index + 1;
        item.append(key, document.createTextNode(text));
        return item;
      }));
      options.hidden = !(p.options || []).length;
    },

    'robot.say': (p) => { who.textContent = p.student_name || ''; },
  };

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
