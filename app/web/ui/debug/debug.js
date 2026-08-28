// The diagnostics overlay. It reads and does nothing else - there is no
// endpoint behind this page that changes anything, on purpose.
//
// It polls rather than subscribing: half of what it shows is host state that
// no event announces, and a panel that only updates when the robot speaks is
// useless exactly when the robot has stopped speaking.

(() => {
  const $ = (id) => document.getElementById(id);
  const BUSY = 60;
  const PEGGED = 85;
  const MISSING = '—';

  let every = 1000;

  const text = (value) => (value === null || value === undefined ? MISSING : String(value));

  const rows = (target, pairs) => {
    target.replaceChildren(...pairs.flatMap(([key, value]) => {
      const name = document.createElement('dt');
      name.textContent = key;
      const cell = document.createElement('dd');
      cell.textContent = text(value);
      return [name, cell];
    }));
  };

  const table = (target, headings, body) => {
    const head = document.createElement('tr');
    for (const label of headings) {
      const cell = document.createElement('th');
      cell.textContent = label;
      head.append(cell);
    }
    target.replaceChildren(head, ...body.map((values) => {
      const row = document.createElement('tr');
      values.forEach((value, index) => {
        const cell = document.createElement('td');
        cell.textContent = text(value);
        if (index) cell.className = 'num';
        row.append(cell);
      });
      return row;
    }));
  };

  // --- panels --------------------------------------------------------------

  const drawVision = (vision) => {
    if (!vision.running && vision.cycles === undefined) {
      rows($('vision'), [['running', 'no']]);
      return;
    }
    const capture = Object.values(vision.capture || {})[0] || {};
    rows($('vision'), [
      ['source', vision.source],
      ['detect fps', vision.detect_fps_target],
      ['cycles', vision.cycles],
      ['cycle ms', vision.cycle_ms],
      ['skipped', vision.skipped],
      ['errors', vision.errors],
      ['captured', capture.captured],
      ['dropped', capture.dropped],
      ['empty reads', capture.empty_reads],
      ['tracks', vision.tracks],
      ['embed calls', vision.embed_calls],
      ['embed ms', vision.embed_ms],
      ['matches', vision.matches],
      ['unknown', vision.unknowns],
      ['too small', vision.skipped_too_small],
      ['enrolled', vision.enrolled],
    ]);
  };

  const drawHost = (host) => {
    $('cores').replaceChildren(...host.cores.map((busy) => {
      const core = document.createElement('div');
      core.className = 'core' + (busy >= PEGGED ? ' pegged' : busy >= BUSY ? ' busy' : '');
      const bar = document.createElement('span');
      bar.style.height = Math.max(2, busy) + '%';
      const label = document.createElement('b');
      label.textContent = Math.round(busy) + '%';
      core.append(bar, label);
      return core;
    }));

    rows($('host'), [
      ['temperature', host.temperature_c === null ? MISSING : host.temperature_c + ' C'],
      ['load', host.load.length ? host.load.join('  ') : MISSING],
      ['memory free', host.memory.available_mb ? host.memory.available_mb + ' MB' : MISSING],
      ['memory total', host.memory.total_mb ? host.memory.total_mb + ' MB' : MISSING],
      ['uptime', host.uptime_s + ' s'],
    ]);

    $('throttle').replaceChildren(...host.throttled.map((flag) => {
      const chip = document.createElement('i');
      chip.textContent = flag.replace(/_/g, ' ');
      return chip;
    }));
  };

  const drawLatency = (latency) => {
    table($('latency'), ['pair', 'last', 'mean', 'worst', 'n'],
      Object.entries(latency).map(([name, gauge]) =>
        [name, gauge.last_ms, gauge.mean_ms, gauge.worst_ms, gauge.count]));
  };

  const drawTracks = (tracks) => {
    table($('tracks'), ['id', 'student', 'x', 'y', 'w', 'h', 'yaw', 'pitch', 'attention', 'seen for'],
      tracks.map((t) => [
        t.track_id, t.student_id || MISSING, t.x, t.y, t.w, t.h,
        Math.round(t.yaw), Math.round(t.pitch),
        t.attention.toFixed(2), t.seen_for.toFixed(1),
      ]));
  };

  const drawRates = (events) => {
    table($('rates'), ['event', 'per second', 'total'],
      Object.entries(events.rates).map(([name, rate]) => [name, rate, events.counts[name] || 0]));
  };

  const drawLlm = (llm) => {
    const priced = llm.cost > 0;
    $('llmTotals').replaceChildren();
    const totals = document.createElement('div');
    totals.innerHTML = '';
    totals.append(
      label('in ', llm.input_tokens),
      label(' out ', llm.output_tokens),
      label(priced ? ` cost ${llm.currency} ` : ' cost ', priced ? llm.cost : 'unpriced'),
    );
    $('llmTotals').append(totals);

    $('llm').replaceChildren(...llm.calls.map((call) => {
      const box = document.createElement('details');
      const head = document.createElement('summary');
      head.textContent = `${call.provider}/${call.model || 'default'}`;
      const meta = document.createElement('span');
      meta.textContent = `${call.ms} ms · ${call.input_tokens} in · ${call.output_tokens} out`;
      head.append(meta);

      const body = document.createElement('pre');
      body.textContent = call.prompt.map((m) => `[${m.role}]\n${m.content}`).join('\n\n')
        + '\n\n[answer]\n' + call.answer;
      box.append(head, body);
      return box;
    }));
  };

  const label = (name, value) => {
    const span = document.createElement('span');
    span.textContent = name;
    const strong = document.createElement('b');
    strong.textContent = text(value);
    span.append(strong);
    return span;
  };

  const drawEvents = (recent) => {
    $('events').replaceChildren(...[...recent].reverse().map((entry) => {
      const line = document.createElement('div');
      line.className = 'line';
      const when = document.createElement('time');
      when.textContent = new Date(entry.at * 1000).toLocaleTimeString();
      const name = document.createElement('b');
      name.textContent = entry.event;
      const body = document.createElement('span');
      body.textContent = JSON.stringify(entry.payload);
      line.append(when, name, body);
      return line;
    }));
  };

  const drawPlugins = (body) => {
    rows($('plugins'), Object.entries(body.chosen).map(([family, chosen]) => {
      const known = body.available[family] || [];
      const shown = Array.isArray(chosen) ? chosen.join(', ') : chosen;
      return [family, known.length > 1 ? `${shown}  (of ${known.length})` : shown];
    }));
  };

  // --- the loop ------------------------------------------------------------

  const tick = async () => {
    try {
      const body = await (await fetch('/api/debug/metrics')).json();
      $('mode').textContent = body.mode;
      $('mode').className = body.mode;
      $('uptime').textContent = body.uptime_s + ' s';
      $('poll').textContent = 'live';
      $('poll').className = 'live';

      drawVision(body.vision);
      drawHost(body.host);
      drawLatency(body.latency);
      drawTracks(body.tracks);
      drawRates(body.events);
      drawLlm(body.llm);
      drawEvents(body.events.recent);
    } catch (error) {
      $('poll').textContent = 'no answer';
      $('poll').className = '';
    }
    setTimeout(tick, every);
  };

  fetch('/api/debug/config').then((r) => r.json()).then((cfg) => {
    every = cfg.debug.poll_seconds * 1000;
  }).catch(() => {});

  fetch('/api/debug/plugins').then((r) => r.json()).then(drawPlugins).catch(() => {});
  tick();
})();
