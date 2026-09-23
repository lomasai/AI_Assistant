# Putting the robot in a classroom

The robot needs no laptop. A phone or laptop is a remote control for the
teacher, and everything works without one.

## Start it on boot

```bash
sudo cp deploy/lomasai.service /etc/systemd/system/
sudo systemctl enable --now lomasai
journalctl -u lomasai -f
```

The file names a user and a directory; change those two lines if the robot
lives somewhere else.

## What it can do with nobody touching a screen

| | How |
|---|---|
| Begin a class | Somebody says **"start the class"** (`flow.start_phrases`) |
| Choose the lesson | The robot asks, a child answers out loud, and it writes that lesson |
| Ask questions | Just talk - no button, all the way through the lesson |
| Answer the quiz | Out loud, marked and answered by name |
| Meet a new child | The robot asks their name and enrols them (`enrolment.by_voice`) |
| Show its face | Its own screen, drawn directly - no browser on the Pi |
| File its traces | Commits and pushes `data/logs` when a class ends (`sync`) |

## How a class is taught

`flow.sequence` is `[attendance, greeting, topic, teach, quiz, wrapup]`.

`teach` is the lesson, the questions and the quiz interleaved: the robot
says one idea, then hands it back - either inviting doubts or asking one
child, by name, a question about what it just said. A question from the
class holds the next idea until it has been answered, so nothing is ever
spoken over a child.

Knobs, all under `flow.teach`:

| | |
|---|---|
| `check_every` | ideas between checks. 1 is after every one |
| `check_style` | `alternate` (a question, then an invitation), `doubts`, `question` |
| `name_a_child` | ask one child by name, taken round the roster |
| `gap_seconds` / `doubt_wait_seconds` | how long the room gets |
| `answer_hold_seconds` | the longest a child's question may hold the lesson |

A school that wants the old read-it-all-out lesson sets
`flow.sequence: [attendance, greeting, topic, lesson, interaction, quiz, wrapup]`
and nothing else changes.

## Traces, without typing git

```yaml
sync:
  enabled: true          # true in the pi profile
  paths: [data/logs]     # only these, never the whole working tree
  "on": [session_closed, shutdown]
  push: true             # false keeps the commits until there is a network
```

It commits and pushes those paths when a class ends and when the robot is
switched off. A push that fails leaves the commit behind, and the next class
sends it.

## What still wants a screen

- **Reports.** A teacher reads these after class, sitting down:
  `http://raspberrypi.local:8080/teacher/` -> Report.
- **Removing a child**, and correcting who the robot thinks spoke.
- **Volume.** The slider is in the bar at the top of the teacher's page, in
  every tab, with a mute beside it. On the Pi it moves the sound card's own
  mixer, so whatever else plays obeys it too, and the level is remembered
  across a reboot (`data/volume.json`). A school that never wants the robot
  above conversation volume sets `speech.tts.volume.max_level`.

## Switching the standalone parts off

A school that wants a teacher in charge of each of these sets:

```yaml
flow:
  start_phrases: []        # only the teacher's page starts a class
enrolment:
  by_voice: false          # only the teacher's page enrols a child
speech:
  audio:
    hands_free: false      # press to talk, as before
```
