# Putting the robot in a classroom

The robot needs no laptop. A phone or laptop is a remote control for the
teacher, and everything works without one.

## Is this robot able to do what it is set up to do?

```bash
python tools/doctor.py --mode pi
```

One line per thing the profile asks for. It downloads nothing and changes
nothing, and it is the first thing to run after installing anything -
because installing one optional part quietly upgrades another. mediapipe
brings its own OpenCV, which brings numpy 2, which is a different ABI from
the one the system camera bindings were built against, and nothing says so
until a class is running and the camera is dark.

`BROKEN` means installed but unusable, and the cure is almost always:

```bash
pip install "numpy<2" "opencv-python-headless<5"
```

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

## Putting a hand up

☝ `Pointing_Up`, read by the model Google trained for it. Held up, it means
"I would like to ask something": the robot finishes the sentence it is
saying, stops, says "Yes, Ananya?", listens, answers, and picks the lesson
up where it left off - capped at `signs.asking.per_step` interruptions per
idea, with a cooldown per child so the same one cannot have every turn.

**The version matters on a Pi 4.** mediapipe 1.x is compiled for a
processor with AES instructions, which the BCM2711 does not have. It does
not fail politely - it aborts the process with an illegal instruction, at
boot. 0.10.x was built before that:

```bash
pip install "mediapipe==0.10.18"
python tools/fetch_models.py --hands
python tools/doctor.py --mode pi          # says which of the three is wrong
python tools/signs_check.py --mode pi --hands   # what it costs here
```

The doctor separates the three things that can be wrong: not installed,
installed but aborts on this processor, installed and runs but has no
model. The reader asks a separate process whether the import survives
before doing it itself, so the worst case is a robot with no hands rather
than a robot that will not start.

Dials, in order: `signs.fps` (3 is plenty - a hand stays up for seconds),
`signs.hands.every` (look at one read in two), then `signs.hands.reader:
none`.

**What is not here, and why.** A reader with no model - skin colour in an
arch beside the face - was written when 1.x aborted, and removed after four
rounds: in a room of wood and warm paint most of a wall is skin-coloured,
requiring movement did not help because a Pi camera with auto-exposure
changes every pixel every frame, and learning which patches are usually
skin did not either. It reported a raised hand at a person sitting still.
Colour is not a hand.

**Cards** are the other way, for a school with a printer: two milliseconds a
frame, never a wall, and the card says which child is holding it.

```yaml
signs: {enabled: true, cards: {reader: aruco}}
```

```bash
python tools/make_cards.py --mode pi --issue --spare 4
```

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

Because the robot commits its own traces, its working tree is usually dirty
and it is usually a commit or two behind. So on the robot, always:

```bash
git pull --rebase --autostash
```

A plain `git pull` answers "cannot pull with rebase: You have unstaged
changes", which is the trace file it is writing this second.

## What still wants a screen

- **Reports.** A teacher reads these after class, sitting down:
  `http://raspberrypi.local:8080/teacher/` -> Report.
- **Removing a child**, and correcting who the robot thinks spoke.
- **Volume** starts at whatever `speech.tts.volume.level` says, once, and
  after that `data/volume.json` wins - delete that file to start again.
  Careful with ALSA percentages: on a Pi's headphone jack, 80% is -17 dB,
  which is a seventh of the amplitude and inaudible under a fan.
- **Volume** can be changed from either screen. On the robot's own face
  there are three round buttons in the bottom right - mute, quieter, louder -
  and the up, down and M keys do the same; a bar shows the level and fades.
  The teacher's page has the slider in the bar at the top, in every tab. On the Pi it moves the sound card's own
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
