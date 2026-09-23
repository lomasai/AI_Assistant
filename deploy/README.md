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

One signal, and only one: **"I would like to ask something."** Answers are
spoken - a child saying why they think it is sunlight is the lesson - and
who raised the hand comes from the face beside it.

Not a sign language on purpose. A classroom full of signs to remember is a
class learning the robot instead of the subject.

**Hands** are what the pi profile uses, through `raised_hand`: skin above
and beside a face the camera has already found. No wheel, no model, a few
milliseconds, and it knows one thing - somebody's hand is up.

```bash
python tools/signs_check.py --mode pi --hands
```

It prints what reading hands costs and, separately, what finding the faces
cost - the robot does not pay that twice, because the lesson's own detector
has already found them. Dials, in order: `signs.fps` (3 is plenty),
`signs.hands.every` (one read in two), then `signs.hands.reader: none`.

**Not mediapipe on a Pi 4.** Its aarch64 wheel is compiled for a processor
with AES instructions, which this one does not have: importing it does not
raise, it aborts - `FATAL ERROR: compiled with aes enabled`, the whole
process, at boot. The reader asks a separate process whether it can run
before importing it, so choosing it on the wrong machine now degrades
instead of killing the robot. On a machine that can run it, it reads hand
*shapes*, and `signs.hands.actions` maps them (`Pointing_Up` ☝ and the
rest) to meanings.

Installing it also drags in its own OpenCV, which drags in numpy 2, which
is a different ABI from the one picamera2 was built against - the camera
then fails with `numpy.dtype size changed`. To undo that:

```bash
pip uninstall -y mediapipe opencv-contrib-python
pip install "numpy<2" "opencv-python-headless<5"
python tools/doctor.py --mode pi
```

**What happens when a hand goes up** (`signs.asking`): the robot finishes
the sentence it is saying and stops there, keeps what it had not said, says
"Yes, Ananya?", listens, answers, then picks the lesson up exactly where it
left off. `per_step` caps how many interruptions one idea may absorb;
`cooldown_seconds` stops the same confident child having every turn. A hand
nobody can name still gets a turn - the speaker chain works out who
afterwards, from the voice.

**Cards** are the other way to raise a hand, for a school with a printer:
markers on ~10 cm matte card, about 2 ms a frame, no model and no install,
and the card itself says who is holding it.

```bash
python tools/make_cards.py --mode pi --issue --spare 4
python tools/signs_check.py --mode pi              # measure the range here
```

Cut on the white, never into it: the quiet border is what makes a marker
readable. Answering a quiz by which edge is up is built and **off**
(`signs.cards.answering`) - it is for a class too big to hear one at a time.

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
