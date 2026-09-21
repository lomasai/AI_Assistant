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
| Ask questions | Just talk - no button, during the interaction |
| Answer the quiz | Out loud, marked and answered by name |
| Meet a new child | The robot asks their name and enrols them (`enrolment.by_voice`) |
| Show its face | Its own screen, drawn directly - no browser on the Pi |

## What still wants a screen

- **Reports.** A teacher reads these after class, sitting down:
  `http://raspberrypi.local:8080/teacher/` -> Report.
- **Removing a child**, and correcting who the robot thinks spoke.

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
