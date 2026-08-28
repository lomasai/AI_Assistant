// LomasAI ESP32-S3 firmware, stub.
//
// What this board is for, in one sentence: everything that must happen inside
// a few milliseconds. The Pi runs Python and can pause for garbage collection
// at any moment, so nothing that a pause would ruin is allowed to live there.
//
// This board owns:
//   * servo PWM and keyframe interpolation
//   * cliff, tilt, over-current and e-stop cut-outs
//   * the drive motors and their encoders
//
// The Pi owns: what to say, who to look at, which gesture. Intent, never
// execution.
//
// The four TODOs below are the whole job. Everything else - framing, the
// command set, the gesture table - is generated or given.

#include <Arduino.h>
#include "protocol.h"
#include "gestures_generated.h"

// --- what the Pi uploaded, and what we tell it -----------------------------

static lomas_limits_t limits = {120, 350, 4500, 220, 400};  // replaced at connect
static lomas_telemetry_t telemetry;
static uint8_t telemetry_hz = 20;
static bool halted = false;

// --- framing ---------------------------------------------------------------

static uint8_t rx[LOMAS_FRAME_OVERHEAD + LOMAS_MAX_PAYLOAD];
static uint16_t rx_len = 0;

static void send_frame(uint8_t kind, uint8_t seq, const uint8_t *payload, uint8_t length) {
    uint8_t body[LOMAS_HEADER_SIZE + LOMAS_MAX_PAYLOAD];
    body[0] = LOMAS_PROTOCOL_VERSION;
    body[1] = seq;
    body[2] = kind;
    body[3] = length;
    for (uint8_t i = 0; i < length; i++) body[LOMAS_HEADER_SIZE + i] = payload[i];

    Serial.write(LOMAS_START_A);
    Serial.write(LOMAS_START_B);
    Serial.write(body, LOMAS_HEADER_SIZE + length);
    Serial.write(lomas_crc8(body, LOMAS_HEADER_SIZE + length));
}

static void ack(uint8_t seq, lomas_error_t code) {
    uint8_t payload[2] = {seq, (uint8_t)code};
    send_frame(code == LOMAS_ERR_OK ? LOMAS_REPLY_ACK : LOMAS_REPLY_NACK, seq, payload, 2);
}

// Sent the instant it happens. Do not wait for the telemetry tick.
static void report(lomas_event_t event, uint16_t detail) {
    uint8_t payload[3] = {(uint8_t)event, (uint8_t)(detail & 0xFF), (uint8_t)(detail >> 8)};
    send_frame(LOMAS_REPLY_EVENT, 0, payload, 3);
}

// --- the safety loop -------------------------------------------------------
//
// This runs on its own core at the cliff sensors' poll rate and is the one
// piece of this system that has to be right. Nothing may block it: no serial
// write that could stall, no delay(), no allocation.

static void safety_task(void *) {
    for (;;) {
        // TODO 1: read the four VL53L0X, the MPU6050 and the ACS712, and the
        // e-stop pin. Cut power to the servo rail and the BTS7960B enable
        // lines the moment any of them is past `limits`. Cut first, report
        // second - report() writes to serial and serial can wait.
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// --- motion ----------------------------------------------------------------

static void start_gesture(uint8_t id, uint8_t speed) {
    for (uint8_t i = 0; i < GESTURE_COUNT; i++) {
        if (LOMAS_GESTURES[i].id != id) continue;
        // TODO 2: interpolate LOMAS_GESTURES[i].keyframes onto the PCA9685 at
        // 50 Hz, scaled by `speed`. Clamp every channel to the limits in
        // gestures_generated.h; the Pi clamps too, and both are on purpose.
        // Call report(LOMAS_EVENT_GESTURE_DONE, id) at the end.
        return;
    }
    report(LOMAS_EVENT_GESTURE_DONE, id);
}

static void handle(uint8_t seq, uint8_t command, const uint8_t *payload, uint8_t length) {
    // A halted board refuses movement rather than queueing it. A robot that
    // catches up on three gestures the moment it is cleared hurts someone.
    if (halted && command != LOMAS_CMD_CLEAR_HALT && command != LOMAS_CMD_PING) {
        ack(seq, LOMAS_ERR_HALTED);
        return;
    }

    switch (command) {
        case LOMAS_CMD_PING:
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_SET_LIMITS:
            if (length != sizeof(lomas_limits_t)) { ack(seq, LOMAS_ERR_BAD_LENGTH); break; }
            memcpy(&limits, payload, sizeof(limits));
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_SET_TELEMETRY_HZ:
            if (length != 1) { ack(seq, LOMAS_ERR_BAD_LENGTH); break; }
            telemetry_hz = payload[0];
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_GESTURE:
            if (length != 2) { ack(seq, LOMAS_ERR_BAD_LENGTH); break; }
            start_gesture(payload[0], payload[1]);
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_LOOK_AT:
            // TODO 3: payload is int16 yaw, int16 pitch in decidegrees. Move
            // the neck servos there, clamped.
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_MOVE:
            // TODO 4: payload is int16 linear, int16 angular in thousandths
            // of full scale. Drive the BTS7960B pair, and stop if the
            // ultrasonic ring says an obstacle is inside limits.obstacle_mm.
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_STOP_MOTION:
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_HALT:
            halted = true;
            // Cut first. The ack is a courtesy; the cut is the contract.
            ack(seq, LOMAS_ERR_OK);
            break;

        case LOMAS_CMD_CLEAR_HALT:
            halted = false;
            ack(seq, LOMAS_ERR_OK);
            break;

        default:
            ack(seq, LOMAS_ERR_UNKNOWN_COMMAND);
    }
}

// Resynchronising, because a serial line drops bytes and one bad frame must
// cost one frame rather than every frame after it.
static void pump_serial() {
    while (Serial.available()) {
        uint8_t byte = Serial.read();

        if (rx_len == 0 && byte != LOMAS_START_A) continue;
        if (rx_len == 1 && byte != LOMAS_START_B) { rx_len = 0; continue; }
        rx[rx_len++] = byte;

        if (rx_len < LOMAS_FRAME_OVERHEAD) continue;

        uint8_t length = rx[5];
        uint16_t total = LOMAS_FRAME_OVERHEAD + length;
        if (rx_len < total) continue;

        const uint8_t *body = &rx[2];
        if (lomas_crc8(body, LOMAS_HEADER_SIZE + length) == rx[total - 1] &&
            body[0] == LOMAS_PROTOCOL_VERSION) {
            handle(body[1], body[2], &body[LOMAS_HEADER_SIZE], length);
        } else {
            ack(body[1], LOMAS_ERR_BAD_CRC);
        }
        rx_len = 0;
    }
}

// --- entry points ----------------------------------------------------------

void setup() {
    Serial.begin(921600);
    xTaskCreatePinnedToCore(safety_task, "safety", 4096, nullptr, configMAX_PRIORITIES - 1,
                            nullptr, 1);
}

void loop() {
    pump_serial();

    static uint32_t next_telemetry = 0;
    uint32_t now = millis();
    if (telemetry_hz && now >= next_telemetry) {
        next_telemetry = now + (1000 / telemetry_hz);
        telemetry.flags = halted ? LOMAS_FLAG_HALTED : 0;
        telemetry.uptime_ms = now;
        send_frame(LOMAS_REPLY_TELEMETRY, 0, (const uint8_t *)&telemetry,
                   sizeof(lomas_telemetry_t));
    }
}
