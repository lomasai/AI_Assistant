// The contract between the Pi and this board.
//
// This file mirrors packages/lomas_hal/protocol.py. A test parses both and
// fails if any value has drifted, so change them together or not at all.
//
// Frame layout, little-endian throughout:
//
//     A5 5A | VER | SEQ | CMD | LEN | PAYLOAD (LEN bytes) | CRC8
//
// CRC8 covers VER through the end of PAYLOAD. Polynomial 0x07, zero init,
// no reflection - eight lines, no table, safe in an interrupt.

#pragma once
#include <stdint.h>

#define LOMAS_PROTOCOL_VERSION 1

#define LOMAS_START_A 0xA5
#define LOMAS_START_B 0x5A
#define LOMAS_HEADER_SIZE 4
#define LOMAS_FRAME_OVERHEAD 7
#define LOMAS_MAX_PAYLOAD 255

#define LOMAS_CRC_POLYNOMIAL 0x07
#define LOMAS_CRC_INIT 0x00

// Angles and distances travel as integers. A tenth of a degree is finer than
// any servo on this robot.
#define LOMAS_DECIDEGREES 10

// Pi to ESP32. Intent only - never a servo angle.
typedef enum {
    LOMAS_CMD_PING = 0x01,
    LOMAS_CMD_SET_LIMITS = 0x02,
    LOMAS_CMD_SET_TELEMETRY_HZ = 0x03,

    LOMAS_CMD_GESTURE = 0x10,
    LOMAS_CMD_LOOK_AT = 0x11,
    LOMAS_CMD_MOVE = 0x12,
    LOMAS_CMD_STOP_MOTION = 0x13,

    LOMAS_CMD_HALT = 0x20,
    LOMAS_CMD_CLEAR_HALT = 0x21,
} lomas_command_t;

// ESP32 to Pi.
typedef enum {
    LOMAS_REPLY_ACK = 0x80,
    LOMAS_REPLY_NACK = 0x81,
    LOMAS_REPLY_TELEMETRY = 0x90,
    LOMAS_REPLY_EVENT = 0x91,
} lomas_reply_t;

typedef enum {
    LOMAS_ERR_OK = 0x00,
    LOMAS_ERR_BAD_CRC = 0x01,
    LOMAS_ERR_BAD_LENGTH = 0x02,
    LOMAS_ERR_UNKNOWN_COMMAND = 0x03,
    LOMAS_ERR_UNKNOWN_GESTURE = 0x04,
    LOMAS_ERR_OUT_OF_RANGE = 0x05,
    LOMAS_ERR_BUSY = 0x06,
    LOMAS_ERR_HALTED = 0x07,
    LOMAS_ERR_NOT_CALIBRATED = 0x08,
} lomas_error_t;

// Telemetry status bits. Every one is a decision this board has already
// taken; the Pi reads them and reports.
typedef enum {
    LOMAS_FLAG_HALTED = 1 << 0,
    LOMAS_FLAG_ESTOP = 1 << 1,
    LOMAS_FLAG_CLIFF = 1 << 2,
    LOMAS_FLAG_TILT = 1 << 3,
    LOMAS_FLAG_OVERCURRENT = 1 << 4,
    LOMAS_FLAG_STALLED = 1 << 5,
    LOMAS_FLAG_CALIBRATED = 1 << 6,
} lomas_flag_t;

// Sent the moment it happens, not at the next telemetry tick. A cliff
// detected 40 ms ago is old news.
typedef enum {
    LOMAS_EVENT_ESTOP_PRESSED = 0x01,
    LOMAS_EVENT_ESTOP_RELEASED = 0x02,
    LOMAS_EVENT_CLIFF_DETECTED = 0x03,
    LOMAS_EVENT_TILT_DETECTED = 0x04,
    LOMAS_EVENT_OVERCURRENT = 0x05,
    LOMAS_EVENT_STALL = 0x06,
    LOMAS_EVENT_GESTURE_DONE = 0x07,
} lomas_event_t;

#define LOMAS_ULTRASONIC_COUNT 6
#define LOMAS_CLIFF_COUNT 4
#define LOMAS_TELEMETRY_SIZE 35

// Must pack to exactly LOMAS_TELEMETRY_SIZE bytes. The Python side unpacks
// "<B6H4H3hhHI" and will refuse anything else.
typedef struct __attribute__((packed)) {
    uint8_t flags;
    uint16_t ultrasonic_mm[LOMAS_ULTRASONIC_COUNT];
    uint16_t cliff_mm[LOMAS_CLIFF_COUNT];
    int16_t pitch_ddeg;
    int16_t roll_ddeg;
    int16_t yaw_ddeg;
    int16_t current_ma;
    uint16_t battery_mv;
    uint32_t uptime_ms;
} lomas_telemetry_t;

// Uploaded once at connect, then owned by this board. The Pi never compares
// against these numbers - that comparison has to happen in real time.
typedef struct __attribute__((packed)) {
    uint16_t cliff_mm;
    uint16_t obstacle_mm;
    uint16_t current_ma;
    int16_t tilt_ddeg;
    uint16_t stall_ms;
} lomas_limits_t;

static inline uint8_t lomas_crc8(const uint8_t *data, uint16_t length) {
    uint8_t crc = LOMAS_CRC_INIT;
    for (uint16_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (uint8_t bit = 0; bit < 8; bit++) {
            crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ LOMAS_CRC_POLYNOMIAL) : (uint8_t)(crc << 1);
        }
    }
    return crc;
}
