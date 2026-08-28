// Generated from config/hardware/gestures.yaml and servos.yaml.
// Do not edit. Run tests/test_hardware.py to regenerate.
#pragma once
#include <stdint.h>

#define GESTURE_COUNT 8
#define JOINT_COUNT 6

typedef struct { uint8_t channel; int16_t degrees; } lomas_joint_target_t;
typedef struct { uint16_t at_ms; uint8_t count; const lomas_joint_target_t *targets; } lomas_keyframe_t;
typedef struct { uint8_t id; const char *name; uint8_t frames; const lomas_keyframe_t *keyframes; } lomas_gesture_t;

static const lomas_joint_target_t celebrate_f0[] = {{4, 10}, {5, 10}, {2, 0}, {3, 0}};
static const lomas_joint_target_t celebrate_f1[] = {{4, 30}, {5, 30}, {2, 110}, {3, 110}};
static const lomas_joint_target_t celebrate_f2[] = {{4, 20}, {5, 20}, {2, 95}, {3, 95}};
static const lomas_joint_target_t celebrate_f3[] = {{4, 10}, {5, 10}, {2, 0}, {3, 0}};
static const lomas_keyframe_t celebrate_frames[] = {{0, 4, celebrate_f0}, {400, 4, celebrate_f1}, {1000, 4, celebrate_f2}, {1600, 4, celebrate_f3}};

static const lomas_joint_target_t namaste_f0[] = {{4, 10}, {5, 10}, {2, 0}, {3, 0}};
static const lomas_joint_target_t namaste_f1[] = {{4, 95}, {5, 95}, {2, 45}, {3, 45}};
static const lomas_joint_target_t namaste_f2[] = {{4, 100}, {5, 100}, {2, 45}, {3, 45}};
static const lomas_joint_target_t namaste_f3[] = {{4, 10}, {5, 10}, {2, 0}, {3, 0}};
static const lomas_keyframe_t namaste_frames[] = {{0, 4, namaste_f0}, {600, 4, namaste_f1}, {1400, 4, namaste_f2}, {2200, 4, namaste_f3}};

static const lomas_joint_target_t nod_f0[] = {{1, 0}};
static const lomas_joint_target_t nod_f1[] = {{1, 18}};
static const lomas_joint_target_t nod_f2[] = {{1, -8}};
static const lomas_joint_target_t nod_f3[] = {{1, 0}};
static const lomas_keyframe_t nod_frames[] = {{0, 1, nod_f0}, {300, 1, nod_f1}, {600, 1, nod_f2}, {900, 1, nod_f3}};

static const lomas_joint_target_t point_left_f0[] = {{4, 10}, {2, 0}};
static const lomas_joint_target_t point_left_f1[] = {{4, 15}, {2, 85}};
static const lomas_joint_target_t point_left_f2[] = {{4, 15}, {2, 85}};
static const lomas_joint_target_t point_left_f3[] = {{4, 10}, {2, 0}};
static const lomas_keyframe_t point_left_frames[] = {{0, 2, point_left_f0}, {500, 2, point_left_f1}, {1600, 2, point_left_f2}, {2100, 2, point_left_f3}};

static const lomas_joint_target_t point_right_f0[] = {{5, 10}, {3, 0}};
static const lomas_joint_target_t point_right_f1[] = {{5, 15}, {3, 85}};
static const lomas_joint_target_t point_right_f2[] = {{5, 15}, {3, 85}};
static const lomas_joint_target_t point_right_f3[] = {{5, 10}, {3, 0}};
static const lomas_keyframe_t point_right_frames[] = {{0, 2, point_right_f0}, {500, 2, point_right_f1}, {1600, 2, point_right_f2}, {2100, 2, point_right_f3}};

static const lomas_joint_target_t rest_f0[] = {{4, 10}, {5, 10}, {1, 0}, {0, 0}, {2, 0}, {3, 0}};
static const lomas_keyframe_t rest_frames[] = {{0, 6, rest_f0}};

static const lomas_joint_target_t shake_f0[] = {{0, 0}};
static const lomas_joint_target_t shake_f1[] = {{0, -22}};
static const lomas_joint_target_t shake_f2[] = {{0, 22}};
static const lomas_joint_target_t shake_f3[] = {{0, 0}};
static const lomas_keyframe_t shake_frames[] = {{0, 1, shake_f0}, {300, 1, shake_f1}, {600, 1, shake_f2}, {900, 1, shake_f3}};

static const lomas_joint_target_t thinking_f0[] = {{1, 0}, {0, 0}};
static const lomas_joint_target_t thinking_f1[] = {{1, -10}, {0, 14}};
static const lomas_joint_target_t thinking_f2[] = {{1, -10}, {0, 14}};
static const lomas_keyframe_t thinking_frames[] = {{0, 2, thinking_f0}, {500, 2, thinking_f1}, {1200, 2, thinking_f2}};

static const lomas_gesture_t LOMAS_GESTURES[GESTURE_COUNT] = {
    {7, "celebrate", 4, celebrate_frames},
    {1, "namaste", 4, namaste_frames},
    {2, "nod", 4, nod_frames},
    {4, "point_left", 4, point_left_frames},
    {5, "point_right", 4, point_right_frames},
    {0, "rest", 1, rest_frames},
    {3, "shake", 4, shake_frames},
    {6, "thinking", 3, thinking_frames}
};
