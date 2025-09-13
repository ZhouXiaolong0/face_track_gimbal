#include "servo_control.h"
#include "stm32f1xx_hal.h"
#include "tim.h"

//  channel 1 是俯仰的舵机，数值越小，越俯，数值越大越仰， 800 是竖直，低于800
//  就太里了 channel 1 min 800 max 1700
//	channel 2 是底座的舵机 channel 2 min 500 max 2500
void Servo_Init(void) {
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, SERVO1_CENTER);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, SERVO2_CENTER);
  HAL_Delay(500);
}

void Servo_SelfTest(void) {
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1000);
  HAL_Delay(300);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, 1500);
  HAL_Delay(300);

  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 1200);
  HAL_Delay(300);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 1800);
  HAL_Delay(300);

  // 回到中立
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, SERVO1_CENTER);
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, SERVO2_CENTER);

  //  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 1000);
  //  HAL_Delay(300);
  //  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, 2000);
  //  HAL_Delay(300);
}

/**
 * @brief 将 dx 偏移量映射到 PWM 脉宽 (1000 ~ 2000)
 * @param dx    偏移量 (-240 ~ 240)
 * @return      映射后的 PWM 脉宽 (1000 ~ 2000)
 */
uint16_t DX_To_Pulse(int dx) {
  // 防止除以 0（万一 DX_MIN == DX_MAX），返回中点
  if (DX_MAX == DX_MIN) {
    return (uint16_t)((SERVO_MIN + SERVO_MAX) / 2);
  }

  // 限幅
  if (dx < DX_MIN)
    dx = DX_MIN;
  if (dx > DX_MAX)
    dx = DX_MAX;

  // 先把 dx 映射到 0..1 的 ratio
  float ratio = (float)(dx - DX_MIN) / (float)(DX_MAX - DX_MIN); // 0..1

#if DX_REVERSE
  // 反转方向：ratio -> 1 - ratio
  ratio = 1.0f - ratio;
#endif

  // 再把 ratio 映射到 SERVO_MIN..SERVO_MAX
  float pulse_f = (float)SERVO_MIN + ratio * (float)(SERVO_MAX - SERVO_MIN);

  // 最后做一次保护性限幅并四舍五入
  if (pulse_f < SERVO_MIN)
    pulse_f = SERVO_MIN;
  if (pulse_f > SERVO_MAX)
    pulse_f = SERVO_MAX;

  return (uint16_t)(pulse_f + 0.5f);
}

/**
 * @brief 使用 dx 偏移量控制底座舵机的 PWM
 * @param dx    偏移量 (-240 ~ 240)
 */
void ControlServoWithDX(int dx) {
  static uint16_t last_pulse = 1500; // 当前 PWM 脉宽
  uint16_t target_pulse = DX_To_Pulse(dx);

  // ---- 加一个死区，避免小抖动 ----
  //  if (abs(target_pulse - last_pulse) < 5) {
  //    return; // 差值太小就不动
  //  }

  // ---- 比例控制步长 ----
  int error = target_pulse - last_pulse;
  int step = error / 5; // 比例因子，偏差大时走大步
  if (step == 0) {
    step = (error > 0) ? 1 : -1; // 确保至少能动
  }

  // ---- 只走一步，不阻塞 ----
  last_pulse += step;
  __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, last_pulse);
}
