#ifndef INC_SERVO_CONTROL_H_
#define INC_SERVO_CONTROL_H_

#include "main.h"
#include "stm32f1xx_hal.h"

// 中立位置
#define SERVO1_CENTER 1250
#define SERVO2_CENTER 1500

// ================= 舵机映射参数 =================
// 在实际标定中测出来的值：
//   - 当目标在画面最左边时，对应的 dx 和 PWM 脉宽
//   - 当目标在画面最中间时，对应的 dx 和 PWM 脉宽
//   - 当目标在画面最右边时，对应的 dx 和 PWM 脉宽
#define DX_MIN -250 // 左边界时的 dx（像素）
#define DX_MAX 240  // 右边界时的 dx（像素）
#define DX_CENTER 0 // 中心 dx（一般就是 0）

#define SERVO_MIN 1000    // 舵机左边界 PWM 脉宽
#define SERVO_MAX 2000    // 舵机右边界 PWM 脉宽
#define SERVO_CENTER 1500 // 舵机中心 PWM 脉宽（中点校准）

// 方向开关：0 = 正常方向（dx 增大 -> pulse 增大）
//           1 = 反转方向（dx 增大 -> pulse 减小）
#define DX_REVERSE 1

void Servo_Init(void);
void Servo_SelfTest(void);

uint16_t DX_To_Pulse(int dx);
void ControlServoWithDX(int dx);

#endif /* INC_SERVO_CONTROL_H_ */
