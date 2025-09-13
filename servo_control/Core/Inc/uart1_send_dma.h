#ifndef __UART_DMA_H
#define __UART_DMA_H

#include "main.h"
#include "usart.h" // huart2 在这里声明的

void UART_DMA_Init(void);                 // 初始化（可选）
void UART_StartTx(void);                  // 内部使用
int _write(int file, char *ptr, int len); // printf 重定向

#endif
