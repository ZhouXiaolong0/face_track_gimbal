#include "uart2_receive.h"
#include "uart1_send_dma.h" // UART1 DMA发送相关函数
#include <freertos_queue.h>

extern UART_HandleTypeDef huart2;

// ------------------- 接收缓冲 -------------------
#define RX2_BUFFER_SIZE 64        // UART2 接收缓冲区大小
uint8_t rx2_buf[RX2_BUFFER_SIZE]; // 用于存放 UART2 接收到的字节
uint8_t rx2_index = 0;            // 当前写入缓冲区的位置索引

// ------------------- UART2 中断回调 -------------------
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
  // 判断触发回调的是不是 UART2
  if (huart->Instance == USART2) {
    uint8_t data = rx2_buf[rx2_index]; // 取出刚接收到的字节

    // 更新索引
    rx2_index++;                      // 移动到下一个缓冲位置
    if (rx2_index >= RX2_BUFFER_SIZE) // 防止越界
      rx2_index = 0;

    // ------------------- 检测整行结束 -------------------
    // 判断行结束符
    if (data == '\r' || data == '\n') {
      rx2_buf[rx2_index - 1] = 0; // 字符串结束
      char *comma = strchr((char *)rx2_buf, ',');

      if (comma) {
        *comma = 0;
        Offset_t offset;
        offset.dx = atoi((char *)rx2_buf);
        offset.dy = atoi(comma + 1);

        BaseType_t xHigherPriorityTaskWoken = pdFALSE;
        xQueueSendFromISR(offsetQueue, &offset, &xHigherPriorityTaskWoken);
        portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
      }

      rx2_index = 0;
    }

    // 再次调用 HAL_UART_Receive_IT() 启动下一个字节的中断接收
    HAL_UART_Receive_IT(&huart2, &rx2_buf[rx2_index], 1);
  }
}

// ------------------- 初始化接收 -------------------
void UART2_InitReceive(void) {
  rx2_index = 0; // 初始化索引
  // 启动 UART2 单字节中断接收
  HAL_UART_Receive_IT(&huart2, &rx2_buf[rx2_index], 1); // 启动 UART2 单字节接收
}
