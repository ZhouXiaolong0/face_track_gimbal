#include <uart1_send_dma.h>

// DMA发送缓冲区（循环缓冲区）
uint8_t tx_buf[256];

// 缓冲区头尾索引，分别指向下一个写入位置和下一个发送位置
volatile uint16_t tx_head = 0;
volatile uint16_t tx_tail = 0;

// DMA发送状态标志：0表示空闲，1表示正在发送
volatile uint8_t tx_busy = 0;

// 当前DMA发送长度
uint16_t tx_len = 0;

// ===================== 初始化函数 =====================
void UART_DMA_Init(void) {
  // 初始化缓冲区状态
  tx_head = 0;
  tx_tail = 0;
  tx_busy = 0;
}

// ===================== 启动DMA发送 =====================
void UART_StartTx(void) {
  // 如果DMA空闲，并且缓冲区有数据需要发送
  if (!tx_busy && tx_head != tx_tail) {
    tx_busy = 1; // 标记DMA正在发送

    // 计算本次要发送的数据长度
    // 如果头指针在尾指针前面，说明数据在缓冲区尾部到缓冲区末尾之间
    tx_len =
        (tx_head > tx_tail) ? (tx_head - tx_tail) : (sizeof(tx_buf) - tx_tail);

    // 调用HAL库的DMA发送函数
    HAL_UART_Transmit_DMA(&huart1, &tx_buf[tx_tail], tx_len);
  }
}

// ===================== DMA发送完成回调 =====================
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
  // 判断是否是USART1触发的回调
  if (huart->Instance == USART1) {
    // 更新尾指针，标记已经发送的数据
    tx_tail += tx_len;
    if (tx_tail >= sizeof(tx_buf))
      tx_tail = 0; // 循环缓冲区回绕

    tx_busy = 0; // DMA发送完成，空闲

    UART_StartTx(); // 继续发送剩余数据
  }
}

// ===================== 重定向printf =====================
int _write(int file, char *ptr, int len) {
  for (int i = 0; i < len; i++) {
    // 计算下一个写入位置
    uint16_t next = (tx_head + 1) % sizeof(tx_buf);

    // 如果缓冲区满，阻塞等待
    while (next == tx_tail) {
    } // 缓冲区满，阻塞等待

    // 写入数据到缓冲区
    tx_buf[tx_head] = ptr[i];
    tx_head = next;
  }

  // 启动DMA发送
  UART_StartTx();

  return len; // 返回写入的字节数
}
