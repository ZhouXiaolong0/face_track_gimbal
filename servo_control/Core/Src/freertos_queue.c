#include "freertos_queue.h"

QueueHandle_t offsetQueue;

void Queue_Init(void)
{
    offsetQueue = xQueueCreate(10, sizeof(Offset_t));
}
