#ifndef FREERTOS_QUEUE_H
#define FREERTOS_QUEUE_H
#include "FreeRTOS.h"
#include "queue.h"
#include <stdint.h>

typedef struct {
    int16_t dx;
    int16_t dy;
} Offset_t;

extern QueueHandle_t offsetQueue;

void Queue_Init(void);

#endif
