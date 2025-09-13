#include "telemetry/system_metric.h"
#include <string>

SystemMetric::SystemMetric(DataLogger &logger) : logger_(logger) {
  logger_.log({"cpu_usage", "mem_usage", "gpu_usage"});
}

void SystemMetric::record(float cpu, float mem, float gpu) {
  logger_.log({std::to_string(cpu), std::to_string(mem), std::to_string(gpu)});
}
