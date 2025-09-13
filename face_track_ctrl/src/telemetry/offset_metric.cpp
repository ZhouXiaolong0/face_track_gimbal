#include "telemetry/offset_metric.h"
#include <string>

OffsetMetric::OffsetMetric(DataLogger &logger) : logger_(logger) {
  // logger_.log({"raw_dx","raw_dy","filtered_dx","filtered_dy"});
}

void OffsetMetric::record(int raw_dx, int raw_dy, float filtered_dx,
                          float filtered_dy) {
  logger_.log({std::to_string(raw_dx), std::to_string(raw_dy),
               std::to_string(filtered_dx), std::to_string(filtered_dy)});
}
