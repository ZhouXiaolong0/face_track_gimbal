#include "telemetry/system_info.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// ---------------- CPU ----------------
float SystemInfo::getCpuUsage() {
  static long long last_user = 0, last_nice = 0, last_sys = 0, last_idle = 0;
  float percent = 0.0f;

  std::ifstream file_stat("/proc/stat");
  std::string line;
  std::getline(file_stat, line);
  std::istringstream ss(line);

  std::string cpu;
  long long user, nice, sys, idle, iowait, irq, softirq, steal;
  ss >> cpu >> user >> nice >> sys >> idle >> iowait >> irq >> softirq >> steal;

  if (last_user != 0 || last_nice != 0 || last_sys != 0 || last_idle != 0) {
    long long diff_user = user - last_user;
    long long diff_nice = nice - last_nice;
    long long diff_sys = sys - last_sys;
    long long diff_idle = idle - last_idle;

    long long total = diff_user + diff_nice + diff_sys + diff_idle;
    if (total > 0) {
      percent = (float)(diff_user + diff_nice + diff_sys) * 100.0f / total;
    }
  }

  last_user = user;
  last_nice = nice;
  last_sys = sys;
  last_idle = idle;

  return percent;
}

// ---------------- 内存 ----------------
float SystemInfo::getMemUsage() {
  std::ifstream file_mem("/proc/meminfo");
  std::string key;
  long total = 0, free = 0, buffers = 0, cached = 0;

  while (file_mem >> key) {
    if (key == "MemTotal:")
      file_mem >> total;
    else if (key == "MemFree:")
      file_mem >> free;
    else if (key == "Buffers:")
      file_mem >> buffers;
    else if (key == "Cached:")
      file_mem >> cached;
    else {
      std::string dummy;
      std::getline(file_mem, dummy);
    }
  }

  long used = total - free - buffers - cached;
  return total > 0 ? (float)used * 100.0f / total : 0.0f;
}

// ---------------- GPU ----------------
float SystemInfo::getGpuUsage() {
  const char *gpu_load_path = "/sys/devices/gpu.0/load";
  std::ifstream gpu_file(gpu_load_path);
  if (!gpu_file.is_open()) {
    std::cerr << "Failed to open GPU load file: " << gpu_load_path << std::endl;
    return 0.0f;
  }

  int gpu_raw = 0;
  gpu_file >> gpu_raw;
  gpu_file.close();

  // Jetson load 文件单位为 1/10%，所以除以 10
  float gpu_usage = static_cast<float>(gpu_raw) / 10.0f;

  // 保证范围 0~100
  if (gpu_usage < 0.0f)
    gpu_usage = 0.0f;
  if (gpu_usage > 100.0f)
    gpu_usage = 100.0f;

  return gpu_usage;
}

// ---------------- Collect ----------------
SystemInfoData SystemInfo::collect() {
  return {getCpuUsage(), getMemUsage(), getGpuUsage()};
}
