#include "app.h"

int main() {
  try {
    App app;
    app.run();

  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
