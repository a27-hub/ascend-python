#include <torch/extension.h>
#include "../utils.h"
#include "../ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &(ascendc_path::run_add_custom), "x + y");
}