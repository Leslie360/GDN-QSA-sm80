#include <torch/extension.h>
#include <vector>

#include "qsa_indexer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &qsa_indexer_forward, "QSA indexer forward",
          py::arg("q"), py::arg("raw_keys"),
          py::arg("cos_q"), py::arg("sin_q"),
          py::arg("cos_k"), py::arg("sin_k"),
          py::arg("r"), py::arg("block_topk"));
    m.def("qsa_indexer_topk_only", &qsa_indexer_topk_only_forward,
          "QSA indexer topK-only fused forward (no block_scores)",
          py::arg("q"), py::arg("raw_keys"),
          py::arg("cos_q"), py::arg("sin_q"),
          py::arg("cos_k"), py::arg("sin_k"),
          py::arg("r"), py::arg("block_topk"));
}
