#include <vector>
#include <torch/extension.h>
#include "mkl.h"
#include <algorithm>
#include <cmath>

void fff_l1(torch::Tensor& IN, torch::Tensor& W1, torch::Tensor& W2, torch::Tensor& OUT, int batch_size, int hidden_dim, int layer_size, int depth) {

    // Based on
    // https://github.com/pbelcak/UltraFastBERT/blob/main/benchmark_cpu/fff.cpp
    int* current_nodes = (int*)mkl_calloc(batch_size, sizeof(int), 32);
    const float sqrt2 = std::sqrt(2);

    for (int d = 0; d < depth+1;++d) {
        float* mi = IN.data_ptr<float>();
        float* mo = OUT.data_ptr<float>();

        for (int i = 0; i < batch_size; ++i) {

            float logits = cblas_sdot(hidden_dim, mi, 1, W1.data_ptr<float>() + (current_nodes[i] * hidden_dim), 1);

            float val = logits * std::erfc(-logits / sqrt2) / 2; // GELU activation

            cblas_saxpy(layer_size, val, W2.data_ptr<float>() + (current_nodes[i] * layer_size), 1, mo, 1);

            current_nodes[i] = 2 * current_nodes[i] + 1 + (val > 0.f ? 1 : 0);

            mi += hidden_dim;
            mo += layer_size;
        }
    }

    mkl_free(current_nodes);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fff_l1", &fff_l1, "fff_l1");
}

