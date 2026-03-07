#include <stdlib.h>
#include <math.h>
#include "nn.h"
#include "engine.h"

/* ---------------- Random utilities ---------------- */

/* Standard normal distribution using Box–Muller */
static float rand_normal() {
float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

}

/* ---------------- Parameter initialization ---------------- */

/* He initialization for ReLU networks */
static void init_weights(Tensor* W, int in_features) {
float scale = sqrtf(2.0f / in_features);

for (int i = 0; i < W->size; i++) {
    W->data[i] = rand_normal() * scale;
}

}

/* Bias initialized to zero */
static void init_bias(Tensor* b) {
    for (int i = 0; i < b->size; i++) {
        b->data[i] = 0.0f;
    }
}

/* ---------------- Linear Layer ---------------- */

Linearlayer* linear_create(int in_features, int out_features) {
    Linearlayer* layer = (Linearlayer*)malloc(sizeof(Linearlayer));

    layer->in_features = in_features;
    layer->out_features = out_features;

    /* Weight matrix: (in_features x out_features) */
    layer->W = tensor_create_matrix(in_features, out_features);

    /* Bias vector: (1 x out_features) */
    layer->b = tensor_create_matrix(1, out_features);

    /* Initialize parameters */
    init_weights(layer->W, in_features);
    init_bias(layer->b);

    return layer;

}

/* Forward pass: y = x @ W + b */
Tensor* linear_forward(Linearlayer* layer, Tensor* x) {
    Tensor* matmul_out = tensor_matmul(x, layer->W);
    Tensor* y = tensor_add(matmul_out, layer->b);

    tensor_release(matmul_out);
    return y;

}

/* Return parameters for optimizer */
Tensor** linear_params(Linearlayer* layer, int* n_params) {
    Tensor** params = (Tensor**)malloc(sizeof(Tensor*) * 2);

    params[0] = layer->W;
    params[1] = layer->b;

    *n_params = 2;
    return params;

}

/* Free layer */
void linear_free(Linearlayer* layer) {
    if (!layer) return;

    tensor_release(layer->W);
    tensor_release(layer->b);
    free(layer);

}

