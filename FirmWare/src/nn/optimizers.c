#include "nn_types.h"
#include <math.h>

void sgd_update(float* weights,const float* grad_weights,int size,float learning_rate){
    if (!weights || !grad_weights) {
        return;
    }

    for (int i = 0; i < size; i++) {
        weights[i] -= learning_rate * grad_weights[i];
    }
}


void sgd_update_bias(float* bias,const float* grad_bias,int size,float learning_rate){
    if (!bias || !grad_bias) {
        return;
    }

    for (int i = 0; i < size; i++) {
        bias[i] -= learning_rate * grad_bias[i];
    }
}


// --- SGD with Momentum ---
// v = momentum * v + grad
// w -= lr * v
void sgd_momentum_update(float* weights, const float* grad_weights, float* velocity,
                         int size, float learning_rate, float momentum) {
    if (!weights || !grad_weights || !velocity) return;
    for (int i = 0; i < size; i++) {
        velocity[i] = momentum * velocity[i] + grad_weights[i];
        weights[i] -= learning_rate * velocity[i];
    }
}

// --- SGD with Momentum + L2 Weight Decay ---
// v = momentum * v + grad + weight_decay * w
// w -= lr * v
void sgd_momentum_update_l2(float* weights, const float* grad_weights, float* velocity,
                            int size, float learning_rate, float momentum, float weight_decay) {
    if (!weights || !grad_weights || !velocity) return;
    for (int i = 0; i < size; i++) {
        float grad_with_l2 = grad_weights[i] + weight_decay * weights[i];
        velocity[i] = momentum * velocity[i] + grad_with_l2;
        weights[i] -= learning_rate * velocity[i];
    }
}

void sgd_momentum_update_bias(float* bias, const float* grad_bias, float* velocity,
                              int size, float learning_rate, float momentum) {
    if (!bias || !grad_bias || !velocity) return;
    for (int i = 0; i < size; i++) {
        velocity[i] = momentum * velocity[i] + grad_bias[i];
        bias[i] -= learning_rate * velocity[i];
    }
}

// --- Learning Rate Schedule: warmup + exponential decay ---
float lr_schedule(float base_lr, int epoch) {
    if (epoch < 5) {
        // Linear warmup: 0.1*base_lr -> base_lr over 5 epochs
        return base_lr * (0.1f + 0.9f * ((float)epoch / 5.0f));
    } else {
        // Exponential decay: base_lr * 0.95^(epoch - 5)
        return base_lr * powf(0.95f, (float)(epoch - 5));
    }
}

// --- Cosine Annealing LR Schedule with Warm Restarts ---
float lr_cosine_annealing(float base_lr, int epoch, int T_max, float eta_min) {
    if (epoch < 3) {
        // Linear warmup over first 3 epochs
        return eta_min + (base_lr - eta_min) * ((float)epoch / 3.0f);
    }
    int effective_epoch = epoch - 3;
    int cycle_len = T_max;
    int cycle_pos = effective_epoch % cycle_len;
    float cosine_val = cosf((float)cycle_pos / (float)cycle_len * 3.14159265f);
    return eta_min + 0.5f * (base_lr - eta_min) * (1.0f + cosine_val);
}

// --- Adam Optimizer ---
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// w -= lr * m_hat / (sqrt(v_hat) + eps)
void adam_update(float* weights, const float* grad_weights, float* m, float* v,
                 int size, float learning_rate, float beta1, float beta2,
                 float epsilon, int timestep, float weight_decay) {
    if (!weights || !grad_weights || !m || !v) return;
    float bc1 = 1.0f - powf(beta1, (float)timestep);
    float bc2 = 1.0f - powf(beta2, (float)timestep);
    if (bc1 < 1e-10f) bc1 = 1e-10f;
    if (bc2 < 1e-10f) bc2 = 1e-10f;

    for (int i = 0; i < size; i++) {
        float g = grad_weights[i] + weight_decay * weights[i]; // AdamW-style decoupled weight decay
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

void adam_update_bias(float* bias, const float* grad_bias, float* m, float* v,
                      int size, float learning_rate, float beta1, float beta2,
                      float epsilon, int timestep) {
    if (!bias || !grad_bias || !m || !v) return;
    float bc1 = 1.0f - powf(beta1, (float)timestep);
    float bc2 = 1.0f - powf(beta2, (float)timestep);
    if (bc1 < 1e-10f) bc1 = 1e-10f;
    if (bc2 < 1e-10f) bc2 = 1e-10f;

    for (int i = 0; i < size; i++) {
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad_bias[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad_bias[i] * grad_bias[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        bias[i] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}
