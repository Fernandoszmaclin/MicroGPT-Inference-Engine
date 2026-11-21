/* C implementation of scaled dot-product causal attention. */

#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define MICROGPT_EXPORT __declspec(dllexport)
#else
#define MICROGPT_EXPORT
#endif

MICROGPT_EXPORT void microgpt_causal_attention(
    const float *q,
    const float *k,
    const float *v,
    float *out,
    double *weights,
    size_t heads,
    size_t tokens,
    size_t head_dim
) {
    const double scale = 1.0 / sqrt((double)head_dim);

    for (size_t head = 0; head < heads; ++head) {
        const size_t head_offset = head * tokens * head_dim;
        for (size_t row = 0; row < tokens; ++row) {
            const size_t query_offset = head_offset + row * head_dim;
            double max_score = 0.0;
            for (size_t col = 0; col <= row; ++col) {
                const size_t key_offset = head_offset + col * head_dim;
                double score = 0.0;
                for (size_t channel = 0; channel < head_dim; ++channel) {
                    score += (double)q[query_offset + channel] * k[key_offset + channel];
                }
                score *= scale;
                weights[col] = score;
                if (col == 0 || score > max_score) {
                    max_score = score;
                }
            }

            double total = 0.0;
            for (size_t col = 0; col <= row; ++col) {
                weights[col] = exp(weights[col] - max_score);
                total += weights[col];
            }
            for (size_t channel = 0; channel < head_dim; ++channel) {
                double value = 0.0;
                for (size_t col = 0; col <= row; ++col) {
                    const size_t value_offset = head_offset + col * head_dim;
                    value += weights[col] * v[value_offset + channel];
                }
                out[query_offset + channel] = (float)(value / total);
            }
        }
    }
}
