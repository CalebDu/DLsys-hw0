#include <algorithm>
#include <cmath>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static void mm(const float *a, const float *b, float *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int ki = 0; ki < k; ki++) {
            for (int j = 0; j < n; j++) {
                c[i * n + j] += a[i * k + ki] * b[ki * n + j];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    auto z = new float[batch * k];
    auto Xt = new float[batch * n];
    auto dw = new float[n * k];
    // int bat = 0;
    for (int idx = 0; idx < m; idx += batch) {
        // std::fill(Iy, Iy + batch * k, 0.0f);
        std::fill(z, z + batch * k, 0.0f);
        std::fill(Xt, Xt + batch * n, 0.0f);
        std::fill(dw, dw + n * k, 0.0f);

        auto img = X + idx * n;
        auto label = y + idx;

        // X dot theta
        mm(img, theta, z, batch, k, n);

        for (int i = 0; i < batch; i++) {
            float accum = 0.0f;

            for (int j = 0; j < k; j++) {
                z[i * k + j] = exp(z[i * k + j]);
                accum += z[i * k + j];
            }
            // normalize
            for (int j = 0; j < k; j++) {
                z[i * k + j] /= accum;
            }
            // Z - Iy
            // printf("%d %d\n", int(label[i]), i * k + label[i]);
            z[i * k + label[i]] -= 1.0f;
        }

        // X transpose
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < batch; j++) {
                Xt[i * batch + j] = img[j * n + i];
            }
        }
        // d_w = Xt*(Z - Iy)
        mm(Xt, z, dw, n, k, batch);

        // theta -= lr * d_w
        int len = n * k;
        for (int i = 0; i < len; i++) {
            theta[i] -= lr * (dw[i] / batch);
            // std::printf("%f ", theta[i]);
        }
        // std::printf("/n");
    }

    free(z);
    free(Xt);
    free(dw);
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta, float lr, int batch) {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr), X.request().shape[0],
                X.request().shape[1], theta.request().shape[1], lr, batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
        py::arg("batch"));
}
