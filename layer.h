#ifndef LAYER_H
#define LAYER_H

#include <matrix.h>
#include <map>

class Layer {
public:
    Matrix linearForward(Matrix& activations,
                         Matrix& weights, Matrix& biases);

    Matrix linearActForward(Matrix& activationsPrev, Matrix& weights,
                            Matrix& biases,
                            const std::string& type,
                            const int& counter,
                            const bool& isPredict);

    Matrix sigmoidBackward(Matrix& gradA,
                           Matrix& cache);

    Matrix tanhBackward(Matrix& gradA,
                        Matrix& cache);

    Matrix reluBackward(Matrix& gradA,
                        Matrix& cache);

    void linearBackward(Matrix& gradZ, std::map<std::string, Matrix>& cache,
                        std::map<std::string, Matrix>& grads, const int& counter);

    void linearActBackward(Matrix& gradA, std::map<std::string, Matrix>& cache,
                           std::map<std::string, Matrix>& grads, const int& counter,
                           const std::string& activationType);
};

#endif // LAYER_H
