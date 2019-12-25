#ifndef LAYER_H
#define LAYER_H

#include <matrix.h>
#include <map>

class Layer {
protected:
    Matrix linearForward(Matrix& activations,
                         Matrix& weights, Matrix& biases) const;

    Matrix linearActForward(Matrix& activationsPrev, Matrix& weights,
                            Matrix& biases,
                            const std::string& type,
                            const int& counter,
                            const bool& isPredict) const;

    Matrix sigmoidBackward(Matrix& gradA,
                           Matrix& cache) const;

    Matrix tanhBackward(Matrix& gradA,
                        Matrix& cache) const;

    Matrix reluBackward(Matrix& gradA,
                        Matrix& cache) const;

    void linearBackward(Matrix& gradZ, std::map<std::string, Matrix>& cache,
                        std::map<std::string, Matrix>& grads, const int& counter) const;

    void linearActBackward(Matrix& gradA, std::map<std::string, Matrix>& cache,
                           std::map<std::string, Matrix>& grads, const int& counter,
                           const std::string& activationType) const;
};

#endif // LAYER_H
