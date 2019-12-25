#include "layer.h"

Matrix Layer::linearForward(Matrix &activations,
                            Matrix &weights, Matrix &biases) const {
    Matrix preActivations = weights.multiply(activations).sum(biases);
    assert((preActivations.getRows() == weights.getRows())&&
           (preActivations.getCols() == activations.getCols()));
    return preActivations;
}

Matrix Layer::linearActForward(Matrix &activationsPrev,
                               Matrix &weights,
                               Matrix &biases,
                               const std::string& type,
                               const int &counter, const bool &isPredict) const {

    const std::string& activationType = type;
    assert(activationType == "sigmoid" || activationType == "tanh" ||
           activationType == "relu");
    Matrix summMatrix = linearForward(activationsPrev,
                                      weights,
                                      biases);
   Matrix activationsCur;
   if (activationType == "sigmoid") {
        activationsCur = summMatrix.sigmoidAct();
   }
   else if (activationType == "tanh"){
         activationsCur = summMatrix.tanhAct();
   }
   else if (activationType == "relu") {
        activationsCur = summMatrix.reluAct();
   }

   return activationsCur;
}

Matrix Layer::sigmoidBackward(Matrix& gradA,
                              Matrix& cache) const {
    Matrix derZ = cache.product(
                (Matrix(1,1,1.0).sum(
                     cache.minus())));
    Matrix gradZ = gradA.product(derZ);
    return gradZ;
}

Matrix Layer::tanhBackward(Matrix& gradA,
                           Matrix& cache) const {
    Matrix derZ = Matrix(1,1,1.0).sum(
                     (cache.product(
                        cache).minus()));
    Matrix gradZ = gradA.product(derZ);
    return gradZ;
}

Matrix Layer::reluBackward(Matrix& gradA,
                           Matrix& cache) const {
    Matrix derZ = cache;
    derZ.reluDer();
    Matrix gradZ = gradA.product(derZ);
    return gradZ;
}

void Layer::linearBackward(Matrix& gradZ, std::map<std::string, Matrix>& grads,
                           std::map<std::string, Matrix>& cache,
                           const int& counter) const {
    Matrix prevActivations, weights, biases;
    prevActivations = cache["A" + std::to_string(counter -  1)];
    weights = cache["W" + std::to_string(counter)];
    biases = cache["b" + std::to_string(counter)];
    const int prevActivationsSize = prevActivations.getCols();


    Matrix gradW = gradZ.multiply(prevActivations.transpose()).product(
                Matrix(1, 1, (double)1/prevActivationsSize));

    Matrix gradB = gradZ.sumByAxis(2).product(
                Matrix(1,1,(double)1/prevActivationsSize)).transpose();


    Matrix gradPrevActivations = weights.transpose().multiply(gradZ);

    assert(gradPrevActivations.getRows()==prevActivations.getRows()&&
           gradPrevActivations.getCols()==prevActivations.getCols());

    assert(gradW.getRows()==weights.getRows()&&
           gradW.getCols()==weights.getCols());

    assert(gradB.getRows()==biases.getRows()&&
           gradB.getCols()==biases.getCols());

    grads["dA" + std::to_string(counter)] = gradPrevActivations;
    grads["dW" + std::to_string(counter)] = gradW;
    grads["db" + std::to_string(counter)] = gradB;
}

void Layer::linearActBackward(Matrix& gradA, std::map<std::string, Matrix>& cache,
                              std::map<std::string, Matrix>& grads,
                              const int& counter,
                              const std::string& activationType) const {
    assert(activationType == "sigmoid" || activationType == "tanh" ||
           activationType == "relu");
    Matrix gradZ;
    Matrix cached = cache["A" + std::to_string(counter)];
    if (activationType == "sigmoid") {
        gradZ = sigmoidBackward(gradA, cached);
    }
    else if (activationType == "tanh") {
        gradZ = tanhBackward(gradA, cached);
    }
    else if (activationType == "relu") {
        gradZ = reluBackward(gradA, cached);
    }

    linearBackward(gradZ, grads, cache, counter);
}

