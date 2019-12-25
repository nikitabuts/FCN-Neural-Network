#include "model.h"

Model::Model(const std::vector<int>& layersDims,
             const std::string& initType) {
    this->initType = initType;
    this->layersDims = layersDims;
    init();
}

void Model::init() {
    assert(initType=="zeros"||"random"||"he");
    assert((!layersDims.empty()) && layersCheck(layersDims));
    for (int i = 1; i < this->layersDims.size(); ++i) {
        if (this->initType == "he") {
            this->parameters['W' + std::to_string(i)] = Matrix(this->layersDims[i], this->layersDims[i-1], 0.0001).product(Matrix(1,1,sqrt((double)2/this->layersDims[i-1])));
        }
        else if (this->initType == "random") {
            this->parameters['W' + std::to_string(i)] = Matrix(this->layersDims[i], this->layersDims[i-1], 0.0001);
        }
        else if (this->initType == "zeros") {
            this->parameters['W' + std::to_string(i)] = Matrix(this->layersDims[i], this->layersDims[i-1], 0);
        }

        this->parameters['b' + std::to_string(i)] = Matrix(this->layersDims[i], 1, 0);

        assert((parameters['W' + std::to_string(i)].getRows() == this->layersDims[i])&&
                (parameters['W' + std::to_string(i)].getCols()) == this->layersDims[i-1]);
        assert((parameters['b' + std::to_string(i)].getRows() == this->layersDims[i])&&
                (parameters['b' + std::to_string(i)].getCols()) == 1);
    }
}

Matrix Model::lModelForward(Matrix& X,
                     const bool& isPredict) {
    Matrix activation = X;
    const int loopSize = this->parameters.size() / 2;
    Matrix activationPrev;
    for (int i = 1; i < loopSize; ++i) {
        activationPrev = activation;
        activation = linearActForward(activationPrev,
                                          this->parameters["W" + std::to_string(i)],
                this->parameters["b" + std::to_string(i)],
                this->activationType[i-1],
                i, isPredict);

        if (!isPredict) {
            this->cache["A" + std::to_string(i)] = activation;
            this->cache["A" + std::to_string(i-1)] = activationPrev;
            this->cache["W" + std::to_string(i)] = this->parameters["W" + std::to_string(i)];
            this->cache["b" + std::to_string(i)] = this->parameters["b" + std::to_string(i)];
        }
    }

    Matrix lastActivation = linearActForward(activation,
                                             this->parameters["W"+std::to_string(loopSize)],
            this->parameters["b"+std::to_string(loopSize)],
            this->activationType[loopSize-1],
            loopSize, isPredict).transpose();

    if (!isPredict) {
        this->cache["A" + std::to_string(loopSize)] = lastActivation.transpose();
        this->cache["A" + std::to_string(loopSize-1)] = activation;
        this->cache["W" + std::to_string(loopSize)] = this->parameters["W" + std::to_string(loopSize)];
        this->cache["b" + std::to_string(loopSize)] = this->parameters["b" + std::to_string(loopSize)];
    }


    assert(lastActivation.getCols()==1 &&
           lastActivation.getRows()==X.getCols());
    return lastActivation.transpose();
}

void Model::lModelBackward(Matrix& lastActivationForward,
                    Matrix& labels) {
    assert(lastActivationForward.getRows()==labels.getRows()==1&&
           lastActivationForward.getCols()==labels.getCols());
    const int layersSize = parameters.size()/2;
    const int lastActivationForwardSize = lastActivationForward.getRows();


    Matrix gradLastActivationForward =
            labels.divide(lastActivationForward).sum(
                (Matrix(1,1,1.0).sum(
                     labels.minus()).divide(
                            Matrix(1,1,1.0).sum(
                                lastActivationForward.minus(
                                        )))).minus()).minus();


    linearActBackward(gradLastActivationForward, this->cache, this->grads,
                      layersSize, this->activationType[layersSize - 1]);

    for (int i = layersSize-1; i > 0; --i) {
        Matrix gradA = this->grads["dA" + std::to_string(i+1)];

        linearActBackward(gradA, this->cache, this->grads,
                          i, this->activationType[i - 1]);
    }
}

void Model::updateParams(const float& learningRate) {
    const int layersSize = this->parameters.size()/2;
    Matrix learnRate = Matrix(1,1,learningRate);
    for (int i = 0; i < layersSize; ++i) {
        this->parameters["W" + std::to_string(i+1)] =
                this->parameters["W" + std::to_string(i+1)].sum(
                    learnRate.product(
                        this->grads["dW" +
                            std::to_string(i+1)]).minus());

        this->parameters["b" + std::to_string(i+1)] =
                this->parameters["b" + std::to_string(i+1)].sum(
                    learnRate.product(
                        this->grads["db" +
                            std::to_string(i+1)]).minus());
    }
}

bool Model::layersCheck(const std::vector<int>& layersDims) {
    bool flag = true;
    assert(layersDims.size() > 1);
    for (int i = 0; i < layersDims.size()&&flag; ++i) {
        if (layersDims[i] <= 0) {
            flag = false;
        }
    }
    return flag;
}


int Model::boolMask(const double& value, const double& threshold) {
    assert(threshold > 0 && threshold < 1);
    if (value >= threshold) {
        return 1;
    }
    return 0;
}

Matrix Model::masking(const Matrix &matrix, const double &threshold) {
    Matrix matr = Matrix(matrix.getRows(), matrix.getCols(), 0);
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matr.setValue(i, j, boolMask(matrix.getValue(i, j), threshold));
        }
    }
    return matr;
}

double Model::accuracy(Matrix labels, Matrix probs, const double& threshold) {
    int sum = 0;
    for (int i = 0; i < labels.getRows(); ++i) {
        for (int j = 0; j < labels.getCols(); ++j) {
            if (boolMask(probs.getValue(i, j), threshold) == labels.getValue(i, j)) {
                ++sum;
            }
        }
    }
    return sum/(double)labels.getCols();
}

double Model::computeLoss(Matrix lastActivation, Matrix labels) {
    const int labelsSize = labels.getCols();
    assert(labelsSize==lastActivation.getCols()&&
           labels.getRows()==1 == lastActivation.getRows());
    double cost = -(labels.product(
                        lastActivation.ln()).sum(
                                ((Matrix(1,1,1).sum(labels.minus()))).product(
                                        (Matrix(1,1,1).sum(lastActivation.minus()).ln()))
                   ).sumByAxis().product(Matrix(1,1,(double)1/lastActivation.getCols()))).onlyOne();

    return cost;
}

void Model::fit(Matrix X, Matrix labels,
         const int& numEpochs,
         const std::vector<std::string>& activationType,
         const float& learningRate,
         const bool& print,
         Matrix evalX, Matrix evalLabels) {

    //инициализация весов сети
    assert(X.getCols()==labels.getCols() && labels.getRows()==1);

    assert((evalX.getCols()!=0 && evalX.getRows()!=0 && evalLabels.getRows()==1 && evalLabels.getCols()==evalX.getCols()) ||
            (evalX.getCols()==evalLabels.getCols()==evalX.getRows()==evalX.getCols() &&
             evalLabels.getCols()==0));
    assert(learningRate > 0 && numEpochs > 0);
    assert(X.getRows()==this->layersDims[0]);
    assert(activationType.size() + 1 == this->layersDims.size());
    this->activationType = activationType;

    //Градиентный спуск
    for (int i = 0; i < numEpochs; ++i) {
        Matrix activations = lModelForward(X, false);
        if (print) {
            std::cout << "EPOCH # " << i+1 << std::endl;
            std::cout << std::endl;
            activations.print();
            std::cout << std::endl;
            labels.print();
            std::cout << std::endl;
            std::cout << "---------------------------------" << std::endl;
            std::cout << "     loss(train set):" << computeLoss(activations, labels) << std::endl;
            std::cout << "     accuracy(train set):" << accuracy(labels, activations, 0.5) << std::endl;
            std::cout << "---------------------------------" << std::endl;
            if (evalLabels.getRows() != 0) {
                Matrix evalActivations = lModelForward(evalX, true);
                std::cout << "     loss(eval set):" << computeLoss(evalActivations, evalLabels) << std::endl;
                std::cout << "     accuracy(eval set):" << accuracy(evalLabels, evalActivations, 0.5) << std::endl;
            }
            std::cout << std::endl;
        }
        lModelBackward(activations, labels);
        updateParams(learningRate);
    }
    this->isFit = true;
}

Matrix Model::predict(Matrix& xTest, Matrix& labelsTest,
               const bool& printAccuracy) {
    assert(this->isFit);
    Matrix preds = lModelForward(xTest, true);
    if (printAccuracy) {
        std::cout << "accuracy:" << accuracy(labelsTest, preds, 0.5);
    }
    return masking(preds, 0.5);
}

double Model::crossValScore(Matrix X, Matrix labels, const int& nSplits,
                            const int& cv,
                            const int& numEpochs, const std::vector<std::string> &activationType,
                            const float &learningRate) {
    int baseValue = 0;
    for (int i = 0; i < cv; ++i) {
       // Matrix trainX, testX = X.split(baseValue, , X.getCols(), X.getCols()), X.split();
    }
}
