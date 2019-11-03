#ifndef MODEL_H
#define MODEL_H
#endif // MODEL_H


void printKeys(std::map<std::string, Matrix> ma) {
    for (auto elem:ma) {
        std::cout << elem.first << " ";
    }
    std::cout << std::endl;
}

bool layersCheck(const std::vector<int>& layersDims) {
    bool flag = true;
    assert(layersDims.size() > 1);
    for (int i = 0; i < layersDims.size()&&flag; ++i) {
        if (layersDims[i] <= 0) {
            flag = false;
        }
    }
    return flag;
}

double computeLoss(Matrix lastActivation, Matrix labels) {
    const int labelsSize = labels.getCols();
    assert(labelsSize==lastActivation.getCols()&&
           labels.getRows()==1 == lastActivation.getRows());
    double cost = -(labels.multiply(
                        lastActivation.ln(), true).sum(
                                ((Matrix(1,1,1).sum(labels.minus()))).multiply(
                                        (Matrix(1,1,1).sum(lastActivation.minus()).ln()), true)
                   ).sumByAxis().multiply(Matrix(1,1,(double)1/lastActivation.getCols()),true)).onlyOne();

    return cost;
}


void mapPrint(std::map<std::string, Matrix> mapObj) {
    for (auto& elem: mapObj) {
        matrixPrint(elem.second.getValues());
        std::cout << std::endl;
    }
}

class Model {
public:
    void fit(Matrix X, Matrix labels,
             const int& numEpochs,
             const std::vector<int>& layersDims,
             const float& learningRate,
             const bool& useBestModel,
             const bool& print,
             Matrix evalX=Matrix(), Matrix evalLabels=Matrix()) {

        //инициализация весов сети
        assert(X.getCols()==labels.getCols() && labels.getRows()==1);

        assert((evalX.getCols()!=0 && evalX.getRows()!=0 && evalLabels.getCols()==1 && evalLabels.getRows()==evalX.getRows()) ||
                (evalX.getCols()==evalLabels.getCols()==evalX.getRows()==evalX.getCols() &&
                 evalLabels.getCols()==0));
        assert(learningRate > 0 && numEpochs > 0);
        init(layersDims, "random");
        assert(X.getRows()==layersDims[0]);

        //Градиентный спуск
        for (int i = 0; i < numEpochs; ++i) {
            Matrix activations = lModelForward(X);
            std::cout << activations.getRows() << " " << activations.getCols() << std::endl;
            if (print) {
                std::cout << "EPOCH # " << i+1 << std::endl;
                std::cout << "     loss(train set):" << computeLoss(activations, labels) << std::endl;
                std::cout << std::endl;
            }
            lModelBackward(activations, labels);
            std::cout << "------------" << std::endl;
            printKeys(this->grads);
            std::cout << "------------" << std::endl;
            updateParams(learningRate);
        }
    }



private:
    std::map<std::string, Matrix> parameters, cache, grads;

    std::map<std::string, Matrix> init(const std::vector<int>& layersDims,
                                       const std::string& initType) {
        assert(initType=="zeros"||"random"||"xavier");
        assert((!layersDims.empty()) && layersCheck(layersDims));
        for (int i = 1; i < layersDims.size(); ++i) {
            this->parameters['W' + std::to_string(i)] = Matrix(layersDims[i], layersDims[i-1], 0.0001);
            this->parameters['b' + std::to_string(i)] = Matrix(layersDims[i], 1, 0);

            assert((parameters['W' + std::to_string(i)].getRows() == layersDims[i])&&
                    (parameters['W' + std::to_string(i)].getCols()) == layersDims[i-1]);
            assert((parameters['b' + std::to_string(i)].getRows() == layersDims[i])&&
                    (parameters['b' + std::to_string(i)].getCols()) == 1);
        }
        return this->parameters;
    }

    void updateParams(const float& learningRate) {
        const int layersSize = this->parameters.size()/2;
        Matrix learnRate = Matrix(1,1,learningRate);
        for (int i = 0; i < layersSize; ++i) {
            this->parameters["W" + std::to_string(i+1)] =
                    this->parameters["W" + std::to_string(i+1)].sum(
                        learnRate.multiply(
                            this->grads["dW" +
                                std::to_string(i+1)], true).minus());

            this->parameters["b" + std::to_string(i+1)] =
                    this->parameters["b" + std::to_string(i+1)].sum(
                        learnRate.multiply(
                            this->grads["db" +
                                std::to_string(i+1)], true).minus());
        }
    }


    void lModelBackward(Matrix lastActivationForward,
                        Matrix labels) {
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

        linearActBackward(gradLastActivationForward,
                          layersSize-1, "sigmoid");
        std::cout << "layersSize:" << layersSize << std::endl;
        for (int i = layersSize-2; i > 0; --i) {
            Matrix gradA = this->grads["dA" + std::to_string(i+1)];
            linearActBackward(gradA,
                              i, "sigmoid");
        }
    }

    void linearActBackward(Matrix gradA, const int& counter,
                           std::string activationType) {
        assert(activationType == "sigmoid");
        if (activationType == "sigmoid") {
            Matrix gradZ = sigmoidBackward(gradA, counter);
            linearBackward(gradZ, counter);
        }
    }

    void linearBackward(Matrix gradZ, const int& counter) {
        Matrix prevActivations, weights, biases;
        prevActivations = cache["A" + std::to_string(counter - 1)];
        weights = cache["W" + std::to_string(counter)];
        biases = cache["b" + std::to_string(counter)];
        const int prevActivationsSize = prevActivations.getCols();


        Matrix gradW = gradZ.multiply(prevActivations.transpose(), false).multiply(
                    Matrix(1, 1, (double)1/prevActivationsSize), true);

        Matrix gradB = gradZ.sumByAxis(2).multiply(
                    Matrix(1,1,(double)1/prevActivationsSize), true).transpose();

        Matrix gradPrevActivations = weights.transpose().multiply(gradZ, false);

        assert(gradPrevActivations.getRows()==prevActivations.getRows()&&
               gradPrevActivations.getCols()==prevActivations.getCols());

        std::cout << std::endl;
        assert(gradW.getRows()==weights.getRows()&&
               gradW.getCols()==weights.getCols());

        assert(gradB.getRows()==biases.getRows()&&
               gradB.getCols()==biases.getCols());

        grads["dA" + std::to_string(counter)] = gradPrevActivations;
        grads["dW" + std::to_string(counter+1)] = gradW;
        grads["db" + std::to_string(counter+1)] = gradB;
    }

    Matrix sigmoidBackward(Matrix gradA,
                           const int& counter) {
        Matrix derZ = cache["A" + std::to_string(counter)].sigmoidAct().multiply(
                    (Matrix(1,1,1.0).sum(
                         cache["A" + std::to_string(counter)].sigmoidAct().minus())), true);
        Matrix gradZ = gradA.multiply(derZ, true);
        return gradZ;
    }

    Matrix lModelForward(Matrix X) {
        Matrix activation = X;
        const int loopSize = this->parameters.size() / 2;
        Matrix activationPrev;
        for (int i = 1; i < loopSize; ++i) {
            activationPrev = activation;
            activation = linearActForward(activationPrev,
                                              this->parameters["W" + std::to_string(i)],
                    this->parameters["b" + std::to_string(i)],
                    "sigmoid", i);
            std::cout << std::endl;
        }
        Matrix lastActivation = linearActForward(activation,
                                                 this->parameters["W"+std::to_string(loopSize)],
                this->parameters["b"+std::to_string(loopSize)],
                "sigmoid",  loopSize).transpose();
        assert(lastActivation.getCols()==1 &&
               lastActivation.getRows()==X.getCols());
        return lastActivation.transpose();
    }

    std::map<std::string, Matrix> getParams() const {
        return this->parameters;
    }

    std::map<std::string, Matrix> getCache() const {
        return this->cache;
    }

    Matrix linearActForward(Matrix activationsPrev, Matrix weights,
                            Matrix biases, std::string activationType,
                            const int& counter) {
        assert(activationType == "sigmoid");
        if (activationType == "sigmoid") {
            Matrix summMatrix = linearForward(activationsPrev,
                                              weights,
                                              biases,
                                              counter);
            Matrix activationsCur = summMatrix.sigmoidAct();
            this->cache["A" + std::to_string(counter)] = activationsCur;
            return activationsCur;
        }
    }

    Matrix linearForward(Matrix activations, Matrix weights, Matrix biases, const int& counter) {
        Matrix preActivations = weights.multiply(activations, false);
        assert((preActivations.getRows() == weights.getRows())&&
               (preActivations.getCols() == activations.getCols()));
        this->cache["A" + std::to_string(counter-1)] = activations;
        this->cache["W" + std::to_string(counter)] = weights;
        this->cache["b" + std::to_string(counter)] = biases;
        return preActivations;
    }
};
