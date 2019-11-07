#ifndef MODEL_H
#define MODEL_H
#endif // MODEL_H


void printKeys(std::map<std::string, Matrix> ma) {
    for (auto elem:ma) {
        std::cout << elem.first << ": ";
        std::cout << "(" << elem.second.getRows() << "," << elem.second.getCols() << ") ";
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


int boolMask(const double& value, const double& threshold) {
    assert(threshold > 0 && threshold < 1);
    if (value >= threshold) {
        return 1;
    }
    return 0;
}

double accuracy(Matrix labels, Matrix probs, const double& threshold) {
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
             const std::string& initType,
             const std::vector<std::string>& activationType,
             const float& learningRate,
             const bool& print,
             Matrix evalX=Matrix(), Matrix evalLabels=Matrix()) {

        //инициализация весов сети
        assert(X.getCols()==labels.getCols() && labels.getRows()==1);

        assert((evalX.getCols()!=0 && evalX.getRows()!=0 && evalLabels.getRows()==1 && evalLabels.getCols()==evalX.getCols()) ||
                (evalX.getCols()==evalLabels.getCols()==evalX.getRows()==evalX.getCols() &&
                 evalLabels.getCols()==0));
        assert(learningRate > 0 && numEpochs > 0);
        init(layersDims, initType);
        assert(X.getRows()==layersDims[0]);

        this->activationType = activationType;

        //Градиентный спуск
        for (int i = 0; i < numEpochs; ++i) {
            Matrix activations = lModelForward(X, false);
            if (print) {
                std::cout << "EPOCH # " << i+1 << std::endl;
                std::cout << std::endl;
                matrixPrint(activations.getValues());
                std::cout << std::endl;
                matrixPrint(labels.getValues());
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

    std::map<std::string, Matrix> getParams() const {
        return this->parameters;
    }

    std::map<std::string, Matrix> getCache() const {
        return this->cache;
    }

    Matrix predict(Matrix xTest, Matrix labelsTest,
                   const bool& printAccuracy) {
        assert(this->isFit);
        Matrix preds = lModelForward(xTest, true);
        if (printAccuracy) {
            std::cout << "accuracy:" << accuracy(labelsTest, preds, 0.5);
        }
        return preds;
    }



private:
    std::map<std::string, Matrix> parameters, cache, grads;
    std::vector<std::string> activationType;
    bool isFit = false;

    std::map<std::string, Matrix> init(const std::vector<int>& layersDims,
                                       const std::string& initType) {
        assert(initType=="zeros"||"random"||"he");
        assert((!layersDims.empty()) && layersCheck(layersDims));
        for (int i = 1; i < layersDims.size(); ++i) {
            if (initType == "he") {
                this->parameters['W' + std::to_string(i)] = Matrix(layersDims[i], layersDims[i-1], 0.0001).multiply(Matrix(1,1,sqrt((double)2/layersDims[i-1])), true);
            }
            else if (initType == "random") {
                this->parameters['W' + std::to_string(i)] = Matrix(layersDims[i], layersDims[i-1], 0.0001);
            }
            else if (initType == "zeros") {
                this->parameters['W' + std::to_string(i)] = Matrix(layersDims[i], layersDims[i-1], 0);
            }

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
                          layersSize);
        for (int i = layersSize-1; i > 0; --i) {
            Matrix gradA = this->grads["dA" + std::to_string(i+1)];

            linearActBackward(gradA,
                              i);
        }
    }

    void linearActBackward(Matrix gradA, const int& counter) {
        const std::string& activationType = this->activationType[counter-1];

        assert(activationType == "sigmoid" || activationType == "tanh" ||
               activationType == "relu");
        Matrix gradZ;
        if (activationType == "sigmoid") {
            gradZ = sigmoidBackward(gradA, counter);
        }
        else if (activationType == "tanh") {
            gradZ = tanhBackward(gradA, counter);
        }
        else if (activationType == "relu") {
            gradZ = reluBackward(gradA, counter);
        }
        linearBackward(gradZ, counter);
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

        assert(gradW.getRows()==weights.getRows()&&
               gradW.getCols()==weights.getCols());

        assert(gradB.getRows()==biases.getRows()&&
               gradB.getCols()==biases.getCols());

        grads["dA" + std::to_string(counter)] = gradPrevActivations;
        grads["dW" + std::to_string(counter)] = gradW;
        grads["db" + std::to_string(counter)] = gradB;
    }

    Matrix sigmoidBackward(Matrix gradA,
                           const int& counter) {
        Matrix derZ = cache["A" + std::to_string(counter)].multiply(
                    (Matrix(1,1,1.0).sum(
                         cache["A" + std::to_string(counter)].minus())), true);
        Matrix gradZ = gradA.multiply(derZ, true);
        return gradZ;
    }

    Matrix tanhBackward(Matrix gradA,
                           const int& counter) {
        Matrix derZ = Matrix(1,1,1.0).sum(
                         (cache["A" + std::to_string(counter)].multiply(
                            cache["A" + std::to_string(counter)], true).minus()));
        Matrix gradZ = gradA.multiply(derZ, true);
        return gradZ;
    }

    Matrix reluBackward(Matrix gradA,
                           const int& counter) {
        Matrix derZ = cache["A" + std::to_string(counter)];
        derZ.reluDer();
        Matrix gradZ = gradA.multiply(derZ, true);
        return gradZ;
    }

    Matrix lModelForward(Matrix X,
                         const bool& isPredict) {
        Matrix activation = X;
        const int loopSize = this->parameters.size() / 2;
        Matrix activationPrev;
        for (int i = 1; i < loopSize; ++i) {
            activationPrev = activation;
            activation = linearActForward(activationPrev,
                                              this->parameters["W" + std::to_string(i)],
                    this->parameters["b" + std::to_string(i)],
                    i, isPredict);
        }
        Matrix lastActivation = linearActForward(activation,
                                                 this->parameters["W"+std::to_string(loopSize)],
                this->parameters["b"+std::to_string(loopSize)],
                loopSize, isPredict).transpose();
        assert(lastActivation.getCols()==1 &&
               lastActivation.getRows()==X.getCols());
        return lastActivation.transpose();
    }


    Matrix linearActForward(Matrix activationsPrev, Matrix weights,
                            Matrix biases,
                            const int& counter,
                            const bool& isPredict) {
        const std::string& activationType = this->activationType[counter-1];
        assert(activationType == "sigmoid" || activationType == "tanh" ||
               activationType == "relu");
        Matrix summMatrix = linearForward(activationsPrev,
                                          weights,
                                          biases,
                                          counter);
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

       if (!isPredict) {
           this->cache["A" + std::to_string(counter)] = activationsCur;
           this->cache["A" + std::to_string(counter-1)] = activationsPrev;
           this->cache["W" + std::to_string(counter)] = weights;
           this->cache["b" + std::to_string(counter)] = biases;
       }
       return activationsCur;
    }

    Matrix linearForward(Matrix activations, Matrix weights, Matrix biases, const int& counter) {
        Matrix preActivations = weights.multiply(activations, false).sum(biases);
        assert((preActivations.getRows() == weights.getRows())&&
               (preActivations.getCols() == activations.getCols()));
        return preActivations;
    }
};
