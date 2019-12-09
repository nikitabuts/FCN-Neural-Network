#ifndef MODEL_H

#include <layer.h>
#include <map>

#define MODEL_H


class Model: private Layer
{
public:
    Model(const std::vector<int>& layersDims,
          const std::string& initType);

    void fit(Matrix X, Matrix labels,
             const int& numEpochs,
             const std::vector<std::string>& activationType,
             const float& learningRate,
             const bool& print,
             Matrix evalX=Matrix(), Matrix evalLabels=Matrix());

    Matrix predict(Matrix& xTest, Matrix& labelsTest,
                   const bool& printAccuracy);

    std::map<std::string, Matrix> getParams() const {
        return this->parameters;
    }

    std::map<std::string, Matrix> getCache() const {
        return this->cache;
    }

private:
    std::map<std::string, Matrix> parameters, cache, grads;
    std::vector<std::string> activationType;
    std::vector<int> layersDims;
    std::string initType;
    bool isFit = false;

    std::map<std::string, Matrix> init();

    Matrix lModelForward(Matrix& X,
                         const bool& isPredict);

    void lModelBackward(Matrix& lastActivationForward,
                        Matrix& labels);

    void updateParams(const float& learningRate);

    static bool layersCheck(const std::vector<int>& layersDims);

    static int boolMask(const double& value, const double& threshold);

    static double accuracy(Matrix labels, Matrix probs, const double& threshold);

    static double computeLoss(Matrix lastActivation, Matrix labels);
};

#endif // MODEL_H
