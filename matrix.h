#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <fstream>

class Matrix {
public:
    Matrix();
    Matrix(const int& nRows, const int& nCols, const double& random);
    Matrix(const std::vector<std::vector<double>>& array);
    Matrix(const Matrix& matrix);

    std::vector<std::vector<double>> getValues() const {return this->values;}

    double getValue(const int& row, const int& col) const {
        assert(values.size() > row && values[0].size() > col);
        return this->values[row][col];
    }

    double onlyOne() const{
        assert(!this->values.empty());
        return this->values[0][0];
    }

    static double relu(const double& value);

    static double sigmoid(const double& value);

    int getRows() const {return this->nRows;}

    int getCols() const {return this->nCols;}

    Matrix multiply(const Matrix& secondMatrix);

    Matrix product(const Matrix& secondMatrix);

    Matrix sum(const Matrix& secondMatrix);

    Matrix transpose();

    void print();

    Matrix minus();

    Matrix divide(const Matrix& secondMatrix);

    Matrix sumByAxis(const int& axis = NULL);

    void reluDer();

    Matrix getExp();

    Matrix ln();

    Matrix tanhAct();

    Matrix sigmoidAct();

    Matrix reluAct();



private:
    int nRows, nCols;
    bool baseMatrix;
    std::vector<std::vector<double>> values;

    void setValues(const int& nRows, const int& nCols, const double& random);
    void setValue(const int& row, const int& col, const double& value);
};

#endif // MATRIX_H
