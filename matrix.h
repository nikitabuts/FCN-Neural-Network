#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

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

    Matrix multiply(const Matrix& secondMatrix) const;

    Matrix product(const Matrix& secondMatrix) const;

    Matrix sum(const Matrix& secondMatrix);

    Matrix transpose() const;

    void print() const;

    Matrix minus() const;

    Matrix divide (const Matrix& secondMatrix) const;

    Matrix sumByAxis(const int& axis = NULL) const;

    void reluDer();

    Matrix getExp() const;

    Matrix ln() const;

    Matrix tanhAct() const;

    Matrix sigmoidAct() const;

    Matrix reluAct() const;

    Matrix split(int firstColumn, int lastColumn,
                 int firstRow, int lastRow,
                 int step = 1);

    std::vector<double> operator [] (int index) const;

    void toCSV(const std::string& path) const;

    void setValue(const int& row, const int& col, const double& value);



private:
    int nRows, nCols;
    bool baseMatrix;
    std::vector<std::vector<double>> values;

    void setValues(const int& nRows, const int& nCols, const double& random);
};

#endif // MATRIX_H
