#include "matrix.h"

Matrix::Matrix() {
    this->nRows = 0;
    this->nCols = 0;
    this->values = {{}};
}

Matrix::Matrix(const int& nRows, const int& nCols, const double& random) {
    assert(nRows != 0 && nCols != 0);
    setValues(nRows, nCols, random);
    this->baseMatrix = true;
}

Matrix::Matrix(const std::vector<std::vector<double>>& array) {
    assert(!array.empty());
    for (int i = 0; i < array.size(); ++i) {
        std::vector<double> columnValues;
        for (int j = 0; j < array[i].size(); ++j) {
            columnValues.push_back(array[i][j]);
        }
        this->values.push_back(columnValues);
    }
    this->nRows = array.size();
    this->nCols = array[0].size();
    this->baseMatrix = false;
}

Matrix::Matrix(const Matrix& matrix) {
    this->nRows = matrix.getRows();
    this->nCols = matrix.getCols();
    this->values = matrix.getValues();
}

void Matrix::setValues(const int& nRows, const int& nCols, const double& random) {
    this->nRows = nRows;
    this->nCols = nCols;
    for (int i = 0; i < nRows; ++i) {
        std::vector<double> columnValues;
        for (int j = 0; j < nCols; ++j) {
            double value = random == 0.0001 ? (double)rand()*pow(10, 100)/(pow(10, 100)*RAND_MAX) : random;
            columnValues.push_back(value);
        }
        this->values.push_back(columnValues);
    }
}

void Matrix::setValue(const int& row, const int& col, const double& value) {
    assert(values.size() > row && values[0].size() > col);
    this->values[row][col] = value;
}

Matrix Matrix::multiply(const Matrix& secondMatrix) {
    Matrix matrix = Matrix(this->getRows(), secondMatrix.getCols(), false);
    assert(this->getCols() == secondMatrix.getRows());
    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < secondMatrix.getCols(); ++j) {
            double product = 0;
            for (int k = 0; k < secondMatrix.getRows(); ++k) {
                product += this->getValue(i, k)*secondMatrix.getValue(k, j);
            }
            matrix.setValue(i, j, product);
        }
    }
    return matrix;
}

Matrix Matrix::product (const Matrix& secondMatrix) {
    bool first, second, third, thourth, five, six;
    first = (this->nCols == secondMatrix.getCols() && (this->nRows == secondMatrix.getRows()));
    second = ((this->nCols == 1) && (this->nRows == 1));
    third = ((secondMatrix.getCols() == 1) && (secondMatrix.getRows() == 1));
    thourth = (this->nRows == secondMatrix.getRows() && secondMatrix.getCols() == 1);
    five = (this->nCols==secondMatrix.getCols() && this->nRows==1);
    six = (this->nCols == secondMatrix.getCols() && secondMatrix.getRows() == 1);
    assert(first || (second || third) || thourth || five || six);
    Matrix matrix;
    if (first) {
        matrix = Matrix(this->nRows, this->nCols, 0);
        for (int i = 0; i < this->nRows; ++i) {
            for (int j = 0; j < this->nCols; ++j) {
                matrix.setValue(i, j, this->getValue(i, j)*secondMatrix.getValue(i, j));
            }
        }
    } else if (second) {
        matrix = Matrix(secondMatrix.getRows(), secondMatrix.getCols(), 0);
        for (int i = 0; i < secondMatrix.getRows(); ++i) {
            for (int j = 0; j < secondMatrix.getCols(); ++j) {
                matrix.setValue(i, j, this->getValue(0, 0)*secondMatrix.getValue(i, j));
            }
        }
    } else if (third){
        matrix = Matrix(this->nRows, this->nCols, 0);
        for (int i = 0; i < this->nRows; ++i) {
            for (int j = 0; j < this->nCols; ++j) {
                matrix.setValue(i, j, this->getValue(i, j)*secondMatrix.getValue(0, 0));
            }
        }
    } else if (thourth) {
        matrix = Matrix(this->nRows, this->nCols, 0);
        for (int i = 0; i < this->getRows(); ++i) {
            for (int j = 0; j < this->getCols(); ++j) {
                matrix.setValue(i, j, this->getValue(i, j)*secondMatrix.getValue(j, 0));
            }
        }
    } else if (five) {
        matrix = Matrix(secondMatrix.getRows(), secondMatrix.getCols(), 0);
        for (int i = 0; i < secondMatrix.getRows(); ++i) {
            for (int j = 0; j < secondMatrix.getCols(); ++j) {
                matrix.setValue(i, j, this->getValue(0, j)*secondMatrix.getValue(i, j));
            }
        }
    } else if (six) {
            matrix = Matrix(this->nRows, this->nCols, 0);
            for (int i = 0; i < this->getRows(); ++i) {
                for (int j = 0; j < this->getCols(); ++j) {
                    matrix.setValue(i, j, this->getValue(i, j)*secondMatrix.getValue(i, j));
                }
            }
    }

    return matrix;
}

Matrix Matrix::sum (const Matrix& secondMatrix) {
    bool first, second, third, thourth, five;
    first = (this->nCols == secondMatrix.getCols() && (this->nRows == secondMatrix.getRows()));
    second = ((this->nCols == 1) && (this->nRows == 1));
    third = ((secondMatrix.getCols() == 1) && (secondMatrix.getRows() == 1));
    thourth = (this->nRows == secondMatrix.getRows() && secondMatrix.getCols() == 1);
    five = (this->nCols==secondMatrix.getCols() && this->nRows==1);
    assert(first || (second || third) || thourth||five);
    Matrix matrix;
    if (first) {
        matrix = Matrix(this->nRows, this->nCols, 0);
        for (int i = 0; i < this->nRows; ++i) {
            for (int j = 0; j < this->nCols; ++j) {
                matrix.setValue(i, j, this->getValue(i, j)+secondMatrix.getValue(i, j));
            }
        }
    } else if (second) {
        matrix = Matrix(secondMatrix.getRows(), secondMatrix.getCols(), 0);
        for (int i = 0; i < secondMatrix.getRows(); ++i) {
            for (int j = 0; j < secondMatrix.getCols(); ++j) {
                matrix.setValue(i, j, this->getValue(0, 0)+secondMatrix.getValue(i, j));
            }
        }
    } else if (third){
        matrix = Matrix(this->nRows, this->nCols, 0);
        for (int i = 0; i < this->nRows; ++i) {
            for (int j = 0; j < this->nCols; ++j) {
                matrix.setValue(i, j, this->getValue(i, j)+secondMatrix.getValue(0, 0));
            }
        }
    } else if (thourth) {
        matrix = Matrix(this->nRows, this->nCols, 0);
        for (int i = 0; i < this->getRows(); ++i) {
            for (int j = 0; j < this->getCols(); ++j) {
                matrix.setValue(i, j, this->getValue(i, j)+secondMatrix.getValue(j, 0));
            }
        }
    } else if (five) {
        matrix = Matrix(secondMatrix.getRows(), secondMatrix.getCols(), 0);
        for (int i = 0; i < secondMatrix.getRows(); ++i) {
            for (int j = 0; j < secondMatrix.getCols(); ++j) {
                matrix.setValue(i, j, this->getValue(0, j)+secondMatrix.getValue(i, j));
            }
        }
    }
    return matrix;
}

Matrix Matrix::transpose() {
    Matrix matrix = Matrix(this->nCols, this->nRows, false);
    for (int i = 0; i < this->nRows; i++) {
        for (int j = 0; j < this->nCols; j++) {
            matrix.setValue(j, i, this->getValue(i, j));
        }
    }
    return matrix;
}

Matrix Matrix::minus() {
    Matrix matrix;
    matrix = Matrix(this->nRows, this->nCols, false);
    for (int i = 0; i < this->nRows; ++i) {
        for (int j = 0; j < this->nCols; ++j) {
            matrix.setValue(i, j, -this->getValue(i, j));
        }
    }
    return matrix;
}

Matrix Matrix::divide(const Matrix &secondMatrix) {
    assert(this->getRows()==secondMatrix.getRows()&&
           this->getCols()==secondMatrix.getCols());
    Matrix matrix = Matrix(this->getRows(), this->getCols(), 0);
    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            matrix.setValue(i, j, (double)this->getValue(i, j)/secondMatrix.getValue(i, j));
        }
    }
    return matrix;
}

Matrix Matrix::sumByAxis(const int& axis) {
    Matrix matrix;
    assert(axis >=0 && axis <=2 || axis == NULL);
    if (axis == NULL || axis == 0) {
        double counter = 0.0;
        for (int i = 0; i < this->getRows(); ++i) {
            for (int j = 0; j < this->getCols(); ++j) {
                counter += this->getValue(i, j);
            }
        }
        if (counter != 0.0001) {
            matrix = Matrix(1, 1, counter);
        } else {
            matrix = Matrix(1, 1, 0.0001);
        }
    } else if (axis == 1) {
        matrix = Matrix(1, this->nCols, 0);
        for (int i = 0; i < this->nCols; ++i) {
            double sum = 0;
            for (int j = 0; j < this->nRows; ++j) {
                sum += this->getValue(j, i);
            }
            matrix.setValue(0, i, sum);
        }
    } else {
        matrix = Matrix(1, this->nRows, 0);
        for (int i = 0; i < this->nRows; ++i) {
            double sum = 0;
            for (int j = 0; j < this->nCols; ++j) {
                sum += this->getValue(i, j);
            }
            matrix.setValue(0, i, sum);
        }
    }
    return matrix;
}

void Matrix::reluDer() {
    for (int i = 0; i < this->getRows(); ++i) {
        for (int j = 0; j < this->getCols(); ++j) {
            if (this->getValue(i, j) > 0) {
                this->setValue(i, j, 1);
            } else {
              this->setValue(i, j, 0);
            }
        }
    }
}

Matrix Matrix::getExp() {
    Matrix matrix = Matrix(this->values);
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matrix.setValue(i, j, exp(matrix.getValue(i, j)));
        }
    }
}

Matrix Matrix::ln() {
    Matrix matrix = Matrix(this->values);
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matrix.setValue(i, j, log(matrix.getValue(i, j)));
        }
    }
    return matrix;
}

Matrix Matrix::sigmoidAct() {
    Matrix matrix = Matrix(this->values);
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matrix.setValue(i, j, sigmoid(matrix.getValue(i, j)));
        }
    }
    return matrix;
}

Matrix Matrix::tanhAct() {
    Matrix matrix = Matrix(this->values);
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matrix.setValue(i, j, tanh(matrix.getValue(i, j)));
        }
    }
    return matrix;
}

Matrix Matrix::reluAct() {
    Matrix matrix = Matrix(this->values);
    for (int i = 0; i < matrix.getRows(); ++i) {
        for (int j = 0; j < matrix.getCols(); ++j) {
            matrix.setValue(i, j, relu(matrix.getValue(i, j)));
        }
    }
    return matrix;
}

double Matrix::relu(const double& value) {
    if (value > 0) {
        return value;
    }
    return 0;
}

double Matrix::sigmoid(const double& value) {
    return (1 / (1 + pow(M_E, -value)));
}

void Matrix::print() {
    for (const std::vector<double>& array: this->getValues()) {
        for (const double& values: array) {
            std::cout << values << " ";
        }
        std::cout << std::endl;
    }
}
