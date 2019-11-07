#include <iostream>
#include <vector>
#include <ctime>
#include <assert.h>
#include <matrix.h>
#include <map>
#include <string>
#include <cmath>
#include <model.h>




int main()
{
    Model net;
    Matrix X = Matrix({{100, -21, 31, -4234, -43, -9},
                       {200, -12, 9.323256, 6.2, 43, 1.000344},
                       {300, -2, -4, 1, 14, 1000},
                       {-31, 4, 5, 1, -43, 4000},
                       {-434, 3, 5435, 4352, 434, 0},
                       {3131, 1, 1, -3131, 0, 1},
                       {-4324, 65678, 4535, 31, -54, 54},
                       {32, 0, 0, 36, 90, 54},
                       {65, -2, -1, -32, 32.3, 131.2},
                       {0, 43.32, 32.43, -43.1, 200, 2000}});
    Matrix labels = Matrix({{0, 0, 1, 0, 1, 1}});
    Matrix testX = Matrix({{100, 32, 0, -313, 435, 0.0000000000001},
                       {32, -32, -323, 434, 889, 434},
                       {1, -21, 2, 2, 434, -3111, 32345},
                       {54.32, 0.43400032, -4345, 221, 442},
                       {326, -3.21, -0.1111},
                       {3, -1, -0.234, 4.215, -4.2155, 9.993},
                       {4, -1488, -3, 228, 5892, 597, -325},
                       {9, 228, 10, 1000, 1000, -11103.4},
                       {3115, -323, 201, -3215, -31.352, 0.63455},
                       {346, 323, -1.112, -4.2157, 3.998, -7.5223}});
    Matrix labelsTest = Matrix({{0, 0, 1, 1, 0, 0}});

    net.fit(X, labels, 1000000, {10, 300, 300, 300, 1}, "he", {"tanh", "relu", "tanh", "sigmoid"}, 0.05, true,
            testX, labelsTest);
    std::cout << "-------------" << std::endl;
    matrixPrint(net.predict(testX, labelsTest, false).getValues());
    std::cout << "-------------" << std::endl;
    matrixPrint(net.predict(testX, labelsTest, false).getValues());

    return 0;
}
