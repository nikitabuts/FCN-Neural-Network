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
    Model net({6, 512, 1}, "he");
    Matrix X = Matrix({{0, 0, 1, 0, 1, 1},
                       {0, 1, 1, 0, 0, 0},
                       {0, 1, 0, 1, 1, 0},
                       {1, 1, 0, 0, 0, 0},
                       {0, 0, 1, 1, 1, 1},
                       {1, 0, 1, 1, 0, 1}});
    Matrix labels = Matrix({{1, 1, 0, 0, 1, 1, 0, 1, 0, 0}});
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

    net.fit(testX.transpose(), labels, 100, {"tanh", "sigmoid"}, 0.01, true);
    std::cout << "-------------" << std::endl;
    net.predict(X, labels, false).print();


    return 0;
}
