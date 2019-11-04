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
    Matrix X = Matrix({{0, 0, 1, 1, 0, 1},
                       {1, 0, 0, 0, 1, 1},
                       {0, 0, 0, 1, 1, 1},
                       {0, 1, 1, 1, 0, 0},
                       {1, 1, 1, 1, 0, 0},
                       {0, 1, 1, 0, 0, 1},
                       {0, 0, 1, 1, 0, 0},
                       {0, 1, 0, 1, 1, 0},
                       {1, 0, 1, 0, 1, 0},
                       {0, 1, 1, 0, 1, 1}});
    Matrix labels = Matrix({{0, 0, 1, 0, 1, 1}});
    net.fit(X, labels, 1000, {10, 50, 1}, 0.05, false, true);

    return 0;
}
