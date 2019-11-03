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
    Matrix X = Matrix(500, 10, 0.0001);
    Matrix labels = Matrix({{1, 0, 1, 0, 0, 1, 1, 1, 0, 1}});
    net.fit(X, labels, 10, {500, 50, 256, 256, 128, 64, 1}, 0.01, false, true);
    return 0;
}
