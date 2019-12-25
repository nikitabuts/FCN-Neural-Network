#include <iostream>
#include <vector>
#include <ctime>
#include <assert.h>
#include <matrix.h>
#include <map>
#include <string>
#include <cmath>
#include <model.h>

Matrix loadCSV(const std::string &path, bool missFirst){

   std::ifstream ifs(path.c_str());
   assert(ifs);
   std::vector<std::vector<double>> values;
   std::string line;
   while (ifs.good())
   {
      std::getline(ifs, line);
      if (!missFirst) {
        std::replace(line.begin(), line.end(), ',',  ' ');
        std::stringstream ss(line);
        std::string current;
        std::vector<double> temp;
        while (ss >> current)
        {
            try
            {
                if (current.size()) {
                    const double d = std::stod(current);
                    temp.push_back(d);
                }
            }
            catch (const std::exception& e)
            {
            }
        }
        if (current.size()) {
            values.push_back(temp);
        }

      } else {
          missFirst = false;
      }
    }
   ifs.close();
   return Matrix(values).transpose();
}



int main()
{
    Matrix X, XTest, labelsTest, labels;


    X = loadCSV("C:\\GratificationBase.csv", true);
    labels = loadCSV("C:\\GratificationTest.csv", true);
    XTest = loadCSV("C:\\GratificationBase_2.csv", true);
    labelsTest = loadCSV("C:\\GratificationTest_2.csv", true);



    std::cout << "Loaded" << std::endl;
    std::cout << X.getRows() << " " << X.getCols() << std::endl;
    std::cout << labels.getRows() << " " << labels.getCols() << std::endl;

    Model net({X.getRows(), 128, 1}, "he");
    net.fit(X, labels, 100, {"tanh", "sigmoid"}, 0.01, true);

    std::cout << "Predictions:" << std::endl;
    net.predict(X, labels, false).print();

    return 0;
}
