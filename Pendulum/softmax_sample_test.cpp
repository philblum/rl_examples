#include <iostream>
#include <string>
#include <map>
#include <random>
 
int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // 50% of the time, generate a random number between 0 and 1
    // 50% of the time, generate a random number between 10 and 15
    std::vector<double> i{ 0, 1, 2, 3 };
    std::vector<double> w{ 0.15, 0.30, 0.7 }; 
    std::piecewise_constant_distribution<> d(i.begin(), i.end(), w.begin());
 
    std::map<int, int> hist;
    for(int n=0; n<10000; ++n)
        ++hist[std::floor(d(gen))];

    for(auto p : hist) {
        std::cout << p.first << ' ' << std::string(p.second/100, '*') << '\n';
    }
}
