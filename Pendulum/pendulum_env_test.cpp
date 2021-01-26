
#include "pendulum_env.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "rl_env.hpp"


using namespace rl::env;

void print_vec(const std::vector<uint32_t>& vec)
{
    for (const auto& v : vec)
        std::printf("%d ", v);
    std::printf("\n");
}

template <class T>
bool compare_vec(const std::vector<T>& v1,
                 const std::vector<T>& v2)
{
    if (v1.size() != v2.size())
        return false;

    for (uint32_t i=0; i < v1.size(); ++i)
        if (v1[i] != v2[i])
            return false;

    return true;
}

std::vector<std::vector<uint32_t>> expected = {
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 8, 3, 9, 10, 6, 11},
    {12, 13, 8, 14, 9, 10, 15, 11},
    {12, 13, 16, 14, 17, 18, 15, 19},
    {20, 21, 16, 22, 17, 18, 23, 19},
    {0, 1, 2, 3, 24, 25, 26, 27},
    {0, 1, 8, 3, 28, 29, 26, 30},
    {12, 13, 8, 14, 28, 29, 31, 30},
    {12, 13, 16, 14, 32, 33, 31, 34},
    {20, 21, 16, 22, 32, 33, 35, 34},
    {36, 37, 38, 39, 24, 25, 26, 27},
    {36, 37, 40, 39, 28, 29, 26, 30},
    {41, 42, 40, 43, 28, 29, 31, 30},
    {41, 42, 44, 43, 32, 33, 31, 34},
    {45, 46, 44, 47, 32, 33, 35, 34},
    {36, 37, 38, 39, 4, 5, 6, 7},
    {36, 37, 40, 39, 9, 10, 6, 11},
    {41, 42, 40, 43, 9, 10, 15, 11},
    {41, 42, 44, 43, 17, 18, 15, 19},
    {45, 46, 44, 47, 17, 18, 23, 19},
    {0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 8, 3, 9, 10, 6, 11},
    {12, 13, 8, 14, 9, 10, 15, 11},
    {12, 13, 16, 14, 17, 18, 15, 19},
    {20, 21, 16, 22, 17, 18, 23, 19},
};

int main()
{
    std::printf("Pendulum Environment Test\n");
    bool pass = true;

    std::shared_ptr<Environment> env = std::make_shared<PendulumEnvironment>();
    EnvironmentInit params;
    Observation obs;

    env->env_init(params);
    env->env_start();
    for (int i=0; i < 100; ++i)
    {
        obs = env->env_step(0);
        std::printf("(angle, velocity) = (%f, %f)\n", obs.state.angle, obs.state.velocity);
    }
#if 0
    PendulumTileCoder tc;
    const float pi = PendulumTileCoder::pi;
    
    tc.initialize(4096, 8, 2);

    int k = 0;
    for (int i=0; i < 5; ++i)
        for (int j=0; j < 5; ++j)
        {
            float angle = -pi + (i * 2*pi) / (5 - 1);
            float velocity = - 2*pi + (j * 4*pi) / (5 - 1);
            //std::printf("(angle, velocity) = (%f, %f) ", angle, velocity);

            auto tiles = tc.get_tiles(angle, velocity);

            if (!compare_vec<uint32_t>(expected[k],tiles))
            {
                pass = false;
                std::printf("test failed!\nexpected: ");
                print_vec(expected[k]);
                std::printf("instead of: ");
                print_vec(tiles);
            }
            ++k;
        }
#endif
    std::printf("Pendulum Environment Test %s\n", pass ? "Passed" : "Failed");
}
