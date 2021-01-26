
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>
#include <numeric>
#include <vector>

#include "rl.hpp"
#include "mountain_car_environment.hpp"
#include "sarsa_agent.hpp"

using namespace rl;
using namespace env;
using namespace agent;


int main()
{
    constexpr unsigned int num_runs = 20;
    constexpr unsigned int num_episodes = 100;

    std::shared_ptr<Agent> agent = std::make_shared<SarsaAgent>();
    std::shared_ptr<Environment> env = std::make_shared<MountainCarEnvironment>();

    constexpr float step_size = 0.5;
    AgentInit agent_params{3, 0, 0.1, step_size, 1.0, 0, 8, 8, 4096};
    EnvironmentInit env_params;

    // a list of pairs of (num_tilings, num_tiles)
    const std::vector<std::pair<unsigned int, unsigned int>> agent_options = { {2, 16}, {32, 4}, {8, 8} };
    const unsigned int num_opts = 3;

    std::array<std::array<std::array<unsigned int, num_runs>, num_opts>, num_episodes> all_steps;
    std::array<std::array<float, num_opts>, num_episodes> avg_steps;

    for (unsigned int opt=0; opt < num_opts; ++opt)
    {
        auto tic = std::chrono::steady_clock::now();

        unsigned int num_tilings{0};
        unsigned int num_tiles{0};
        std::tie(num_tilings, num_tiles) = agent_options[opt];
        agent_params.num_tilings = num_tilings;
        agent_params.num_tiles = num_tiles;
        agent_params.step_size = step_size / num_tilings;

        for (unsigned int run=0; run < num_runs; ++run)
        {
            agent_params.seed = run;
            RL rl(env, agent);
            rl.rl_init(env_params, agent_params);

            for (unsigned int episode=0; episode < num_episodes; ++episode)
            {
                rl.rl_episode(15000);
                all_steps[episode][opt][run] = rl.rl_num_steps();
            }
        }

        for (unsigned int episode=0; episode < num_episodes; ++episode)
        {
            float sum = std::accumulate(all_steps[episode][opt].cbegin(), all_steps[episode][opt].cend(), 0);
            avg_steps[episode][opt] = sum / num_runs;
        }

        auto toc = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = toc - tic;
        std::printf("step_size: %4.3f, num_tilings: %d, num_tiles: %d, %5.2f it/s, %4.3f s/it, (elapsed %f s)\n",
                agent_params.step_size, agent_params.num_tilings, agent_params.num_tiles,
                num_runs/diff.count(), diff.count()/num_runs, diff.count());
    }

    std::ofstream fout;
    fout.open ("avg_steps.txt");

    for (unsigned int episode=0; episode < num_episodes; ++episode)
    {
        for (unsigned int opt=0; opt < num_opts; ++opt)
            fout << avg_steps[episode][opt] << " ";
        fout << std::endl;
    }
    fout.close();

}

