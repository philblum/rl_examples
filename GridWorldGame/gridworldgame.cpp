
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
#include "gridworldgame_environment.hpp"
#include "expected_sarsa_agent.hpp"
#include "q_learning_agent.hpp"

using namespace rl;
using namespace env;
using namespace agent;

int main()
{
    std::printf("%s: start\n", __func__);

    constexpr unsigned int num_runs = 100;
    constexpr unsigned int num_episodes = 250;

    const std::map<std::string, std::shared_ptr<Agent>> agents {
        { "Expected Sarsa", std::make_shared<ExpectedSarsaAgent>() },
        { "Q Learning", std::make_shared<QLearningAgent>() },
    };

    std::map<std::string, std::array<std::array<float, num_runs>, num_episodes>> all_returns;
    std::vector<float> avg_returns(num_episodes);

    std::shared_ptr<Environment> env = std::make_shared<GridWorldGameEnvironment>();

    AgentInit agent_params{4, 250, 0.1, 0.1, 0.8, 0};
    EnvironmentInit env_params;

    auto begin = std::chrono::steady_clock::now();
    for (auto agent : agents)
    {
        for (unsigned int run=0; run < num_runs; ++run)
        {
            agent_params.seed = run;
            RL rl(env, agent.second);
            rl.rl_init(env_params, agent_params);

            for (unsigned int episode=0; episode < num_episodes; ++episode)
            {
                rl.rl_episode(0);
                all_returns[agent.first][episode][run] = rl.rl_return();
            }
        }

    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - begin;
    std::printf("%5.2f it/s, %4.3f s/it, (total %f)\n", num_runs/diff.count(), diff.count()/num_runs, diff.count());

    std::ofstream f;
    f.open ("avg_returns.txt");

    for (unsigned int episode=0; episode < num_episodes; ++episode)
    {
        for (auto agent : agents)
        {
            if(episode == 0) std::printf("%s\n", agent.first.c_str());
            float sum = std::accumulate(all_returns[agent.first][episode].cbegin(), all_returns[agent.first][episode].cend(), 0);
            avg_returns[episode] = sum / num_runs;
            f << sum / num_runs << " ";
        }
        f << std::endl;
    }
    f.close();

    std::printf("%s: done\n", __func__);

}
