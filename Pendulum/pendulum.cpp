
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "rl.hpp"
#include "pendulum_env.hpp"
#include "actor_critic_agent.hpp"

using namespace rl;
using namespace env;
using namespace agent;


int main()
{
    constexpr unsigned int num_runs = 25;
    constexpr unsigned int max_steps = 20000;

    std::shared_ptr<Agent> agent = std::make_shared<ActorCriticAgent>();
    std::shared_ptr<Environment> env = std::make_shared<PendulumEnvironment>();

    constexpr double step_size = 0.5;
    EnvironmentInit env_params = {0, true};

    AgentInit agent_params;
    agent_params.num_actions = 3;
    agent_params.index_hash_table_size = 4096;
    agent_params.num_tilings = 32;
    agent_params.num_tiles = 8;
    agent_params.actor_step_size = 0.25 / agent_params.num_tilings;
    agent_params.critic_step_size = 2.0 / agent_params.num_tilings;
    agent_params.avg_reward_step_size = std::pow(2, -6);
    agent_params.seed = 0;
    agent_params.use_seed = true;

    std::array<std::array<double, max_steps>, num_runs> returns;
    std::array<std::array<double, max_steps>, num_runs> exp_avg_rewards;

    auto gen = std::mt19937(std::random_device{}());
    auto rand_int = std::uniform_int_distribution<>(0);

    auto tic = std::chrono::steady_clock::now();

    RL rl(env, agent);
    for (unsigned int run=0; run < num_runs; ++run)
    {
        env_params.seed = rand_int(gen);
        agent_params.seed = rand_int(gen);

        rl.rl_init(env_params, agent_params);
        rl.rl_start();

        double total_return{0};

        // exponential average reward without initial bias
        double exp_avg_reward{0};
        double exp_avg_reward_ss{0.01};
        double exp_avg_reward_normalizer{0};

        for (unsigned int step=0; step < max_steps; ++step)
        {
            Observation obs;
            std::tie(obs, std::ignore) = rl.rl_step();

            total_return += obs.reward;

            exp_avg_reward_normalizer += exp_avg_reward_ss * (1.0 - exp_avg_reward_normalizer);
            auto ss = exp_avg_reward_ss / exp_avg_reward_normalizer;
            exp_avg_reward += ss * (obs.reward - exp_avg_reward);

            returns[run][step] = total_return;
            exp_avg_rewards[run][step] = exp_avg_reward;
        }
    }

    auto toc = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = toc - tic;
    std::printf("step_size: %4.3f, num_tilings: %d, num_tiles: %d, %5.2f it/s, %6.5f s/it, (elapsed %f s)\n",
            agent_params.step_size, agent_params.num_tilings, agent_params.num_tiles,
            (num_runs*max_steps)/diff.count(), diff.count()/(num_runs*max_steps), diff.count());

    std::ofstream fout;
    fout.open ("returns.txt");

    for (unsigned int run=0; run < num_runs; ++run)
        for (unsigned int step=0; step < max_steps; ++step)
            fout << returns[run][step] << std::endl;

    fout.close();

    fout.open ("exp_avg_rewards.txt");

    for (unsigned int run=0; run < num_runs; ++run)
        for (unsigned int step=0; step < max_steps; ++step)
            fout << exp_avg_rewards[run][step] << std::endl;

    fout.close();

}

