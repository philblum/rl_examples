
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>

#include "actor_critic_agent.hpp"
#include "pendulum_tc.hpp"
#include "pendulum_env.hpp"
#include "rl.hpp"
#include "rl_agent.hpp"
#include "rl_types.hpp"

using namespace pendulum_tc;

void print_vec(const std::vector<unsigned int>& vec)
{
    for (const auto& v : vec)
        std::printf("%d ", v);
    std::printf("\n");
}

void print_vec(const std::vector<double>& vec)
{
    for (const auto& v : vec)
        std::printf("%10.8f ", v);
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

bool compare_vec(const std::vector<double>& v1,
                 const std::vector<double>& v2,
                 const double tol,
                 unsigned count=0)
{
    std::size_t size = 0;
    if (count == 0 && v1.size() != v2.size())
        return false;
    else if (count == 0)
        count = v1.size();

    for (uint32_t i=0; i < count; ++i)
    {
        if (std::abs(v1[i] - v2[i]) > tol)
            return false;
    }
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

std::vector<double> expected_prob = { 0.03511903, 0.25949646, 0.70538451 };

std::vector<double> expected_critic_weights =
{
    -0.39238658, -0.39238658, -0.39238658, -0.39238658, -0.39238658,
     -0.39238658, -0.39238658, -0.39238658, 0.0, 0.0
};

std::vector<std::vector<double>> expected_actor_weights =
{
    { 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.0, 0.0 },
    { -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, -0.02615911, 0.0, 0.0 },
    { 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.01307955, 0.0, 0.0 }
};


class ActorCriticAgentTest : public ActorCriticAgent
{
public:
    ActorCriticAgentTest() : ActorCriticAgent() { };
    ~ActorCriticAgentTest() {};

    void set_actor_weights(const Action action, const double value)
    {
        using index = Float2D::index;
        for(index j = 0; j < index_hash_table_size; ++j)
        {
            actor_weights[action][j] = value;
            //std::printf("actor_weights[%d][%d] = %f\n", action, j, actor_weights[action][j]);
        }
    }
    void set_actor_weights(const Action action, const std::vector<uint32_t>& tiles, const std::vector<double>& values)
    {
        using index = Float2D::index;
        for(index j = 0; j < tiles.size(); ++j)
        {
            actor_weights[action][tiles[j]] = values[j];
        }
    }
    std::vector<double> get_actor_weights(const Action action)
    {
        std::vector<double> w(index_hash_table_size, 0);

        using index = Float2D::index;
        for(index j = 0; j < index_hash_table_size; ++j)
            w[j] = actor_weights[action][j];

        return w;
    }

    void print_actor_weights(const Action action, unsigned int start_idx) const
    {
        using index = Float2D::index;
        for(index j = start_idx; j < start_idx + 10; ++j)
        {
            std::printf("actor_weights[%d][%lu] = %f\n", action, j, actor_weights[action][j]);
        }
    }

    void print_critic_weights(unsigned int start_idx) const
    {
        for(unsigned int j = start_idx; j < start_idx + 10; ++j)
        {
            std::printf("critic_weights[%d] = %f\n", j, critic_weights[j]);
        }
    }

    std::vector<double> get_critic_weights() { return critic_weights; }

    std::vector<double> return_softmax_prob(const std::vector<uint32_t>& tiles)
    {
        return get_softmax_prob(actor_weights, tiles);
    }

    Action get_prev_action() const { return prev_action; }

    std::vector<uint32_t> get_prev_tiles() const { return prev_tiles; }

    double get_avg_reward() const { return avg_reward; }
};

int main()
{
    std::printf("Pendulum Test\n");
    PendulumTileCoder tc;
    const double pi = PendulumTileCoder::pi;
    
    tc.initialize(4096, 8, 2);
    bool pass = true;

    int k = 0;
    for (int i=0; i < 5; ++i)
        for (int j=0; j < 5; ++j)
        {
            double angle = -pi + (i * 2*pi) / (5 - 1);
            double velocity = - 2*pi + (j * 4*pi) / (5 - 1);
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
    std::printf("Pendulum Tile Coder Test %s\n", pass ? "Passed" : "Failed");

    pass = true;
    AgentInit params;
    params.num_actions = 3;
    params.index_hash_table_size = 4096;
    params.num_tilings = 8;
    params.num_tiles = 8;
    params.seed = 0;
    params.use_seed = true;

    tc.initialize(4096, 8, 8);
    std::shared_ptr<ActorCriticAgentTest> agent = std::make_shared<ActorCriticAgentTest>();
    agent->agent_init(params);
    agent->set_actor_weights(0, -1.0 / params.num_tilings);
    agent->set_actor_weights(1, 1.0 / params.num_tilings);
    agent->set_actor_weights(2, 2.0 / params.num_tilings);

    auto tiles = tc.get_tiles(-pi, 0.0);
    auto p = agent->return_softmax_prob(tiles);

    if (!compare_vec(expected_prob, p, 0.000001))
    {
        pass = false;
        std::printf("softmax test1 failed!\nexpected: ");
        print_vec(expected_prob);
        std::printf("instead of: ");
        print_vec(p);
    }

    tc.initialize(4096, 32, 8);
    tiles.resize(32); for (int i=0; i < 32; ++i) tiles[i] = i;
    agent->set_actor_weights(0, 0.00818123);
    agent->set_actor_weights(1, 0.-0.01636246);
    agent->set_actor_weights(2, 0.00818123);
    expected_prob = { 0.40717638, 0.18564724, 0.40717638 };
    double c =  0.2617993877991494;

    p = agent->return_softmax_prob(tiles);
    if (!compare_vec(expected_prob, p, 0.0000001))
    {
        pass = false;
        std::printf("softmax test2 failed!\nexpected: ");
        print_vec(expected_prob);
        std::printf("instead of: ");
        print_vec(p);
    }

    tiles = { 0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 63, 60, 61, 64, 56, 55, 54, 53, 52, 50, 49, 48, 58, 62 };
    agent->set_actor_weights(0, tiles,
            { 0.00213973, -0.05899639, -0.04635183, -0.04635183, -0.04635183, -0.04635183, -0.04635183, -0.04635183,
              -0.02991508, -0.03232526, -0.01094126, -0.02842379, -0.04635183, -0.04635183, -0.04635183, -0.04635183,
              -0.04635183 -0.04635183, 0.0, 0.0102764, 0.00981778,  0.0, 0.01209439,  0.01209022,
              0.01344424, -0.00314583, -0.00312463, -0.00308989, -0.00304784,  0.01354162, 0.01164353, 0.00955686 });
    agent->set_actor_weights(1, tiles, {-3.88545674e-02, -3.27481224e-02, -6.28532910e-02, -6.28532910e-02,
     -6.28532910e-02, -6.28532910e-02, -6.28532910e-02, -6.28532910e-02,
     -6.51069193e-02, -3.10638768e-02, -3.40211865e-02, -6.18173436e-02,
     -6.28532910e-02, -6.28532910e-02, -6.28532910e-02, -6.28532910e-02,
     -6.28532910e-02, -6.28532910e-02,  0.00000000e+00,  2.72189839e-04,
     -8.12375332e-05,  0.00000000e+00,  1.63817238e-03,  1.63504825e-03,
     -3.25301815e-02, -3.04838071e-02, -3.04557024e-02, -3.04091534e-02,
     -3.03522191e-02, -3.23975813e-02,  1.30000361e-03, -2.81681470e-04 });
    agent->set_actor_weights(2, tiles, { 0.03671484,  0.09174451,  0.10920512,  0.10920512,  0.10920512,  0.10920512,
      0.10920512,  0.10920512,  0.095022,    0.06338914,  0.04496244,  0.09024113,
      0.10920512,  0.10920512,  0.10920512,  0.10920512,  0.10920512,  0.10920512,
      0.0,         -0.01054859, -0.00973654,  0.0,         -0.01373257, -0.01372527,
      0.01908594,  0.03362963,  0.03358033,  0.03349904,  0.03340006,  0.01885597, -0.01294353, -0.00927518 });
    expected_prob = {0.0747286,  0.04245703, 0.88281437 };

    p = agent->return_softmax_prob(tiles);
    if (!compare_vec(expected_prob, p, 0.0000001))
    {
        pass = false;
        std::printf("softmax test3 failed!\nexpected: ");
        print_vec(expected_prob);
        std::printf("instead of: ");
        print_vec(p);
    }

    std::printf("Pendulum Softmax Test %s\n", pass ? "Passed" : "Failed");

    pass = true;
    params.actor_step_size = 1e-1 / params.num_tilings;
    params.critic_step_size = 1.0 / params.num_tilings;
    params.avg_reward_step_size = 1e-2;
    params.seed = 0;
    params.use_seed = true;

    ActorCriticAgentTest test_agent;
    test_agent.agent_init(params);

    State state = {-pi, 0.0};

    test_agent.agent_start(state);
    std::vector<uint32_t> expected_prev_tiles{0, 1, 2, 3, 4, 5, 6, 7};
    if (!compare_vec<uint32_t>(test_agent.get_prev_tiles(),
            std::vector<uint32_t> {0, 1, 2, 3, 4, 5, 6, 7}))
    {
        pass = false;
        std::printf("test failed!\nexpected: ");
        print_vec(expected_prev_tiles);
        std::printf("instead of: ");
        print_vec(test_agent.get_prev_tiles());
    }

    if (test_agent.get_prev_action() == params.num_actions)
    {
        pass = false;
        std::printf("test failed!\nexpected: %d instead of: %d\n",
                1, test_agent.get_prev_action());
    }
    std::printf("Pendulum agent_start() Test %s\n", pass ? "Passed" : "Failed");

    pass = true;
    std::shared_ptr<ActorCriticAgentTest> agent2 = std::make_shared<ActorCriticAgentTest>();
    std::shared_ptr<Environment> env = std::make_shared<PendulumEnvironment>();
    EnvironmentInit env_params;

    RL rl(env, agent2);
    rl.rl_init(env_params, params);
    rl.rl_start();
    rl.rl_step();

    if (agent2->get_prev_action() != 2)
    {
        pass = false;
        std::printf("test failed!\nexpected: %d instead of: %d\n",
                2, agent2->get_prev_action());
    }
    if (std::abs(agent2->get_avg_reward() + 0.03139) > 0.00005)
    {
        pass = false;
        std::printf("test failed!\nexpected: %f instead of: %f\n",
                -0.03139, agent2->get_avg_reward());
    }
    for (int i = 0; i < 3; ++i)
    {
        if (!compare_vec(expected_actor_weights[i], agent2->get_actor_weights(i), 0.0001, 10))
        {
            pass = false;
            std::printf("test failed!\nexpected: ");
            print_vec(expected_critic_weights);
            std::printf("instead of: ");
            print_vec(agent2->get_critic_weights());
            break;
        }
    }
    if (!compare_vec(expected_critic_weights, agent2->get_critic_weights(), 0.0005, 10))
    {
        pass = false;
        std::printf("test failed!\nexpected: ");
        print_vec(expected_critic_weights);
        std::printf("instead of: ");
        print_vec(agent2->get_critic_weights());
    }

    std::printf("Pendulum rl_step() Test %s\n", pass ? "Passed" : "Failed");

    std::random_device rd;
    std::mt19937 gen(rd());

    p = { 0.4, 0.3, 0.3 };
    static const std::vector<double> i{ 0, 1, 2, 3 };
    std::piecewise_constant_distribution<double> d(i.begin(), i.end(),
            p.begin());
    Action action = std::floor(d(gen));

    std::map<int, int> hist;
    for(int n=0; n<10000; ++n) {
        Action action = d(gen);
        ++hist[action];
    }
    for(auto p : hist) {
        std::cout << p.first << ' ' << std::string(p.second/100, '*') << '\n';
    }


}
