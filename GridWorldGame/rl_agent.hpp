#pragma once

#include <random>
#include <string>
#include <utility>
#include "boost/multi_array.hpp"
#include "rl_types.hpp"

namespace rl {
namespace agent {

struct AgentInit {
    unsigned int num_actions{0};
    unsigned int num_states{0};
    float epsilon{0.1};   // for exploration
    float step_size{0.1}; // alpha
    float discount{1.0};  // the discount factor
    unsigned int seed{0};
};

class Agent {
public:
    Agent () = default;
    virtual ~Agent() = default;

    virtual void agent_init(const AgentInit& params) = 0;
    virtual Action agent_start(const State state) = 0;
    virtual Action agent_step(const float reward, const State state) = 0;
    virtual void agent_end(const float reward) = 0;
    virtual void agent_cleanup() = 0;
    virtual std::string agent_message(const std::string& message) = 0;

protected:
    unsigned int num_actions{0};
    unsigned int num_states{0};
    float epsilon{0.1};   // for exploration
    float step_size{0.1}; // alpha
    float discount{1.0};  // the discount factor
    unsigned int seed{0};
    std::mt19937 gen; //Standard mersenne_twister_engine
    std::uniform_real_distribution<> rand_real;
    std::uniform_int_distribution<> rand_int;

    State prev_state{0};
    Action prev_action{0};

    using Float2D = boost::multi_array<float, 2>;
    Float2D q_values;
    Float2D weights;

    /* argmax but with random tie-breaking */
    std::pair<Action, float> argmax(Float2D::array_view<1>::type q_view)
    {
        float top = -HUGE_VALF;
        std::vector<unsigned int> ties;

        using index = Float2D::index;
        for(index i = 0; i < num_actions; ++i)
        {
            if (q_view[i] > top)
            {
                top = q_view[i];
                ties.clear();
            }
            if (q_view[i] == top)
                ties.push_back(i);
        }

        std::uniform_int_distribution<> sample(0, ties.size()-1);
        Action winning_idx = ties[sample(gen)];
        float winning_val = q_view[winning_idx];

        return std::make_pair(winning_idx, winning_val);
    }

    /* selects an action using epsilon greedy with random tie-breaking */
    std::pair<Action, float> select_action(const std::vector<uint32_t>& tiles)
    {
        std::vector<float> q_values(num_actions, 0);
        for(Action i = 0; i < num_actions; ++i)
        {
            for (std::size_t j=0; j < tiles.size(); ++j)
                q_values[i] += weights[i][tiles[j]];
        }

        float top = -HUGE_VALF;
        std::vector<unsigned int> ties;

        for(unsigned i = 0; i < num_actions; ++i)
        {
            if (q_values[i] > top)
            {
                top = q_values[i];
                ties.clear();
            }
            if (q_values[i] == top)
                ties.push_back(i);
        }

        std::uniform_int_distribution<> sample(0, ties.size()-1);
        Action winning_idx = ties[sample(gen)];
        float winning_val = q_values[winning_idx];

        return std::make_pair(winning_idx, winning_val);
    }

};

} // agent
} // rl
