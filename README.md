# tianshou
Tianshou(天授) is a reinforcement learning platform.

![alt text](https://github.com/sproblvem/tianshou/blob/master/docs/figures/tianshou_architecture.png "Architecture of tianshou")

## data
TODO:

Replay Memory

Multiple wirter/reader

Importance sampling

## simulator
go(for AlphaGo)

## environment
gym

## core
TODO:

Optimizer

MCTS

## agent (optional)

DQNAgent etc.

Pontential Bugs:
0. Wrong calculation of eval value
UCTNode.cpp
106     if (to_move == FastBoard::WHITE) {
107         net_eval = 1.0f - net_eval;
108     }

309         if (tomove == FastBoard::WHITE) {
310             score = 1.0f - score;
311         }

1. create children only on leaf node
UCTSearch.cpp
 60     if (!node->has_children() && m_nodes < MAX_TREE_SIZE) {
 61         float eval;
 62         auto success = node->create_children(m_nodes, currstate, eval);
 63         if (success) {
 64             result = SearchResult(eval);
 65         }
 66     }



