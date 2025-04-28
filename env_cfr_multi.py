import random
import copy
import numpy as np
from itertools import combinations
from enum import Enum

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class Stage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    def __repr__(self):
        return f"{self.rank}{self.suit}"

def create_deck():
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    suits = ['H','D','C','S']
    deck = [Card(r,s) for r in ranks for s in suits]
    random.shuffle(deck)
    return deck

def card_value(c):
    order = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
             'T':10,'J':11,'Q':12,'K':13,'A':14}
    return order[c.rank]

def check_straight(vals):
    vals = sorted(set(vals))
    if len(vals) < 5:
        return False, None
    for i in range(len(vals)-4):
        window = vals[i:i+5]
        if window[-1] - window[0] == 4:
            return True, window[-1]
    if {14,2,3,4,5}.issubset(vals):
        return True, 5
    return False, None

def evaluate_5card_hand(cards):
    vals = sorted([card_value(c) for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    flush = len(set(suits)) == 1
    straight, high = check_straight(vals)
    freq = {v: vals.count(v) for v in vals}
    freq_sorted = sorted(freq.items(), key=lambda x:(x[1], x[0]), reverse=True)
    if flush and straight:
        return (8, high)
    if freq_sorted[0][1] == 4:
        four = freq_sorted[0][0]
        kick = max(v for v in vals if v != four)
        return (7, four, kick)
    if freq_sorted[0][1] == 3 and len(freq_sorted)>1 and freq_sorted[1][1]>=2:
        return (6, freq_sorted[0][0], freq_sorted[1][0])
    if flush:
        return (5, tuple(vals))
    if straight:
        return (4, high)
    if freq_sorted[0][1] == 3:
        kickers = sorted([v for v in vals if v != freq_sorted[0][0]], reverse=True)
        return (3, freq_sorted[0][0], tuple(kickers))
    if len(freq_sorted)>=2 and freq_sorted[0][1]==2 and freq_sorted[1][1]==2:
        p1,p2 = freq_sorted[0][0], freq_sorted[1][0]
        kick = max(v for v in vals if v not in (p1,p2))
        return (2, max(p1,p2), min(p1,p2), kick)
    if freq_sorted[0][1] == 2:
        kickers = sorted([v for v in vals if v != freq_sorted[0][0]], reverse=True)
        return (1, freq_sorted[0][0], tuple(kickers))
    return (0, tuple(vals))

def evaluate_hand(hole, community):
    best = (-1,)
    for combo in combinations(hole+community, 5):
        rank = evaluate_5card_hand(combo)
        if rank > best:
            best = rank
    return best

class MultiAgentHoldem:
    def __init__(self, num_players=4, starting_chips=100, small_blind=1, big_blind=2, max_raises=4):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises = max_raises
        self.reset()

    def reset(self):
        self.deck = create_deck()
        self.community_cards = []
        self.hands = [[self.deck.pop(), self.deck.pop()] for _ in range(self.num_players)]
        self.folded = [False] * self.num_players
        self.chips = [self.starting_chips for _ in range(self.num_players)]
        self.pot = 0
        self.stage = Stage.PREFLOP
        self.raises_this_round = 0
        self.round_actions = 0
        self.current_player = 0
        return self._get_observation(self.current_player)

    def _get_legal_moves(self, player_id):
        if self.folded[player_id]:
            return []
        moves = [Action.FOLD.value, Action.CALL.value]
        if self.raises_this_round < self.max_raises:
            moves.append(Action.RAISE.value)
        return moves

    def _get_observation(self, player_id):
        stage_oh = [0]*len(Stage)
        stage_oh[self.stage.value] = 1
        hole_vals = [card_value(c) for c in self.hands[player_id]]
        comm_len = len(self.community_cards)
        return {
            'stage': stage_oh,
            'hole': hole_vals,
            'pot': self.pot,
            'community_len': comm_len,
            'legal_moves': self._get_legal_moves(player_id)
        }

    def step(self, player_id, action):
        if player_id != self.current_player or self.folded[player_id]:
            return self._get_observation(player_id), 0, False, {'result': 'not_your_turn'}

        if action not in self._get_legal_moves(player_id):
            return self._get_observation(player_id), -1, False, {'result': 'illegal'}

        act = Action(action)
        reward, done, info = 0, False, {}

        if act == Action.FOLD:
            self.folded[player_id] = True
            info['result'] = 'fold'
        else:
            bet = self.small_blind if act == Action.CALL else self.big_blind
            self.pot += bet
            self.chips[player_id] -= bet
            if act == Action.RAISE:
                self.raises_this_round += 1

        self.round_actions += 1
        self._next_player()

        if self._all_acted():
            self.round_actions = 0
            self.raises_this_round = 0
            self._advance_stage()

        if self.stage == Stage.SHOWDOWN or self._only_one_left():
            done = True
            reward, info = self._settle_game()

        return self._get_observation(player_id), reward, done, info

    def _only_one_left(self):
        return sum(not f for f in self.folded) == 1

    def _next_player(self):
        while True:
            self.current_player = (self.current_player + 1) % self.num_players
            if not self.folded[self.current_player]:
                break

    def _all_acted(self):
        return self.round_actions >= self.num_players

    def _advance_stage(self):
        if self.stage == Stage.SHOWDOWN:
            return
        if self.stage == Stage.FLOP:
            self.community_cards += [self.deck.pop() for _ in range(3)]
        elif self.stage in [Stage.TURN, Stage.RIVER]:
            self.community_cards.append(self.deck.pop())
        self.stage = Stage(self.stage.value + 1)

    def _settle_game(self):
        alive = [i for i in range(self.num_players) if not self.folded[i]]
        scores = [(i, evaluate_hand(self.hands[i], self.community_cards)) for i in alive]
        scores.sort(key=lambda x: x[1], reverse=True)
        winner = scores[0][0]
        self.chips[winner] += self.pot
        return self.pot, {'result': 'win', 'winner': winner}

###########################################################################################

import random
import copy
import numpy as np
from itertools import combinations
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 你的 MultiAgentHoldem 和辅助函数都放在这里（略） ---

# --- 1. Random Agent ---
class RandomAgent:
    def select_action(self, obs):
        legal_moves = obs['legal_moves']
        return random.choice(legal_moves) if legal_moves else Action.FOLD.value

# --- 2. Simplified Tabular CFR Agent ---
class CFR_Agent:
    def __init__(self):
        self.regret_sum = defaultdict(lambda: np.zeros(3))  # 3 actions: FOLD, CALL, RAISE
        self.strategy_sum = defaultdict(lambda: np.zeros(3))

    def _get_strategy(self, infoset):
        regrets = self.regret_sum[infoset]
        positive_regrets = np.maximum(regrets, 0)
        normalizing_sum = positive_regrets.sum()
        if normalizing_sum > 0:
            return positive_regrets / normalizing_sum
        else:
            return np.array([1/3, 1/3, 1/3])  # Uniform random if no positive regret

    def select_action(self, obs):
        infoset = self._extract_infoset(obs)
        strategy = self._get_strategy(infoset)
        legal = obs['legal_moves']
        probs = np.array([strategy[a] if a in legal else 0 for a in range(3)])
        probs /= probs.sum()
        return np.random.choice(range(3), p=probs)

    def train(self, env, iterations=1):
        for _ in range(iterations):
            self._cfr(env)

    def _cfr(self, env):
        env_copy = copy.deepcopy(env)
        self._cfr_recursive(env_copy, player_id=0)

    def _cfr_recursive(self, env, player_id):
        if env.stage == Stage.SHOWDOWN or env._only_one_left():
            return self._get_payoff(env, player_id)

        current_player = env.current_player
        obs = env._get_observation(current_player)
        infoset = self._extract_infoset(obs)
        legal_moves = obs['legal_moves']

        strategy = self._get_strategy(infoset)

        util = np.zeros(3)
        node_utility = 0

        for a in legal_moves:
            env_copy = copy.deepcopy(env)
            obs_next, reward, done, info = env_copy.step(current_player, a)
            util[a] = -self._cfr_recursive(env_copy, player_id)
            node_utility += strategy[a] * util[a]

        for a in legal_moves:
            regret = util[a] - node_utility
            self.regret_sum[infoset][a] += regret
            self.strategy_sum[infoset][a] += strategy[a]

        return node_utility

    def _get_payoff(self, env, player_id):
        alive = [i for i in range(env.num_players) if not env.folded[i]]
        scores = [(i, evaluate_hand(env.hands[i], env.community_cards)) for i in alive]
        scores.sort(key=lambda x: x[1], reverse=True)
        winner = scores[0][0]
        return 1 if winner == player_id else -1

    def _extract_infoset(self, obs):
        # 只用 hole cards 和 stage 简化成 infoset
        return tuple(obs['hole'] + obs['stage'])

# --- 3. 训练和测试主循环 ---
if __name__ == "__main__":
    random_agents = [RandomAgent(), RandomAgent()]
    cfr_agent = CFR_Agent()
    agents = [cfr_agent] + random_agents  # player 0 是 CFR agent

    num_players = 3
    env = MultiAgentHoldem(num_players=num_players)

    total_rounds = 100
    cfr_training_iterations_per_game = 10

    win_counts = np.zeros(num_players)
    win_record = []

    for episode in range(total_rounds):
        obs = env.reset()
        done = False
        while not done:
            player = env.current_player
            obs = env._get_observation(player)
            action = agents[player].select_action(obs)
            obs, reward, done, info = env.step(player, action)
            if done and 'winner' in info:
                win_counts[info['winner']] += 1

        # After each round, train CFR agent
        cfr_agent.train(env, iterations=cfr_training_iterations_per_game)
        win_record.append(win_counts.copy())

        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}: Win counts = {win_counts}")

    # Plotting the win rates
    win_record = np.array(win_record)
    plt.figure(figsize=(10,6))
    for pid in range(num_players):
        plt.plot((win_record[:, pid] / (np.arange(1, total_rounds+1))), label=f"Player {pid} {'(CFR)' if pid == 0 else '(Random)'}")
    plt.xlabel("Game")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Evolution Over Games")
    plt.legend()
    plt.grid(True)
    plt.show()


# if __name__ == '__main__':
#     env = MultiAgentHoldem(num_players=3)
#     obs = env.reset()
#     done = False
#     while not done:
#         player = env.current_player
#         legal = obs['legal_moves']
#         action = random.choice(legal) if legal else Action.FOLD.value
#         obs, reward, done, info = env.step(player, action)
#         print(f"Player {player} Action: {Action(action).name} | Reward: {reward} | Info: {info}")

    
