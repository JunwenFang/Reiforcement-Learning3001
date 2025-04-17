import random
import copy
import numpy as np
from itertools import combinations
from enum import Enum

###########################
# 枚举定义：动作与阶段
###########################
class Action(Enum):
    """
    定义可执行的动作：
    - FOLD: 弃牌
    - CALL: 跟注/看牌
    - RAISE: 加注
    """
    FOLD = 0
    CALL = 1
    RAISE = 2

class Stage(Enum):
    """
    定义游戏阶段，用于状态表示和阶段推进：
    - PREFLOP: 翻牌前
    - FLOP: 翻牌
    - TURN: 转牌
    - RIVER: 河牌
    - SHOWDOWN: 摊牌
    """
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

###########################
# 扑克牌评估函数
###########################
class Card:
    """
    表示一张扑克牌，包括点数和花色。
    """
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    def __repr__(self):
        return f"{self.rank}{self.suit}"


def create_deck():
    """
    生成并随机打乱一副标准 52 张扑克牌。
    返回列表形式的 Card 对象。
    """
    ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    suits = ['H','D','C','S']
    deck = [Card(r,s) for r in ranks for s in suits]
    random.shuffle(deck)
    return deck


def card_value(c):
    """
    将 Card.rank 转换为对应的数值，A 为最大 (14)。
    用于比较牌型大小。
    """
    order = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
             'T':10,'J':11,'Q':12,'K':13,'A':14}
    return order[c.rank]


def check_straight(vals):
    """
    检查给定点数列表中是否存在顺子。
    返回 (是否顺子, 顺子最高点数)
    """
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
    """
    评估 5 张牌的手牌强度，返回元组。
    """
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
    """
    从 7 张牌选最佳 5 张，返回评估结果。
    """
    best = (-1,)
    for combo in combinations(hole+community, 5):
        rank = evaluate_5card_hand(combo)
        if rank > best:
            best = rank
    return best

###########################
# 自定义环境：1v1 有限注 德州扑克（不依赖 gym）
###########################
class TexasHoldemEnv:
    """
    纯 Python 实现的 Heads-up Limit 德州扑克环境，
    不依赖任何外部框架，只用 reset/step 接口。
    """
    def __init__(self, starting_chips=100, small_blind=1, big_blind=2, max_raises=4):
        # 初始筹码与盲注设置
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_raises = max_raises
        # 内部状态
        self.reset()

    def reset(self):
        """
        重置环境状态：洗牌、发二人手牌、初始化彩池与阶段。
        返回初始观测向量。
        """
        self.deck = create_deck()
        self.community_cards = []
        self.pot = self.small_blind + self.big_blind
        self.stage = Stage.PREFLOP
        self.raises_this_round = 0
        self.round_actions = 0
        self.player_cards = [self.deck.pop(), self.deck.pop()]
        self.opponent_cards = [self.deck.pop(), self.deck.pop()]
        self.betting_sequence = []
        return self._get_observation()

    def _get_observation(self):
        """
        构建观测向量：
        [阶段 one-hot] + [彩池] + [玩家底牌点数] + [公共牌数]
        """
        stage_oh = [0]*len(Stage)
        stage_oh[self.stage.value] = 1
        hole_vals = [card_value(c) for c in self.player_cards]
        comm_len = len(self.community_cards)
        return stage_oh + [self.pot] + hole_vals + [comm_len]

    def _get_legal_moves(self):
        """
        计算合法动作：始终允许弃牌和跟注，未超加注上限可加注。
        """
        moves = [Action.FOLD.value, Action.CALL.value]
        if self.raises_this_round < self.max_raises:
            moves.append(Action.RAISE.value)
        return moves

    def step(self, action):
        """
        执行动作，推进环境。
        返回 (obs, reward, done, info)。
        info 包含 'result': 'win'/'lose'/'draw'/'fold'.
        """
        # 合法性检查
        if action not in self._get_legal_moves():
            return self._get_observation(), -1.0, False, {'result':'illegal'}
        act = Action(action)
        reward, done, info = 0, False, {}
        # 弃牌
        if act == Action.FOLD:
            reward, done, info['result'] = -self.big_blind, True, 'fold'
        else:
            # 跟注或加注
            bet = self.small_blind if act == Action.CALL else self.big_blind
            self.pot += bet
            if act == Action.RAISE:
                self.raises_this_round += 1
            self.round_actions += 1
            self.betting_sequence.append((self.stage.name, act.name))
            # 阶段推进
            if self.round_actions >= 2:
                self.round_actions = 0
                if self.stage == Stage.RIVER:
                    self.stage = Stage.SHOWDOWN
                else:
                    self.stage = Stage(self.stage.value + 1)
                self.raises_this_round = 0
            # 摊牌结算
            if self.stage == Stage.SHOWDOWN:
                pr = evaluate_hand(self.player_cards, self.community_cards)
                orank = evaluate_hand(self.opponent_cards, self.community_cards)
                if pr > orank:
                    reward, info['result'] = self.pot, 'win'
                elif pr < orank:
                    reward, info['result'] = -self.pot, 'lose'
                else:
                    info['result'] = 'draw'
                done = True
        # 发公共牌
        if not done and self.stage in (Stage.FLOP, Stage.TURN, Stage.RIVER) and self.round_actions == 0:
            if self.stage == Stage.FLOP:
                for _ in range(3): self.community_cards.append(self.deck.pop())
            else:
                self.community_cards.append(self.deck.pop())
        return self._get_observation(), reward, done, info

    def clone(self):
        """
        返回深拷贝，用于无副作用的模拟。
        """
        return copy.deepcopy(self)

if __name__ == '__main__':
    env = TexasHoldemEnv()
    obs = env.reset()
    print('Obs:', obs)
    obs, r, d, i = env.step(Action.CALL.value)