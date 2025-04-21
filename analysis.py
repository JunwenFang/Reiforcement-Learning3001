import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_DIR = "log"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# === 日志文件查找 === #
log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log*")))
print(">>> Found log files:", log_files)
if not log_files:
    raise RuntimeError("❌ No log files found. Please check LOG_DIR and filename pattern.")

# === 正则表达式模式 === #
win_pat    = re.compile(r"Player (\d+) won")
stack_pat  = re.compile(r"Player (\d+) got \[.*?\] and \$([0-9]+(?:\.[0-9]+)?)")
action_pat = re.compile(r"Seat \d+ \(([^)]+)\): Action\.([A-Z_]+)")

# === 数据存储结构 === #
win_counts    = defaultdict(int)
stack_hist    = defaultdict(list)
action_counts = defaultdict(lambda: defaultdict(int))

# === 工具函数 === #
def calc_max_drawdown(series):
    peak = series[0]
    max_dd = 0
    for x in series:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    return max_dd

def sanitize(name: str) -> str:
    return re.sub(r'[\\/]', '_', name)
winners = []

hand_actions = defaultdict(list)  # hand_index → [(agent, action)]

hand_index = 0

# === read log === #
for path in log_files:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if m := win_pat.search(line):
                win_counts[int(m.group(1))] += 1
            if m := stack_pat.search(line):
                pid, stk = int(m.group(1)), float(m.group(2))
                stack_hist[pid].append(stk)
            if m := action_pat.search(line):
                agent, act = m.group(1), m.group(2)
                action_counts[agent][act] += 1
                hand_actions[hand_index].append((agent, act))
            if win_pat.search(line):
                hand_index += 1
            if m := win_pat.search(line):
                winners.append(int(m.group(1)))
            

# === 胜率 & 筹码统计 === #
total_hands = sum(win_counts.values())
print(f"Total hands parsed: {total_hands}")

records = []
for pid, wins in win_counts.items():
    s_list = stack_hist.get(pid, [])
    stats = {
        "Player ID":    pid,
        "Wins":         wins,
        "Win Rate":     wins / total_hands if total_hands else 0,
        "Stack Start":  s_list[0] if s_list else np.nan,
        "Stack End":    s_list[-1] if s_list else np.nan,
        "Mean Stack":   np.mean(s_list) if s_list else np.nan,
        "Std Stack":    np.std(s_list) if s_list else np.nan,
        "Max Drawdown": calc_max_drawdown(s_list) if s_list else np.nan
    }
    records.append(stats)

df_summary = pd.DataFrame(records)
summary_path = os.path.join(OUT_DIR, "agent_summary.csv")
float_cols = df_summary.select_dtypes(include='float').columns
df_summary[float_cols] = df_summary[float_cols].applymap(lambda x: round(x, 2) if pd.notnull(x) else x)

summary_path = os.path.join(OUT_DIR, "agent_summary.csv")
df_summary.to_csv(summary_path, index=False)
print(f"agent summary saved to {summary_path}.")

action_rate = {
    agent: {act: count / sum(act_counts.values()) for act, count in act_counts.items()}
    for agent, act_counts in action_counts.items()
}
print("Action frequency calculated.")

# === action_rate === #
for agent, rates in action_rate.items():
    safe_name = sanitize(agent)
    plt.figure()
    actions, freqs = zip(*rates.items())
    plt.bar(actions, freqs)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.title(f"{agent} Action Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{safe_name}_action_dist.png"))
    plt.close()

# === Visualization 2: Chip Curve === #
plt.figure()
for pid, s_list in stack_hist.items():
    plt.plot(s_list, label=f"P{pid}")
plt.xlabel("Hand Number")
plt.ylabel("Stack ($)")
plt.title("Stack Over Hands")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stack_trend.png"))
plt.show()
plt.close()


# === early vs late === #
split_index = len(winners) // 2 

print(hand_actions[0] )
action_counts_early = defaultdict(lambda: defaultdict(int))
action_counts_late  = defaultdict(lambda: defaultdict(int))
early = list(hand_actions.items())[:split_index]
late  = list(hand_actions.items())[split_index:]

for _, actions in early:
    for agent, act in actions:
        action_counts_early[agent][act] += 1

for _, actions in late:
    for agent, act in actions:
        action_counts_late[agent][act] += 1

print(action_counts_early)
for agent in action_counts_early:
    print(f"Generating action compare plot for agent: {agent}")

    acts_early = action_counts_early[agent]
    acts_late  = action_counts_late.get(agent, {})

    all_actions = sorted(set(acts_early) | set(acts_late))
    
    early_vals = [acts_early.get(a, 0) for a in all_actions]
    late_vals  = [acts_late.get(a, 0) for a in all_actions]

    total_early = sum(early_vals)
    total_late  = sum(late_vals)

    early_freq = [v / total_early if total_early else 0 for v in early_vals]
    late_freq  = [v / total_late  if total_late  else 0 for v in late_vals]

    x = np.arange(len(all_actions))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, early_freq, width, label='Early', alpha=0.7)
    plt.bar(x + width/2, late_freq,  width, label='Late',  alpha=0.7)
    plt.xticks(x, all_actions, rotation=45)
    plt.ylabel("Frequency")
    plt.title(f"{agent} Action Distribution (Early vs Late)")
    plt.legend()
    plt.tight_layout()

    safe_name = agent.replace("/", "_").replace("\\", "_")
    plt.savefig(os.path.join(OUT_DIR, f"{safe_name}_action_compare.png"))
    plt.show()

    plt.close()

# === learning_curve === #
if not winners:
    raise RuntimeError("❌ No winner data found in logs.")

# 设置滑动窗口
window_size = 200
rolling_win_rates = defaultdict(list)

# 玩家 ID 列表（可根据实际对局添加更多）
players = [0, 1, 2]

# 计算每个玩家的滑动胜率
for pid in players:
    for i in range(len(winners) - window_size + 1):
        window = winners[i:i + window_size]
        rate = window.count(pid) / window_size
        rolling_win_rates[pid].append(rate)

plt.figure(figsize=(10, 6))
for pid in players:
    plt.plot(rolling_win_rates[pid], label=f"Player {pid}")

plt.xlabel("Hand # (x1)")
plt.ylabel("Win Rate (Rolling Avg)")
plt.title("Player Win Rate over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learning_curve.png"))
plt.show()
print("✅ Saved learning_curve.png to output/")
plt.close()



# === Threshold Compliance === #
# T = 0.5  # 设置你的 threshold
# with open("log/default_info.log") as f:
#     for line in f:
#         if m := equity_act_pat.search(line):
#             equity, action = float(m.group(1)), m.group(2)
#             if equity >= T:
#                 threshold_stats["above_T"]["total"] += 1
#                 if "RAISE" in action:
#                     threshold_stats["above_T"]["raised"] += 1
#             else:
#                 threshold_stats["below_T"]["total"] += 1
#                 if "FOLD" in action:
#                     threshold_stats["below_T"]["folded"] += 1

# # 输出命中率
# rate_raise = threshold_stats["above_T"]["raised"] / threshold_stats["above_T"]["total"]
# rate_fold  = threshold_stats["below_T"]["folded"] / threshold_stats["below_T"]["total"]
# print(f"Raise accuracy when equity ≥ T: {rate_raise:.2f}")
# print(f"Fold accuracy when equity < T: {rate_fold:.2f}")










