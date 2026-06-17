import argparse, numpy as np
from scipy.stats import mannwhitneyu
from evolved_comm.config import Config
from evolved_comm import env
from evolved_comm.policies import oracle_policy

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=int, default=150)
parser.add_argument("--group", type=int, default=8)
parser.add_argument("--n", type=int, default=400)
args = parser.parse_args()

cfg = Config(episode_steps=args.episode, group_size=args.group)
rng = np.random.default_rng(1)

# Per-group total yield for the oracle, with signalling either on (mute=False)
# or fully disabled (mute=True) to isolate the channel's value.
oracle_yield = lambda mute: env.simulate(cfg, args.n, oracle_policy(cfg, mute=mute), rng).intake.sum(1)
signalling_yield, mute_yield = oracle_yield(False), oracle_yield(True)
p_value = mannwhitneyu(signalling_yield, mute_yield, alternative="greater").pvalue
print(f"signalling oracle yield: median {np.median(signalling_yield):.1f}  mean {signalling_yield.mean():.2f}")
print(f"mute oracle       yield: median {np.median(mute_yield):.1f}  mean {mute_yield.mean():.2f}")
print(f"lift = {100*(signalling_yield.mean()/mute_yield.mean()-1):+.1f}%   Mann-Whitney p = {p_value:.2e}")
