"""Parallel experiment runner.

Builds the full job matrix (controllers x regimes x alphas x channel x seeds),
runs jobs across CPU cores, writes one compressed .npz per job, logs live
progress to JSONL, and checkpoints long runs so they survive interruption.
Re-running skips completed jobs, so it is fully restartable.

Examples
--------
# the headline sweep on a workstation:
python run_experiments.py --controllers feedforward,recurrent \
    --regimes colony,individual,hybrid --alphas 0.5 --ablate \
    --seeds 15 --gens 800 --pop 300 --workers 12

# a quick test:
python run_experiments.py --controllers feedforward --regimes colony \
    --seeds 2 --gens 50 --pop 80 --workers 2
"""
from __future__ import annotations
import os, json, time, argparse
import numpy as np
from dataclasses import replace
from multiprocessing import Pool, cpu_count
from evolved_comm.config import Config
from evolved_comm import evolution


def tag_of(controller, regime, alpha, ablate, seed):
    """Build the unique filename stem identifying one experiment job."""
    alpha_tag = f"-a{alpha:g}" if regime == "hybrid" else ""
    channel_tag = "ablate" if ablate else "signal"
    return f"{controller}_{regime}{alpha_tag}_{channel_tag}_s{seed}"


def run_one(spec):
    """Run a single job (one cell of the matrix) and save its compressed result.

    ``spec`` is the packed tuple a worker receives. If the output ``.npz``
    already exists the job is skipped (caching), so the whole sweep is
    restartable. Returns a short status string for the progress log.
    """
    base_cfg, controller, regime, alpha, ablate, seed, outdir, mi_every = spec
    tag = tag_of(controller, regime, alpha, ablate, seed)
    result_path = os.path.join(outdir, tag + ".npz")
    if os.path.exists(result_path):
        return tag + " (cached)"
    cfg = replace(base_cfg, controller=controller, regime=regime, alpha=alpha,
                  ablate_signal=ablate, seed=seed)
    checkpoint_path = os.path.join(outdir, "ckpt_" + tag + ".pkl")
    log_path = os.path.join(outdir, tag + ".jsonl")

    def log_fn(gen, record):
        with open(log_path, "a") as f:
            f.write(json.dumps({"gen": gen, **record}) + "\n")

    start_time = time.time()
    hist, final = evolution.evolve(cfg, mi_every=mi_every, probe_groups=32,
                                   checkpoint_path=checkpoint_path, log_fn=log_fn)
    result = dict(controller=controller, regime=regime, alpha=alpha, ablate=ablate,
                  seed=seed, yield_curve=hist["yield"], best_fit=hist["best_fit"],
                  mi_gen=hist["mi_gen"], mi=hist["mi"], mi_null=hist["mi_null"],
                  final_mi=final["mi"], final_null=final["mi_null"],
                  final_yield=float(np.mean(hist["yield"][-10:])),
                  # saved for EVERY run so MI is re-derivable offline without re-simulating:
                  sm_signal=final["signal"], sm_food=final["food_state"],
                  best_genome=np.asarray(final["best_genome"], dtype=object))
    if seed == 0 and not ablate:   # bulky trajectories kept only for the signal-map figures
        result.update(sm_dist=final["dist_food"], sm_pos=final["pos"])
    np.savez_compressed(result_path, **result)
    try:
        os.remove(checkpoint_path)
    except OSError:
        pass
    return f"{tag}  finalMI={final['mi']:.3f}  yield={result['final_yield']:.1f}  ({time.time()-start_time:.0f}s)"


def main():
    """Parse CLI args, enumerate the job matrix, and run it across worker processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", default="feedforward,recurrent")
    parser.add_argument("--regimes", default="colony,individual,hybrid")
    parser.add_argument("--alphas", default="0.5", help="hybrid alpha values, comma-sep")
    parser.add_argument("--ablate", action="store_true", help="also run ablated controls")
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--gens", type=int, default=800)
    parser.add_argument("--pop", type=int, default=300)
    parser.add_argument("--episode", type=int, default=150)
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--eval_repeats", type=int, default=3)
    parser.add_argument("--mi_every", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_cfg = Config(pop_size=args.pop, generations=args.gens, episode_steps=args.episode,
                      group_size=args.group, eval_repeats=args.eval_repeats,
                      checkpoint_every=args.checkpoint_every)
    json.dump(base_cfg.to_dict(), open(os.path.join(args.outdir, "config.json"), "w"), indent=2)

    controllers = args.controllers.split(",")
    regimes = args.regimes.split(",")
    alphas = [float(x) for x in args.alphas.split(",")]
    channels = [False, True] if args.ablate else [False]

    jobs = []
    for controller in controllers:
        for regime in regimes:
            regime_alphas = alphas if regime == "hybrid" else [0.5]
            for alpha in regime_alphas:
                for ablate in channels:
                    for seed in range(args.seeds):
                        jobs.append((base_cfg, controller, regime, alpha, ablate, seed,
                                     args.outdir, args.mi_every))
    pending = [job for job in jobs if not os.path.exists(
        os.path.join(args.outdir, tag_of(job[1], job[2], job[3], job[4], job[5]) + ".npz"))]
    print(f"{len(jobs)} jobs ({len(jobs)-len(pending)} cached) | {args.workers} workers", flush=True)
    start_time = time.time()
    with Pool(args.workers) as pool:
        for done_count, status in enumerate(pool.imap_unordered(run_one, pending), 1):
            print(f"[{done_count}/{len(pending)}] {status} | elapsed {time.time()-start_time:.0f}s", flush=True)
    print(f"ALL DONE in {time.time()-start_time:.0f}s", flush=True)
    open(os.path.join(args.outdir, "DONE"), "w").write("ok")


if __name__ == "__main__":
    main()
