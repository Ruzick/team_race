import numpy as np
import multiprocessing as mp
import subprocess as sp
from multiprocessing.pool import ThreadPool
import shlex
from pathlib import Path
import time

agents = ['geoffrey_agent', 'jurgen_agent',
          'yann_agent', 'yoshua_agent', 'AI', 'AI']

# Keep agent karts the same.  AI random choice in DATA
karts = [
    'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
    'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard',
    'suzanne', 'tux', 'wilber', 'xue'
]

BALL_LOC_BOUNDS = None
BALL_VEL_BOUNDS = None


def randomize_ball(location_bounds, velocity_bounds):
    pass


def generate_match(agent1, agent2, datapath):
    # get random location / ball velocity
    # ball_location, ball_velocity = randomize_ball()

    command = [
        f'python3 -m custom.Data',
        f'-o {str(datapath)}',
        # f'--ball_location {ball_location}',
        # f'--ball_velocity {ball_velocity}',
        agent1,
        agent2
    ]
    return ' '.join(command)


def get_path(agent1, agent2):
    datapath = Path(f'img_data/{agent1}_v_{agent2}/')
    datapath = datapath.resolve()
    return datapath


def call_process(a1, a2, d):

    print(f"Running: {a1} v {a2}")
    cmd = generate_match(a1, a2, d)
    cwd = str(Path.cwd())

    start_time = time.time()
    p = sp.Popen(shlex.split(cmd), stdout=sp.PIPE,
                 stderr=sp.PIPE, cwd=cwd)
    out, err = p.communicate()
    run_time = time.time() - start_time
    print(f"Finished: {a1} v {a2} in {run_time}")

    return (out, err)


def main(n_workers, seed):
    rng = np.random.default_rng(seed=seed)

    # get all match permutations
    combos = np.array(np.meshgrid(agents, agents)).T.reshape(-1, 2)

    # Generate unique directories
    from collections import Counter
    dirs = []
    dup_counter = Counter()
    for combo in combos:
        datapath = get_path(*combo)
        count = dup_counter[str(datapath)]
        dup_counter[str(datapath)] += 1
        if count > 0:
            new_stem = str(datapath.stem) + f'_{count}'
            datapath = datapath.with_stem(new_stem)
        dirs.append(str(datapath))
    dirs = np.array(dirs).reshape(-1, 1)
    combos = np.append(combos, dirs, axis=1)
    print(combos.shape)

    # Run each command as a subprocess
    results = []
    with ThreadPool(n_workers) as pool:
        for a1, a2, d in combos:
            Path(d).mkdir(exist_ok=True, parents=True)
            res = pool.apply_async(call_process, (a1, a2, d,))
            results.append(res)

        for res in results:
            # print(res)
            print(res.get())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--workers', type=int, default=mp.cpu_count())
    parser.add_argument('-s', '--seed', type=int, default=None)

    args = parser.parse_args()
    main(
        n_workers=args.workers,
        seed=args.seed,
    )
