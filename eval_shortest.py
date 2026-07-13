from os.path import join, exists
import os
import json
import csv
import pickle
from collections import defaultdict

import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset, TrajFastDataset_SimTime, TrajFastShortestDataset
from utils.argparser import get_argparser
import numpy as np
import random
import time
from models_seq.seq_models import Destroyer

from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# EM / PC evaluation utilities
# ============================================================

def normalize_path(path):
    """
    Convert path to tuple[int].
    Remove -1 padding only.
    0 is not removed because 0 can be a valid node id.
    """
    if isinstance(path, torch.Tensor):
        path = path.detach().cpu().tolist()
    elif isinstance(path, np.ndarray):
        path = path.tolist()

    if path is None:
        return tuple()

    if len(path) > 0 and isinstance(path[0], (list, tuple, np.ndarray)):
        if len(path) == 1:
            path = path[0]

    clean_path = []

    for x in path:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().item()
        if isinstance(x, np.generic):
            x = x.item()

        x = int(x)

        # padding만 제거
        if x == -1:
            continue

        clean_path.append(x)

    return tuple(clean_path)


def build_directed_adj_list_from_A(A):
    """
    Convert directed adjacency matrix A to adjacency list.

    A[u, v] == 1 means directed edge u -> v exists.
    """
    if isinstance(A, torch.Tensor):
        A_cpu = A.detach().cpu()
    elif isinstance(A, np.ndarray):
        A_cpu = torch.from_numpy(A)
    else:
        A_cpu = torch.tensor(A)

    A_cpu = A_cpu.bool()
    n = A_cpu.shape[0]

    adj = {}

    for u in range(n):
        neighbors = torch.nonzero(A_cpu[u], as_tuple=False).view(-1).tolist()
        adj[u] = [int(v) for v in neighbors]

    return adj


def check_directed_path_valid(path, adj):
    """
    Check whether every consecutive edge in path exists.

    path = [a, b, c]
    requires:
      a -> b
      b -> c
    """
    path = normalize_path(path)

    if len(path) < 2:
        return {
            "valid": False,
            "reason": "path_length_less_than_2",
            "invalid_edges": [],
            "invalid_nodes": list(path),
        }

    invalid_nodes = []
    invalid_edges = []

    for u, v in zip(path[:-1], path[1:]):
        if u not in adj:
            invalid_nodes.append(u)
            continue

        if v not in adj[u]:
            invalid_edges.append((u, v))

    valid = len(invalid_nodes) == 0 and len(invalid_edges) == 0

    return {
        "valid": valid,
        "reason": "ok" if valid else "invalid_node_or_edge",
        "invalid_edges": invalid_edges,
        "invalid_nodes": sorted(list(set(invalid_nodes))),
    }


def load_shortest_paths_for_em_pc(args):
    """
    Load shortest path data.

    Target:
      {shortest_data_path}/{d_name}_shrink_SP_{shortest_new_idx}
      {shortest_data_path}/{d_name}_shrink_SP_{shortest_new_idx}.pkl
    """
    shortest_data_path = args.shortest_data_path
    if shortest_data_path is None:
        shortest_data_path = args.path

    candidate_paths = [
        join(shortest_data_path, f"{args.d_name}_shrink_SP_{args.shortest_new_idx}"),
        join(shortest_data_path, f"{args.d_name}_shrink_SP_{args.shortest_new_idx}.pkl"),
    ]

    print("[load_shortest_paths_for_em_pc] Candidate files:")
    for p in candidate_paths:
        print("  ", p)

    target_path = None
    for p in candidate_paths:
        if exists(p):
            target_path = p
            break

    if target_path is None:
        raise FileNotFoundError(
            "Cannot find shortest path file. Tried:\n"
            + "\n".join(candidate_paths)
        )

    print(f"[load_shortest_paths_for_em_pc] Loading shortest paths from: {target_path}")

    try:
        with open(target_path, "rb") as f:
            shortest_paths = pickle.load(f)
    except Exception:
        shortest_paths = torch.load(target_path, map_location="cpu")

    shortest_paths = [normalize_path(p) for p in shortest_paths]
    shortest_paths = [p for p in shortest_paths if len(p) >= 2]

    if len(shortest_paths) == 0:
        raise ValueError(f"Shortest path file is empty after normalization: {target_path}")

    print(f"[load_shortest_paths_for_em_pc] Loaded {len(shortest_paths)} shortest paths")

    return shortest_paths


def build_od_to_shortest_lengths(shortest_paths):
    """
    Build OD -> shortest path node length set.

    OD = (start, end)
    """
    od_to_shortest_lengths = defaultdict(set)

    for p in shortest_paths:
        p = normalize_path(p)

        if len(p) < 2:
            continue

        start = p[0]
        end = p[-1]

        od_to_shortest_lengths[(start, end)].add(len(p))

    return od_to_shortest_lengths


def feasible_shorter_lengths_directed(start, end, gen_node_len, adj):
    """
    Check whether feasible directed paths exist with node length smaller than gen_node_len.

    Example:
      gen_node_len = 6

    This checks node lengths:
      2, 3, 4, 5

    It checks exact-length reachability without enumerating all paths.
    """
    if gen_node_len <= 2:
        return []

    feasible_lengths = []
    current_nodes = {start}

    # edge_len = 1 means node_len = 2
    # check node_len 2 ~ gen_node_len - 1
    for edge_len in range(1, gen_node_len - 1):
        next_nodes = set()

        for u in current_nodes:
            for v in adj.get(u, []):
                next_nodes.add(v)

        current_nodes = next_nodes
        node_len = edge_len + 1

        if end in current_nodes:
            feasible_lengths.append(node_len)

        if len(current_nodes) == 0:
            break

    return sorted(set(feasible_lengths))


def evaluate_em_pc(
    gen_paths,
    A,
    shortest_paths,
    save_dir,
    prefix,
):
    """
    Evaluate generated paths.

    Logic:
      1. Check directed graph validity.
      2. If invalid, save the case separately.
      3. If valid, check whether gen path length equals actual shortest path length.
         If same, EM = 1.
      4. Compute PC:
         rank = 1 + number of feasible shorter path lengths.
         PC = 1 / rank.
    """
    os.makedirs(save_dir, exist_ok=True)

    adj = build_directed_adj_list_from_A(A)
    od_to_shortest_lengths = build_od_to_shortest_lengths(shortest_paths)

    records = []
    invalid_cases = []

    valid_scores = []
    em_scores = []
    pc_scores = []

    for idx, raw_path in enumerate(gen_paths):
        path = normalize_path(raw_path)
        gen_len = len(path)

        if gen_len >= 2:
            start = path[0]
            end = path[-1]
            od = (start, end)
        else:
            start = None
            end = None
            od = None

        valid_result = check_directed_path_valid(path, adj)
        valid = int(valid_result["valid"])

        shortest_lengths_same_od = []
        shortest_len = None

        if od is not None and od in od_to_shortest_lengths:
            shortest_lengths_same_od = sorted(list(od_to_shortest_lengths[od]))
            shortest_len = min(shortest_lengths_same_od)

        em = 0
        pc = 0.0
        graph_rank = None
        shorter_feasible_lengths = []

        if valid == 0:
            invalid_cases.append({
                "idx": idx,
                "start": start,
                "end": end,
                "gen_path": list(path),
                "gen_len": gen_len,
                "reason": valid_result["reason"],
                "invalid_nodes": valid_result["invalid_nodes"],
                "invalid_edges": valid_result["invalid_edges"],
                "shortest_len": shortest_len,
                "shortest_lengths_same_od": shortest_lengths_same_od,
            })

        else:
            # EM: valid path이고, 같은 OD의 shortest path 길이와 같으면 EM = 1
            if shortest_len is not None and gen_len == shortest_len:
                em = 1

            # PC: gen path보다 짧은 feasible path length 개수 기반 rank 계산
            shorter_feasible_lengths = feasible_shorter_lengths_directed(
                start=start,
                end=end,
                gen_node_len=gen_len,
                adj=adj,
            )

            graph_rank = len(shorter_feasible_lengths) + 1
            pc = 1.0 / graph_rank

        records.append({
            "idx": idx,
            "start": start,
            "end": end,
            "gen_path": list(path),
            "gen_len": gen_len,

            "valid": valid,
            "invalid_nodes": valid_result["invalid_nodes"],
            "invalid_edges": valid_result["invalid_edges"],

            "shortest_len": shortest_len,
            "shortest_lengths_same_od": shortest_lengths_same_od,

            "em": em,

            "shorter_feasible_lengths": shorter_feasible_lengths,
            "num_shorter_feasible_lengths": len(shorter_feasible_lengths),
            "graph_rank": graph_rank,
            "pc": pc,
        })

        valid_scores.append(valid)
        em_scores.append(em)
        pc_scores.append(pc)

    num_paths = len(records)

    summary = {
        "prefix": prefix,
        "num_paths": num_paths,

        "num_valid": int(np.sum(valid_scores)) if num_paths > 0 else 0,
        "valid_rate": float(np.mean(valid_scores)) if num_paths > 0 else 0.0,

        "num_invalid": len(invalid_cases),
        "invalid_rate": float(len(invalid_cases) / num_paths) if num_paths > 0 else 0.0,

        "num_em": int(np.sum(em_scores)) if num_paths > 0 else 0,
        "em_score": float(np.mean(em_scores)) if num_paths > 0 else 0.0,

        "pc_score": float(np.mean(pc_scores)) if num_paths > 0 else 0.0,
    }

    summary_path = join(save_dir, f"{prefix}_em_pc_summary.json")
    records_path = join(save_dir, f"{prefix}_em_pc_records.csv")
    invalid_path = join(save_dir, f"{prefix}_invalid_cases.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(invalid_path, "w") as f:
        json.dump(invalid_cases, f, indent=2)

    fieldnames = [
        "idx",
        "start",
        "end",
        "gen_path",
        "gen_len",

        "valid",
        "invalid_nodes",
        "invalid_edges",

        "shortest_len",
        "shortest_lengths_same_od",

        "em",

        "shorter_feasible_lengths",
        "num_shorter_feasible_lengths",
        "graph_rank",
        "pc",
    ]

    with open(records_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            row = dict(r)
            row["gen_path"] = json.dumps(row["gen_path"])
            row["invalid_nodes"] = json.dumps(row["invalid_nodes"])
            row["invalid_edges"] = json.dumps(row["invalid_edges"])
            row["shortest_lengths_same_od"] = json.dumps(row["shortest_lengths_same_od"])
            row["shorter_feasible_lengths"] = json.dumps(row["shorter_feasible_lengths"])
            writer.writerow(row)

    print(f"\n[{prefix}] EM / PC evaluation done")
    print(f"[{prefix}] valid: {summary['num_valid']} / {summary['num_paths']} ({summary['valid_rate']:.6f})")
    print(f"[{prefix}] invalid: {summary['num_invalid']} / {summary['num_paths']} ({summary['invalid_rate']:.6f})")
    print(f"[{prefix}] EM: {summary['num_em']} / {summary['num_paths']} ({summary['em_score']:.6f})")
    print(f"[{prefix}] PC: {summary['pc_score']:.6f}")
    print(f"[{prefix}] summary: {summary_path}")
    print(f"[{prefix}] records: {records_path}")
    print(f"[{prefix}] invalid cases: {invalid_path}")

    return summary, records, invalid_cases

def success_arrival_rate(planned_paths, orig_paths):
    hit_cnt = 0
    for planned, orig in zip(planned_paths, orig_paths):
        if planned[-1] == orig[-1]:
            hit_cnt += 1
    return hit_cnt / len(planned_paths)


def print_path_stats(paths, prefix=""):
    if not paths:
        print(f"{prefix} path stats: avg_len=0.0000, unique_count=0, unique_ratio=0.0000")
        return

    path_lengths = [len(path) for path in paths]
    unique_paths = set(tuple(path) for path in paths)
    avg_len = float(np.mean(path_lengths))
    unique_ratio = len(unique_paths) / len(paths)

    print(
        f"{prefix} path stats: avg_len={avg_len:.4f}, "
        f"unique_count={len(unique_paths)}, unique_ratio={unique_ratio:.4f}"
    )

# ============================================================
# refinement 
# ============================================================
def refine(paths, dests, A, n_vertex):
    max_deg = A.long().sum(1).max()
    v_to_ord = dict()  # v : dict from v to ord
    val, ind = A.long().topk(max_deg, dim=1)
    for i in range(n_vertex):
        valid_ind = ind[i][val[i] != 0].cpu().tolist()
        v_to_ord[i] = dict(zip(valid_ind, list(range(len(valid_ind)))))

    # two things: 1) cut the recursive
    refined_paths = []
    for k, path in enumerate(paths):
        # 1) if one step close, directly cut
        destination = dests[k]
        for i, v in enumerate(path):
            if destination in v_to_ord[v]:
                cutted_path = path[:i+1] + [destination]    # org: path[:i] (Edit-JW)
                break
        else:
            cutted_path = path
        # 2) cut the recursive
        showup = set()
        points_filtered = []
        for _, v in enumerate(cutted_path):
            if v not in showup:
                showup.add(v)
                points_filtered.append(v)
            else:
                while points_filtered[-1] != v:
                    showup.discard(points_filtered[-1])
                    points_filtered.pop()
        refined_paths.append(points_filtered)
    return refined_paths

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    set_seed(args.seed)

    # set device
    if args.device == "default":
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(device)

    # set dataset
    if args.d_name == "":
        n_vertex = args.n_vertex
        name = f"v{args.n_vertex}_p{args.n_path}_{args.min_len}{args.max_len}"

        dataset = DataGenerator(
            args.n_vertex,
            args.n_path,
            args.min_len,
            args.max_len,
            device,
            args.path,
            name,
        )

    elif args.d_name != "":
        date = "20190701" if "dj" in args.d_name else "dj"

        # training에서 사용한 dataset_new와 동일한 dataset 사용
        dataset = TrajFastShortestDataset(
            args.d_name,
            [date],
            args.path,
            device,
            is_pretrain=True,
            shuffle=False,
            index=args.shortest_new_idx,
            shortest_data_path=args.shortest_data_path,
            gen_path=args.train_org_gen_path,
        )

        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")

    if args.method != "plan":
        model = torch.load(
            join(args.model_path, f"{args.model_name}.pth"),
            map_location=device,
        )

        disc = torch.load(
            join(args.disc_path, f"{args.disc_name}.pth"),
            map_location=device,
        )

        model.new_nll = False

        model.args = args
        model.device = device
        model.eps_model.device = device
        disc.device = device


        model = model.to(device)
        disc = disc.to(device)

        model.eval()
        disc.eval()

        # Default setting에서는 둘 다 False
        model.applying_mask_intermediate = args.applying_mask_intermediate
        model.applying_mask_intermediate_temperature = args.applying_mask_intermediate_temperature

        destroyer_new = None

        # real path sampling
        real_paths = dataset.get_real_paths(args.eval_num)

        # ------------------------------------------------------------
        # Load shortest paths for EM / PC
        # ------------------------------------------------------------
        if args.d_name == "":
            raise NotImplementedError(
                "Synthetic data case needs separate shortest path loading logic."
            )

        shortest_paths = load_shortest_paths_for_em_pc(args)

        os.makedirs(args.res_path, exist_ok=True)
        os.makedirs("./figs", exist_ok=True)

        print(f"# real_paths: {len(real_paths)}")
        print(f"# shortest_paths: {len(shortest_paths)}")

        # ============================================================
        # 1. Base diffusion generation
        # ============================================================
        base_start_time = time.time()

        base_gen_paths = model.sample(
                args.eval_num,
                args.batch_traj_num,
                real_paths=real_paths,
                bool_prefix=args.bool_prefix,
            )

        if args.use_refinement:
            base_gen_paths = refine(base_gen_paths, [p[-1] for p in real_paths], dataset.A, dataset.n_vertex)

        base_sampling_time = time.time() - base_start_time
        base_success_arrival_rate = success_arrival_rate(base_gen_paths, real_paths)
        
        # Unique paths
        unique_paths = set(tuple(path) for path in base_gen_paths)
        print(f"Ratio of unique paths: {len(unique_paths) / len(base_gen_paths):.4f}")    

        print(f"Base Success Arrival Rate: {base_success_arrival_rate:.4f}")
        print(f"Base sampling time: {base_sampling_time} seconds")
        print("Base Gen_path.", base_gen_paths[0])
        print("real_path.", real_paths[0])

        # torch.save(
        #     base_gen_paths,
        #     join(args.model_path, f"BASE_{args.model_name}_{args.save_name}_gen_paths.pth"),
        # )

        base_summary, base_records, base_invalid_cases = evaluate_em_pc(
            gen_paths=base_gen_paths,
            A=dataset.A,
            shortest_paths=shortest_paths,
            save_dir=args.res_path,
            prefix=f"BASE_{args.model_name}_{args.save_name}",
        )


        base_summary["sampling_time"] = base_sampling_time

        # ============================================================
        # 2. Discriminator-guided diffusion generation
        # ============================================================
        disc_start_time = time.time()

        gen_paths = model.sample_with_disc(
            args.eval_num,
            args.batch_traj_num,
            real_paths=real_paths,
            bool_prefix=args.bool_prefix,
            disc=disc,
            adj_matrix=dataset.A.to(device)
        )

        disc_sampling_time = time.time() - disc_start_time
        
        if args.use_refinement:
            gen_paths = refine(gen_paths, [p[-1] for p in real_paths], dataset.A, dataset.n_vertex)
        
        print(f"Disc sampling time: {disc_sampling_time} seconds")
        print("Disc Gen_path.", gen_paths[0])
        print("real_path.", real_paths[0])

        # torch.save(
        #     gen_paths,
        #     join(args.model_path, f"DISC_{args.model_name}_{args.save_name}_gen_paths.pth"),
        # )

        disc_summary, disc_records, disc_invalid_cases = evaluate_em_pc(
            gen_paths=gen_paths,
            A=dataset.A,
            shortest_paths=shortest_paths,
            save_dir=args.res_path,
            prefix=f"DISC_{args.model_name}_{args.save_name}",
        )

        disc_summary["sampling_time"] = disc_sampling_time

        # ============================================================
        # 3. Save final comparison
        # ============================================================
        compare_summary = {
            "base": base_summary,
            "disc": disc_summary,
        }

        compare_path = join(
            args.res_path,
            f"COMPARE_{args.model_name}_{args.save_name}_em_pc_summary.json",
        )

        with open(compare_path, "w") as f:
            json.dump(compare_summary, f, indent=2)

        line = (
            f"{args.model_name}_{args.save_name}: "
            f"BASE_TIME={base_sampling_time:.6f}, "
            f"BASE_VALID={base_summary['valid_rate']:.6f}, "
            f"BASE_EM={base_summary['em_score']:.6f}, "
            f"BASE_PC={base_summary['pc_score']:.6f}, "
            f"DISC_TIME={disc_sampling_time:.6f}, "
            f"DISC_VALID={disc_summary['valid_rate']:.6f}, "
            f"DISC_EM={disc_summary['em_score']:.6f}, "
            f"DISC_PC={disc_summary['pc_score']:.6f}\n"
        )

        with open("./figs/result_log.txt", "a") as f:
            f.write(line)

        print("\n========== FINAL RESULT ==========")
        print(
            f"BASE | "
            f"VALID={base_summary['valid_rate']:.6f}, "
            f"EM={base_summary['em_score']:.6f}, "
            f"PC={base_summary['pc_score']:.6f}, "
            f"TIME={base_sampling_time:.6f}"
        )
        print(
            f"DISC | "
            f"VALID={disc_summary['valid_rate']:.6f}, "
            f"EM={disc_summary['em_score']:.6f}, "
            f"PC={disc_summary['pc_score']:.6f}, "
            f"TIME={disc_sampling_time:.6f}"
        )
        print(f"Compare summary saved to: {compare_path}")

    # ------------------------------------------------------------
    # Method = Plan
    # ------------------------------------------------------------

    if args.method == "plan":
        from utils.evaluate_plan import Evaluator
        suffix = args.d_name
        model = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)
        evaluator = Evaluator(model, dataset)
        model.build_graph_structure(dataset.G, dataset.A)

        model.args = args
        model.restorer.args = args
        model.device = device
        model = model.to(device)

        model.eval()
    
        # ------------------------------------------------------------
        # Load shortest paths for EM / PC
        # ------------------------------------------------------------
        if args.d_name == "":
            raise NotImplementedError(
                "Synthetic data case needs separate shortest path loading logic."
            )

        shortest_paths = load_shortest_paths_for_em_pc(args)

        os.makedirs(args.res_path, exist_ok=True)
        os.makedirs("./figs", exist_ok=True)

        print(f"# shortest_paths: {len(shortest_paths)}")

        n_samples = args.eval_num
        set_seed(args.seed)
        choices = np.random.choice(len(dataset), n_samples, False).tolist()

        set_batch_size = min(200, n_samples)
        n_batch = (n_samples + set_batch_size - 1) // set_batch_size

        # Generation Starts
        real_paths = [dataset[choices[k]] for k in range(n_samples)]    # real path -> offers (org, dst)

        # ============================================================
        # 1. Base diffusion generation - from "evaluate_plan.py"
        # ============================================================
        base_start_time = time.time()
        base_gen_paths = []

        for i in tqdm(range(n_batch)):
            set_seed(args.seed + i)
            left, right = i * set_batch_size, min((i + 1) * set_batch_size, n_samples)
            origs = [dataset[choices[k]][0] for k in range(left, right)]                # real path 의 시작점
            dests = [dataset[choices[k]][-1] for k in range(left, right)]               # real path 의 도착점

            xs_list = model.plan(origs, dests, eval_nll=False, use_refine=True)

            base_gen_paths.extend(xs_list)

        base_sampling_time = time.time() - base_start_time
        base_success_arrival_rate = success_arrival_rate(base_gen_paths, real_paths)

        # Unique paths
        unique_paths = set(tuple(path) for path in base_gen_paths)
        print(f"Ratio of unique paths: {len(unique_paths) / len(base_gen_paths):.4f}")        

        print(f"Base Success Arrival Rate: {base_success_arrival_rate:.4f}")
        print(f"Base sampling time: {base_sampling_time} seconds")
        print("Base Gen_path.", base_gen_paths[0])
        print("real_path.", real_paths[0])


        # torch.save(
        #     base_gen_paths,
        #     join(args.model_path, f"BASE_{args.model_name}_{args.save_name}_gen_paths.pth"),
        # )

        base_summary, base_records, base_invalid_cases = evaluate_em_pc(
            gen_paths=base_gen_paths,
            A=dataset.A,
            shortest_paths=shortest_paths,
            save_dir=args.res_path,
            prefix=f"BASE_{args.model_name}_{args.save_name}",
        )

        base_summary["sampling_time"] = base_sampling_time
        base_summary["success_arrival_rate"] = base_success_arrival_rate

        # ============================================================
        # 2. Discriminator-guided diffusion generation
        # ============================================================
        disc_start_time = time.time()
        gen_paths = []

        for i in tqdm(range(n_batch)):
            set_seed(args.seed + i)
            left, right = i * set_batch_size, min((i + 1) * set_batch_size, n_samples)
            origs = [dataset[choices[k]][0] for k in range(left, right)]                # real path 의 시작점
            dests = [dataset[choices[k]][-1] for k in range(left, right)]               # real path 의 도착점

            # xs_list = model.plan_with_disc(origs, dests, disc, eval_nll=False, use_refine=True)
            xs_list = model.plan(origs, dests, eval_nll=False, use_refine=True, planning_mode=args.planning_mode)

            gen_paths.extend(xs_list)

        disc_sampling_time = time.time() - disc_start_time
        disc_success_arrival_rate = success_arrival_rate(gen_paths, real_paths)

        print_path_stats(gen_paths, prefix="Disc")
        print(f"Disc Success Arrival Rate: {disc_success_arrival_rate:.4f}")
        print(f"Disc sampling time: {disc_sampling_time} seconds")
        print("Disc Gen_path.", gen_paths[0])
        print("real_path.", real_paths[0])

        # torch.save(
        #     gen_paths,
        #     join(args.model_path, f"DISC_{args.model_name}_{args.save_name}_gen_paths.pth"),
        # )

        disc_summary, disc_records, disc_invalid_cases = evaluate_em_pc(
            gen_paths=gen_paths,
            A=dataset.A,
            shortest_paths=shortest_paths,
            save_dir=args.res_path,
            prefix=f"DISC_{args.model_name}_{args.save_name}",
        )

        disc_summary["sampling_time"] = disc_sampling_time
        disc_summary["success_arrival_rate"] = disc_success_arrival_rate

        # ============================================================
        # 3. Save final comparison
        # ============================================================
        line = (
            f"{args.model_name}_{args.save_name}: "
            f"BASE_TIME={base_sampling_time:.6f}, "
            f"BASE_VALID={base_summary['valid_rate']:.6f}, "
            f"BASE_EM={base_summary['em_score']:.6f}, "
            f"BASE_PC={base_summary['pc_score']:.6f}, "
            f"BASE_SUCCESS_RATE={base_summary['success_arrival_rate']:.6f}, "
            f"DISC_TIME={disc_sampling_time:.6f}, "
            f"DISC_VALID={disc_summary['valid_rate']:.6f}, "
            f"DISC_EM={disc_summary['em_score']:.6f}, "
            f"DISC_PC={disc_summary['pc_score']:.6f}, "
            f"DISC_SUCCESS_RATE={disc_summary['success_arrival_rate']:.6f}\n"
        )
        with open("./figs/planning_result_log.txt", "a") as f:
            f.write(line)

        print("\n========== FINAL RESULT ==========")
        print(
            f"BASE | "
            f"VALID={base_summary['valid_rate']:.6f}, "
            f"EM={base_summary['em_score']:.6f}, "
            f"PC={base_summary['pc_score']:.6f}, "
            f"TIME={base_sampling_time:.6f}, "
            f"SUCCESS_RATE={base_summary['success_arrival_rate']:.6f}"
        )
        print(
            f"DISC | "
            f"VALID={disc_summary['valid_rate']:.6f}, "
            f"EM={disc_summary['em_score']:.6f}, "
            f"PC={disc_summary['pc_score']:.6f}, "
            f"TIME={disc_sampling_time:.6f}, "
            f"SUCCESS_RATE={disc_summary['success_arrival_rate']:.6f}"
        )
 
 