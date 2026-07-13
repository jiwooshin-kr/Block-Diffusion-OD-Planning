from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset , TrajFastDataset_SimTime, TrajFastShortestDataset
from utils.argparser import get_argparser
from utils.evaluate import Evaluator
import numpy as np
import random
import time
from models_seq.seq_models import Destroyer

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = get_argparser()
    args = parser.parse_args()

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
        dataset = DataGenerator(args.n_vertex, args.n_path, args.min_len, args.max_len, device, args.path, name)
    elif args.d_name != "":
        date = None
        dataset = TrajFastShortestDataset(args.d_name, [date], args.path, device, is_pretrain=True, shuffle=False, index=args.shortest_new_idx, shortest_data_path=args.shortest_data_path)
        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")

    if args.method != "plan":
        model = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)
        disc = torch.load(join(args.disc_path, f"{args.disc_name}.pth"), map_location=device)

        model.args = args

        model.device = device
        model.eps_model.device = device
        disc.device = device
        # TODO: need to change device for all modules in model
        model.eval()
        disc.eval()

        real_paths = dataset.get_real_paths(args.eval_num)

        if args.only_nll:
            evaluator = Evaluator(real_paths, real_paths, model, n_vertex, dataset=dataset,
                                  name=join(args.res_path, f"DISC_{args.model_name}_{args.save_name}_pure_gen"),
                                  sim_time=args.sim_time)
            res = evaluator.calculate_nll(disc=disc, adj_matrix=dataset.A.to(device))
            print(res)

        else:
            start_time = time.time()
            gen_paths = model.sample_with_disc(args.eval_num, args.batch_traj_num, 
                                        real_paths=real_paths, bool_prefix=args.bool_prefix,
                                        disc=disc, adj_matrix=dataset.A.to(device))

            print(f'Sampling time: {time.time() - start_time} seconds')
            line = f"{args.model_name}_{args.save_name}: {time.time() - start_time}\n"

            with open("./figs/result_log.txt", "a") as f:
                f.write(line)

            torch.save(gen_paths, join(args.model_path, f"DISC_{args.model_name}_{args.save_name}_gen_paths.pth"))
            evaluator = Evaluator(real_paths, gen_paths, model, n_vertex, dataset=dataset,
                                  name=join(args.res_path, f"DISC_{args.model_name}_{args.save_name}_pure_gen"), sim_time=args.sim_time)
            evaluator.eval(suffix=f"DISC_{args.model_name}_{args.save_name}")
                       
            # Evaluate except NLL !!
            if args.except_nll:
                print("!! Evaluating except NLL !!")
                res = evaluator.eval_except_nll(disc=disc)
            else:
                print("!! Evaluating all metrics !!")
                res = evaluator.eval_all(disc=disc, adj_matrix=dataset.A.to(device))

            print(res)
            with open(join(args.res_path, f"DISC_{args.model_name}_{args.save_name}.res"), "w") as f:
                f.writelines(str(res))

    if args.method == "plan":
        raise NotImplementedError
