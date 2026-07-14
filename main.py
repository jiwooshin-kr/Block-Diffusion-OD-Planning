from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset, TrajFastDataset_SimTime, TrajFastShortestDataset
from utils.argparser import get_argparser
from utils.evaluate import Evaluator

import time

if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
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
    elif 'shortest' in args.model_name:
        dataset = TrajFastShortestDataset(args.d_name, None, args.path, device, is_pretrain=True, index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path)
        print("Shortest exp!")
    elif args.d_name != "":
        date = "20190701" if "dj" in args.d_name else "dj"
        # if args.sim_time == True:
        #     dataset = TrajFastDataset_SimTime(args.d_name, [date], args.path, device, is_pretrain=True)
        # elif args.sim_time == False:
        #     dataset = TrajFastDataset(args.d_name, [date], args.path, device, is_pretrain=True)
        dataset = TrajFastShortestDataset(args.d_name, [date], args.path, device, is_pretrain=True, index=args.shortest_org_idx, shortest_data_path=args.shortest_data_path)

        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")

    # before train, record the infob
    with open(join(args.model_path, f"{args.model_name}.info"), "w") as f:
        f.writelines(str(args))

    # set model
    if args.method == "seq":
        from models_seq.seq_models import Destroyer, Restorer
        from models_seq.eps_models import EPSM, EPSM_SimTime, EPSM_OD, EPSM_OD_EOS
        from models_seq.trainer import Trainer

        suffix = args.d_name

        if args.beta_schedule == 'uniform':
            betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
        elif args.beta_schedule == 'front':
            uuu = torch.linspace(0, 1, args.max_T)
            kkk = 2.0
            sss = (torch.exp(kkk * uuu) - 1) / (torch.exp(kkk * torch.ones_like(uuu)) - 1)
            betas = args.beta_lb + (args.beta_ub - args.beta_lb) * sss
        else:
            raise NotImplementedError
        if args.eos_mode:
            # LLaDA-style: augment the CTMC state space with one virtual <end>
            # state (index V) with total degree eos_deg, split evenly across all
            # vertices. Symmetric => the uniform limiting distribution over V+1
            # states is preserved. The self-loop keeps <end> -> <end> legal in
            # the binarized decode / connectivity loss (no effect on the CTMC).
            V = dataset.n_vertex
            w = args.eos_deg / V
            A_aug = torch.zeros(V + 1, V + 1)
            A_aug[:V, :V] = dataset.A.cpu().float()
            A_aug[:V, V] = w
            A_aug[V, :V] = w
            A_aug[V, V] = w
            destroyer = Destroyer(A_aug, betas, args.max_T, device)
        else:
            destroyer = Destroyer(dataset.A, betas, args.max_T, device)
        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
        dims = eval(args.dims)

        ######################################################## unconditional / O,D-conditional EPSM ##################################################
        if not args.sim_time:
            if args.od_cond and args.eos_mode:
                print(f"O/D conditional EPSM with <end> state (EPSM_OD_EOS), eos_deg={args.eos_deg}, canvas={args.eos_canvas_len}")
                eps_model = EPSM_OD_EOS(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
                                hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
            elif args.od_cond:
                print("O/D conditional EPSM (EPSM_OD)")
                eps_model = EPSM_OD(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
                                hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
            else:
                eps_model = EPSM(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
                                hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
            model = Restorer(eps_model, destroyer, device, args)
            trainer = Trainer(model, dataset, args.model_path, args.model_name, args=args)
        ##################################################################################################################################################

        # Model Size ==================================================================================
        total = sum(p.numel() * p.element_size() for p in eps_model.parameters())
        emb = eps_model.x_embedding.weight.numel() * eps_model.x_embedding.weight.element_size()

        total_p = sum(p.numel() for p in eps_model.parameters())
        emb_p = eps_model.x_embedding.weight.numel()

        print("============================================================")
        print(f"EPSM: {total/1024**2:.2f} MB ({total_p/1e6:.2f}M params)")
        print(f"EPSM w/o Node2Vec: {(total-emb)/1024**2:.2f} MB ({(total_p-emb_p)/1e6:.2f}M params)")
        print("============================================================")
        # Model Size ==================================================================================

        ############################################ simulation-time conditioned EPSM ####################################################################
        # elif args.sim_time:
        #     eps_model = EPSM_SimTime(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device,
        #                     hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
        #
        #     model = Restorer_SimTime(eps_model, destroyer, device)
        #
        #     trainer = Trainer_SimTime(model, dataset, args.model_path, args.model_name)
        ##################################################################################################################################################

        trainer.train_gmm(gmm_samples=args.gmm_samples, n_comp=args.gmm_comp)

        if args.min_lat != -1:
            remove_region = [[[args.min_lat, args.max_lat], [args.min_lng, args.max_lng]]]
        else:
            remove_region = None

        # Train Time ===========================================================================
        torch.cuda.synchronize()
        start = time.time()

        trainer.train(args.n_epoch, args.bs, args.lr, remove_region=remove_region)
    
        torch.cuda.synchronize()
        end = time.time()
        print("============================================================")
        print(f"Training time: {(end-start)/60:.2f} min")
        print("============================================================")
        # Train Time ===========================================================================

        model.eval()
        torch.save(model, join(args.model_path, f"{args.model_name}.pth"))
        model.eval()

    elif args.method == "plan":
        from planner.planner import Planner
        from planner.trainer import Trainer

        suffix = args.d_name

        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
         # [Edit - Jiwoo] ----------------------------------------------
         # restorer = torch.load(f"./sets_model/no_plan_gen_{suffix}.pth")
        restorer = torch.load(join(args.model_path, f"{args.model_name}.pth"), map_location=device)

        if not args.finetune_diffusion:
            print("Freezing restorer parameters...")
            for param in restorer.parameters():
                param.requires_grad = False
         # [Edit - Jiwoo] ----------------------------------------------

        destroyer = restorer.destroyer
        model = Planner(dataset.G, dataset.A, restorer, destroyer, device, x_emb_dim=args.x_emb_dim,
                        pretrain_path=pretrain_path)

        # Model Size ==================================================================================
        total = sum(p.numel() * p.element_size() for p in model.parameters())
        emb = model.x_embedding.weight.numel() * model.x_embedding.weight.element_size()

        total_p = sum(p.numel() for p in model.parameters())
        emb_p = model.x_embedding.weight.numel()

        rest_total = sum(p.numel() * p.element_size() for p in model.restorer.parameters())
        rest_total_p = sum(p.numel() for p in model.restorer.parameters())

        planner_size = total - rest_total
        planner_params = total_p - rest_total_p

        print("============================================================")
        print(f"Planner: {planner_size/1024**2:.2f} MB ({planner_params/1e6:.2f}M params)")
        print(f"Planner w/o Node2Vec: {(planner_size-emb)/1024**2:.2f} MB ({(planner_params-emb_p)/1e6:.2f}M params)")
        print("============================================================")
        # Model Size ==================================================================================

        trainer = Trainer(model, dataset, device, args.model_path)

        # Train Time ===========================================================================
        torch.cuda.synchronize()
        start = time.time()

        trainer.train(args.n_epoch, args.bs, args.lr)
    
        torch.cuda.synchronize()
        end = time.time()
        print("============================================================")
        print(f"Training time: {(end-start)/60:.2f} min")
        print("============================================================")
        # Train Time ===========================================================================

        model.eval()
        # [Edit - Jiwoo] Change model name ---------------------------------------------
        # torch.save(model, join(args.model_path, f"{args.model_name}_plan.pth"))
        torch.save(model, join(args.model_path, f"{args.model_name}_plan_{args.shortest_org_idx}_{args.lr}.pth"))
        # [Edit - Jiwoo] ----------------------------------------------------------------

    if args.method != "plan":
        real_paths = dataset.get_real_paths(args.eval_num)
        if args.od_cond:
            # O/D conditional generation: real (O, D, length) from test paths,
            # O clamped as prefix during sampling, D given only as condition (soft)
            real_paths = sorted(real_paths, key=len)
            gen_paths = model.sample(args.eval_num, real_paths=real_paths, bool_prefix=True, bool_od=True)
            n_gen = max(len(gen_paths), 1)
            o_match = sum(1 for g, r in zip(gen_paths, real_paths) if len(g) > 0 and g[0] == r[0])
            d_reach = sum(1 for g, r in zip(gen_paths, real_paths) if len(g) > 0 and g[-1] == r[-1])
            print("============================================================")
            print(f"O match rate: {o_match / n_gen:.4f}, D reach rate: {d_reach / n_gen:.4f} (n={len(gen_paths)})")
            print("============================================================")
        else:
            gen_paths = model.sample(args.eval_num)
        torch.save(gen_paths, join(args.model_path, "gen_paths.pth"))
        evaluator = Evaluator(real_paths, gen_paths, model, n_vertex, dataset=dataset,
                              name=join(args.res_path, f"{args.model_name}_pure_gen"), sim_time = args.sim_time)
        evaluator.eval(suffix=args.d_name)
        res = evaluator.eval_all()
        print(res)
        with open(join(args.res_path, f"{args.model_name}.res"), "w") as f:
            f.writelines(str(res))

    if args.method == "plan":
        from utils.evaluate_plan import Evaluator

        suffix = args.d_name
        evaluator = Evaluator(model, dataset)
        evaluator.eval(args.eval_num, suffix)