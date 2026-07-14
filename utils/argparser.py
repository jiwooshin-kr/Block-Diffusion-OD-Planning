import argparse


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="configure all the settings")
    
    # device config
    parser.add_argument("-device", type=str, help="device, [cpu, cuda]", default="default")
    parser.add_argument("-gpu", type=int, help="gpu no", default=0)
    
    # path config 
    parser.add_argument("-path", type=str, help="data path", default="./sets_data")
    parser.add_argument("-model_path", type=str, help="model path", default="./sets_model")
    parser.add_argument("-res_path", type=str, help="results path", default="./sets_res")

    # data config
    parser.add_argument("-d_name", type=str, help="real data name chengdu, xian", default="")
    parser.add_argument("-n_vertex", type=int, help="number of vertices", default=20)
    parser.add_argument("-n_path", type=int, help="number of path", default=8000)
    parser.add_argument("-min_len", type=int, help="path length lower bound", default=4)
    parser.add_argument("-max_len", type=int, help="path length upper bound", default=15)
    parser.add_argument("-sim_time", action="store_true", help="simulation time condition")

    # model config 
    parser.add_argument("-model_name", type=str, help="model name")
    parser.add_argument("-method", type=str, help="method: cold, naive")
    parser.add_argument("-plan", type=int, help="whether to plan, 1 plan, 0 no", default=0)
    parser.add_argument("-destroy", type=str, help="destroy manner: time, space")
    parser.add_argument("-beta_lb", type=float, help="beta lower bound", default=0.0001)
    parser.add_argument("-beta_ub", type=float, help="beta upper bound", default=0.1)
    parser.add_argument("-time_dim", type=int, help="dimension of pos emb", default=20)
    parser.add_argument("-max_T", type=int, help="max time step", default=10)
    parser.add_argument("-gmm_comp", type=int, help="gmm component number", default=4)
    parser.add_argument("-x_emb_dim", type=int, help="vertex embedding dim", default=50)
    parser.add_argument("-drop_cond", type=float, help="drop condition rate", default=0.3)
    parser.add_argument("-dims", type=str, help="temporal unet dims, finnaly multiply by n_groups", default="[100, 150, 200]")
    parser.add_argument("-hidden_dim", type=int, help="hidden dim for time and condition", default=20)
    parser.add_argument("-n_groups", type=int, help="number of groups for group normalization", default=8)
    
    # ngram model config 
    parser.add_argument("-n_gram", type=int, help="n gram", default=1)
    
    # hmm model config 
    parser.add_argument("-hidden_states", type=int, help="hidden states for hmm", default=10)
    
    # lstm model config 
    # hidden_size, num_layers
    parser.add_argument("-hidden_size", type=int, help="hidden size for lstm", default=30)
    parser.add_argument("-num_layers", type=int, help="num_layers for lstm", default=3)
    
    # training config
    parser.add_argument("-n_epoch", type=int, help="number of epoch", default=200)
    parser.add_argument("-bs", type=int, help="batch size", default=32)
    parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("-gmm_samples", type=int, help="gmm samples", default=3000)
    
    # eval config
    parser.add_argument("-eval_num", type=int, help="evaluation sample number, int", default=1000)
    parser.add_argument("-applying_mask_intermediate", action="store_true", help="Apply mask intermediate")
    parser.add_argument("-applying_mask_intermediate_temperature", action="store_true", help="applying_mask_intermediate_temperature")
    parser.add_argument("-save_name", type=str, help="save name", default="")

    parser.add_argument("-min_lat", type=float, help="min_lat", default=-1)
    parser.add_argument("-max_lat", type=float, help="max_lat", default=-1)
    parser.add_argument("-min_lng", type=float, help="min_lng", default=-1)
    parser.add_argument("-max_lng", type=float, help="max_lng", default=-1)

    parser.add_argument("-batch_traj_num", type=int, help="batch_traj_num", default=200)

    parser.add_argument("-disc_path", type=str, help="discriminator path", default="./sets_disc")
    parser.add_argument("-disc_name", type=str, help="discriminator name")
    parser.add_argument("-newA_file", type=str, help="adjacency matrix path", default="None")
    parser.add_argument("-shortest_org_idx", type=str, help="Shortest path index", default=999)
    parser.add_argument("-shortest_new_idx", type=str, help="Shortest path index", default=0)

    parser.add_argument("-shortest_data_path", type=str, help="shortest path", default="./shortest_path_data")

    parser.add_argument("-except_scenario", type=str, help="except_scenario", default=None)
    parser.add_argument("-reverse_weight", type=float, help="reverse_weight", default=1)

    parser.add_argument("-guidance_scale", type=float, help="guidance_scale", default=1.)
    parser.add_argument("-lam_ce", type=float, help="lam_ce", default=1.)
    parser.add_argument("-lam_con", type=float, help="lam_con", default=1.)
    parser.add_argument("-train_timestep_sampling", type=str, help="train_timestep_sampling", default="uniform")
    parser.add_argument("-beta_schedule", type=str, help="beta_schedule", default="uniform")
    parser.add_argument("-bool_prefix", help="bool_prefix", action="store_true")
    parser.add_argument("-only_nll", help="bool_prefix", action="store_true")
    parser.add_argument("-ret_org", help="ret_org", action="store_true")

    parser.add_argument("-train_org_gen_path", type=str, help="train_gen_path", default=None)
    parser.add_argument("-train_new_gen_path", type=str, help="train_gen_path", default=None)

    parser.add_argument("-save_step", type=int, help="save_step", default=1)

    # [Edit - Jiwoo]
    parser.add_argument("-pretrained_model", type=str, help="model name", default=None)
    parser.add_argument("-seed", type=int, help="seed numnber", default=1)
    parser.add_argument("-save_iter", type=int, help="save_iter", default=0)
    parser.add_argument("-n_importance_samples", type=int, help="number of importance samples", default=10)
    parser.add_argument("-except_nll", help="except_nll", action="store_true")
    parser.add_argument("-use_refinement", help="use_refinement", action="store_true")
    parser.add_argument("-use_gnn", help="use_gnn", action="store_true")
    parser.add_argument("-use_logit_reg", help="use logit regularization", action="store_true")
    parser.add_argument("-planning_mode", type=str, help="planning mode", default="hybrid")
    parser.add_argument("-data_subset_ratio", type=float, help="data ratio", default=1.0)
    parser.add_argument("-finetune_diffusion", help="finetune_diffusion", action="store_true")

    # [Edit - Jiwoo] O/D conditional generation
    parser.add_argument("-od_cond", help="O/D conditional diffusion (EPSM_OD)", action="store_true")
    parser.add_argument("-od_dropout", type=float, help="independent dropout rate for O and D conditions", default=0.1)

    # [Edit - Jiwoo] LLaDA-style full-canvas <eos> length handling
    parser.add_argument("-eos_mode", help="fixed canvas + <end> state predicted by the model (no length model)", action="store_true")
    parser.add_argument("-eos_deg", type=float, help="total degree d of the virtual <end> state in the CTMC", default=0.05)
    parser.add_argument("-eos_canvas_len", type=int, help="fixed canvas length L_max", default=64)

    # [Edit - Jiwoo] conditional-generation ablations
    parser.add_argument("-clean_prefix", help="A1: keep condition tokens clean (un-noised) during training and sampling", action="store_true")
    parser.add_argument("-eos_loss_weight", type=float, help="A2: loss weight on <end>-target positions (1.0 = unweighted)", default=1.0)
    parser.add_argument("-dst_token", help="A3: canvas = [dst, ori, v1, ...] with the destination as in-context token", action="store_true")

    # [Edit - Jiwoo] A4: destination-matching losses
    parser.add_argument("-dst_loss_weight", type=float, help="A4a: loss weight on the true endpoint position (1.0 = unweighted)", default=1.0)
    parser.add_argument("-lam_arr", type=float, help="A4b: weight of the mean-field arrival loss -log sum_l p_l(D) p_{l+1}(<end>)", default=0.0)

    # [Edit - Jiwoo] block diffusion (BD3-LM style)
    parser.add_argument("-kernel", type=str, help="within-block noising kernel: mask | graph", default="graph")
    parser.add_argument("-block_size", type=int, help="tokens per diffusion block", default=4)
    parser.add_argument("-bd_hidden_dim", type=int, help="BD transformer hidden dim", default=256)
    parser.add_argument("-bd_n_layers", type=int, help="BD transformer layers", default=6)
    parser.add_argument("-bd_n_heads", type=int, help="BD transformer attention heads", default=8)
    parser.add_argument("-bd_cond_dim", type=int, help="BD conditioning dim (time + OD)", default=128)
    parser.add_argument("-bd_dropout", type=float, help="BD transformer dropout", default=0.1)
    parser.add_argument("-bd_max_len", type=int, help="canvas cap (block multiple); 0 = od_max_len + 2", default=0)
    parser.add_argument("-od_max_len", type=int, help="max generated path length", default=100)
    parser.add_argument("-bd_time_cond", help="time conditioning for the mask kernel (graph kernel: always on)", action="store_true")
    parser.add_argument("-bd_eos_deg", type=float, help="total degree d of the virtual <end> CTMC state (graph kernel)", default=0.05)
    parser.add_argument("-length_mode", type=str, help="open | oracle | both (oracle only sets the block budget)", default="open")

    # [Edit - Jiwoo] respaced (strided) sampling: fewer reverse steps via exact composed CTMC kernels
    parser.add_argument("-sample_steps", type=int, help="number of reverse steps at sampling (0 = full max_T)", default=0)

    return parser