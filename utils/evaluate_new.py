from typing import Any
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from utils.visual import draw_heatmap, draw_paths
from scipy.special import rel_entr

class Evaluator:
    def __init__(self, real_paths, gen_paths, model, n_vertex, dataset, name="e1", sim_time=False, A=None, removal=None) -> None:
        self.real_paths = real_paths
        self.gen_paths = gen_paths
        self.n_vertex = n_vertex
        self.model = model
        self.name = name
        self.dataset = dataset
        self.sim_time = sim_time
        self.A = A
        self.removal = removal

    @staticmethod
    def JS_divergence(p, q):
        M = (p + q)/2
        return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    
    @staticmethod
    def KL_divergence(p,q):
        return scipy.stats.entropy(p, q)

    # Added - Jiwoo
    def calculate_rank_correlation(self):
        # 1. Counts the number of times each edge appears in real and generated paths
        real_edge_cnt = np.zeros((self.n_vertex, self.n_vertex))
        gen_edge_cnt = np.zeros((self.n_vertex, self.n_vertex))
        
        for path in self.real_paths:
            if self.sim_time:
                path = path[0]
            for a, b in zip(path, path[1:]):
                real_edge_cnt[a][b] += 1
            
        for path in self.gen_paths:
            for a, b in zip(path, path[1:]):
                gen_edge_cnt[a][b] += 1

        r = real_edge_cnt.reshape(-1)
        g = gen_edge_cnt.reshape(-1)

        print(f"Sum of real edge counts: {r.sum()}, Sum of generated edge counts: {g.sum()}")

        # 2. Calculate rank correlation metrics
        # <Note: Ranking is done internally by scipy, so we can directly use the counts without manual ranking.>

        # 2.1. Masking
        mask_nonzero_real = (r > 0)

        # 2.2. Spearman's rank correlation
        spearman_corr, _ = scipy.stats.spearmanr(r, g)
        spearman_corr_masked_real, _ = scipy.stats.spearmanr(r[mask_nonzero_real], g[mask_nonzero_real])
        print(f"[Spearman's rho] Entire: {spearman_corr:.4f}, Masked Real: {spearman_corr_masked_real:.4f}")

        # 2.2. Kendall's tau
        kendall_corr, _ = scipy.stats.kendalltau(r, g)
        kendall_corr_masked_real, _ = scipy.stats.kendalltau(r[mask_nonzero_real], g[mask_nonzero_real])
        print(f"[Kendall's tau] Entire: {kendall_corr:.4f}, Masked Real: {kendall_corr_masked_real:.4f}")

        # 3. Return results in a dictionary
        res_dict = {
            "Spearman": spearman_corr,
            "Kendall": kendall_corr,
            "Spearman_masked_real": spearman_corr_masked_real,
            "Kendall_masked_real": kendall_corr_masked_real
        }

        return res_dict
    
    # Added - Jiwoo
    def analyze_edge_distributions(self):
        real_edge_distr = np.zeros((self.n_vertex, self.n_vertex))
        gen_edge_distr = np.zeros((self.n_vertex, self.n_vertex))
                
        for path in self.real_paths:
            if self.sim_time:
                path = path[0]
            for a, b in zip(path, path[1:]):
                real_edge_distr[a][b] += 1
            
        for path in self.gen_paths:
            for a, b in zip(path, path[1:]):
                gen_edge_distr[a][b] += 1
                
        real_edge_distr /= np.sum(real_edge_distr)
        gen_edge_distr /= np.sum(gen_edge_distr)
        
        r = real_edge_distr.reshape(-1)
        g = gen_edge_distr.reshape(-1)
        
        # [Analysis 0] General Information -----------------------------------------------------------------------
        print(f"Numer of real data: {len(self.real_paths)}, number of generated data: {len(self.gen_paths)}")
        print(f"dimension of r & g = n_vertex * n_vertex: {len(r)}")
        print(f"maxium value of r: {r.max()}, maximum value of g: {g.max()}")
        print(f"Number of edges where r>0 and g>0: {((r > 0) & (g > 0)).sum()}")
        print(f"Probability mass of edges where r>0 and g=0: {r[(r > 0) & (g == 0)].sum()}")
        print(f"Probability mass of edges where r=0 and g>0: {g[(r == 0) & (g > 0)].sum()}")
        print("Number of edges where r>0:", np.sum(r > 0))
        print("Number of edges where g>0:", np.sum(g > 0))

        # [Analysis 1] R2 score - original
        r_mean = r.mean()
        rss = ((r - g) ** 2).sum()  # residual sum of squares
        tss = ((r - r_mean) ** 2).sum()  # total sum of squares
        r2 = 1 - rss / tss
        print(f"[Analysis 1] R2 (original): {r2:.4f}")

        # [Analysis 2] R2 score compared to uniform distribution on the support of real distribution 
        real_support = (r != 0)
        k = np.sum(real_support)

        r_uniform = np.zeros_like(r)
        r_uniform[real_support] = 1 / k  # uniform distribution over non-zero edges

        rss_uniform = ((r - r_uniform) ** 2).sum()
        r2_uniform = 1 - rss_uniform / tss
        print(f"[Analysis 2] R2 compared to uniform distribution on the support of real distribution: {r2_uniform:.4f}")

        # [Analysis 3] R2 score on the support of real distribution 
        r_mean_support = r[real_support].mean()
        rss_support = ((r[real_support] - g[real_support]) ** 2).sum()
        tss_support = ((r[real_support] - r_mean_support) ** 2).sum()
        r2_support = 1 - rss_support / tss_support
        print(f"[Analysis 3] R2 on the support of real distribution: {r2_support:.4f}")

        # [Visualization 1] Any non-zero edges 
        mask_nonzero_union = (r > 0) | (g > 0)
        self.visualize_edge_distributions(r, g, mask_nonzero_union, "./sets_res", "any_nonzero_edges.png", eps=1e-8)

        # [Visulization 2] Both non-zero edges 
        mask_nonzero_both = (r > 0) & (g > 0)
        self.visualize_edge_distributions(r, g, mask_nonzero_both, "./sets_res", "both_nonzero_edges.png", xlim=(1e-5, 0.01), ylim=(1e-5, 0.01), eps=1e-8)
    
    # Added - Jiwoo
    def visualize_edge_distributions(self, real_edge, gen_edge, mask, folder_path, file_name, xlim=None, ylim=None, eps=1e-8):
        plt.figure(figsize=(8,6))
        
        vals = np.log10(real_edge[mask] + eps)
        sc = plt.scatter(
            real_edge[mask] + eps,
            gen_edge[mask] + eps,
            c=vals,                 
            cmap="viridis",         
            s=18,
            alpha=0.8
        )

        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Probability of Real Data")
        plt.ylabel("Probability of Generated Data")

        if xlim is not None and ylim is not None:
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], linestyle="--", color="gray")
        else:
            plt.plot([eps, 1], [eps, 1], linestyle="--", color="gray")

        plt.colorbar(sc, label="Probability (log scale) of Real Data")
        plt.tight_layout()
        plt.savefig(f"{folder_path}/{file_name}")
        plt.close()


    def calculate_divergences(self):
        real_edge_distr = np.zeros((self.n_vertex, self.n_vertex))
        gen_edge_distr = np.zeros((self.n_vertex, self.n_vertex))
        
        real_len_distr = np.zeros(self.n_vertex + 1)
        gen_len_distr = np.zeros(self.n_vertex + 1)
        
        for path in self.real_paths:
            if self.sim_time:
                path = path[0]
            for a, b in zip(path, path[1:]):
                real_edge_distr[a][b] += 1
            real_len_distr[len(path)] += 1
            
        for path in self.gen_paths:
            for a, b in zip(path, path[1:]):
                gen_edge_distr[a][b] += 1
            gen_len_distr[len(path)] += 1
                
        real_edge_distr /= np.sum(real_edge_distr)
        gen_edge_distr /= np.sum(gen_edge_distr)
        real_len_distr /= np.sum(real_len_distr)
        gen_len_distr /= np.sum(gen_len_distr)
        
        # edge_distr_kl = Evaluator.KL_divergence(real_edge_distr.reshape(-1) + 1e-5, gen_edge_distr.reshape(-1) + 1e-5)
        # edge_distr_js = Evaluator.JS_divergence(real_edge_distr.reshape(-1) + 1e-5, gen_edge_distr.reshape(-1) + 1e-5)

        r = real_edge_distr.reshape(-1)
        g = gen_edge_distr.reshape(-1)
        m = 0.5 * (r + g)
        edge_distr_js = 0.5 * np.sum(rel_entr(r, m)) + 0.5 * np.sum(rel_entr(g, m))

        r_mean = r.mean()
        rss = ((r - g) ** 2).sum()  # residual sum of squares
        tss = ((r - r_mean) ** 2).sum()  # total sum of squares
        r2 = 1 - rss / tss

        mse = np.mean((r - g) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(r - g))

        res_dict = {
            "JSEV": edge_distr_js,
            # "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        } 
        
        return res_dict
    
    def calculate_nll(self, disc=None, adj_matrix=None):
        nlls = self.model.eval_nll_fix(self.real_paths, disc=disc, adj_matrix=adj_matrix)
        nll_min = np.min(nlls)
        nll_max = np.max(nlls)
        nll_avg = np.mean(nlls)
        res_dict = {
            "nll_avg": nll_avg,
            "nll_min": nll_min, 
            "nll_max": nll_max, 
        }
        return res_dict

    def eval_except_nll(self, disc=None, adj_matrix=None):
        # self.analyze_edge_distributions()               # Added - Jiwoo
        rank_dict = self.calculate_rank_correlation()   # Added - Jiwoo
        div_dict = self.calculate_divergences()
        return {**div_dict, **rank_dict}    # After  - Jiwoo

    def eval_all(self, disc=None, adj_matrix=None):
        # self.analyze_edge_distributions()               # Added - Jiwoo
        rank_dict = self.calculate_rank_correlation()   # Added - Jiwoo
        div_dict = self.calculate_divergences()
        nll_dict = self.calculate_nll(disc=disc, adj_matrix=adj_matrix)
        # return dict(div_dict, **nll_dict)             # Before - Jiwoo
        return {**div_dict, **nll_dict, **rank_dict}    # After  - Jiwoo

    def _convert_from_id_to_lat_lng(self, paths, sim_time=False):
        path_coors = []
        for path in paths:
            if sim_time:
                path_coors.append([[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]] for v in path[0]])
            else:
                path_coors.append([[self.dataset.G.nodes[v]["lat"], self.dataset.G.nodes[v]["lng"]] for v in path])
        return path_coors

    def A_vis(self, suffix):
        A_idx = (self.A != 0).nonzero(as_tuple=False).tolist()
        A_coors = self._convert_from_id_to_lat_lng(A_idx, False)
        A_highlight_coors = self._convert_from_id_to_lat_lng(self.removal["edges_reverse"], False)
        A_count = draw_heatmap(A_coors, f"./figs/seq_A_{suffix}.html", colors=["blue"], no_points=False, weight=3, highlight=A_highlight_coors)

    def eval(self, suffix, res=None):

        # x_min, x_max = 36.361, 36.362
        # y_min, y_max = 127.3575, 127.3585
        #
        # # Finding the path index and coordinate index
        # indices_in_range = []
        # for path_index, path in enumerate(planned_paths_coors):
        #     for coord_index, (x, y) in enumerate(path):
        #         if x_min <= x <= x_max and y_min <= y <= y_max:
        #             indices_in_range.append((path_index, coord_index))
        # unique_path_indices = sorted(set([path_index for path_index, _ in indices_in_range]))

        # path draw
        idx_for_analysis = [189, 346, 414, 434, 435, 458, 459, 473, 492, 532, 650, 655, 662, 718, 725, 743, 764, 765, 773, 800, 812, 878, 895, 923, 939, 953, 964, 971, 984, 987, 989, 995, 1000, 1070, 1073, 1094, 1106, 1108, 1128, 1131, 1136, 1167, 1176, 1186, 1192, 1193, 1240, 1250, 1274, 1276, 1302, 1312, 1319, 1325, 1333, 1353, 1359, 1366, 1371, 1381, 1405, 1410, 1451, 1459, 1463, 1472, 1491, 1494, 1512, 1521, 1537, 1544, 1551, 1555, 1578, 1606, 1610, 1629, 1661, 1666, 1672, 1706, 1719, 1743, 1763, 1775, 1794, 1797, 1815, 1818, 1819, 1845, 1877, 1907, 1923, 1929, 1947, 1950, 1951, 1963]
        filtered_paths = [self.gen_paths[i] for i in idx_for_analysis]
        for i in range(len(idx_for_analysis)):
            draw_paths([filtered_paths[i]], self.dataset.G, f"./figs_path_analysis/PATH_{i}_seq_gen_{suffix}.html")
        real_filtered_paths = [self.real_paths[i] for i in idx_for_analysis]
        for i in range(len(idx_for_analysis)):
            try:
                draw_paths([real_filtered_paths[i]], self.dataset.G, f"./figs_path_analysis/PATH_{i}_seq_real_{suffix}.html")
            except:
                print(f'Loop! ./figs_path_analysis/PATH_{i}_seq_real_{suffix}.html')

        planned_paths_coors = self._convert_from_id_to_lat_lng(self.gen_paths, False)
        gen_path_count = draw_heatmap(planned_paths_coors, f"./figs/seq_gen_{suffix}.html", colors=["red"] * len(planned_paths_coors), no_points=False)
        orig_paths_coors = self._convert_from_id_to_lat_lng(self.real_paths, self.sim_time)
        orig_path_count = draw_heatmap(orig_paths_coors, f"./figs/seq_real_{suffix}.html", colors=["blue"] * len(orig_paths_coors), no_points=False)

