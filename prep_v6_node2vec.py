"""
v6 그래프(1387 노드)의 node2vec 임베딩 캐시를 1회 생성한다.

학습 스크립트를 여러 GPU에서 동시에 띄우면 각 프로세스가 캐시가 없는 것을 보고
동시에 node2vec을 계산/덮어쓰므로(loader/node2vec.py:130), 반드시 학습 전에
이 스크립트를 한 번만 단독 실행할 것.

  python prep_v6_node2vec.py                        # LE/LP 공용 (그래프 동일)

생성물: sets_data_v6/porto_node2vec.pkl, sets_data_v6/porto_path.pkl
구 데이터(1390 노드)의 sets_data/porto_node2vec.pkl 과 충돌하지 않도록 디렉터리를 분리했다.
"""

import argparse
import pickle
from os.path import join

from loader.node2vec import get_node2vec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-index", type=str, default="v6-LE1.0-0.05_normal")
    ap.add_argument("-data", type=str, default="./porto_data_v6")
    ap.add_argument("-out", type=str, default="./sets_data_v6")
    args = ap.parse_args()

    G = pickle.load(open(join(args.data, f"porto_shrink_G_{args.index}.pkl"), "rb"))
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    emb = get_node2vec(G,
                       join(args.out, "porto_node2vec.pkl"),
                       join(args.out, "porto_path.pkl"))
    print(f"node2vec: {len(emb)} vectors, dim={emb[0].shape[0]}")
    assert len(emb) == G.number_of_nodes(), "node2vec size != n_vertex"
    print("PREP_V6_NODE2VEC_OK")
