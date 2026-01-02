import argparse, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data_generators.data_generator import DataGenerator
from models.cadirec import CaDiRec
import sys
def extract_user_repr(args, ckpt_path, device):
    generator = DataGenerator(args)
    model = CaDiRec(device, args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_vecs, all_users = [], []
    with torch.no_grad():
        for batch in generator.test_dataloader:  # 或 valid_dataloader
            input_ids = batch["input_ids"].to(device)
            user_ids  = batch["user_id"].to(device)
            attn_mask = model.get_extended_attention_mask(input_ids)
            seq_emb   = model.add_position_embedding(input_ids)
            seq_out   = model.forward(seq_emb, attn_mask)      # [B,L,D]
            user_repr = seq_out[:, -1, :]                      # 末位表征；也可用 seq_out.mean(1)
            all_vecs.append(user_repr.cpu())
            all_users.append(user_ids.cpu())
    vecs = torch.cat(all_vecs).numpy()
    users = torch.cat(all_users).numpy()
    return vecs, users

def tsne_2d(vecs, seed=0):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
                n_iter=2000, random_state=seed)
    return tsne.fit_transform(vecs)

def plot_scatter(coords, title, path):
    plt.figure(figsize=(5,5))
    plt.scatter(coords[:,0], coords[:,1], s=5, alpha=0.6, c="#8d96a1")
    plt.title(title)
    plt.xlabel("t-SNE-1"); plt.ylabel("t-SNE-2")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--data_path", type=str, default="data/")  # 与训练时一致
    parser.add_argument("--config_mod", type=str, default="configs.cadirec_config")
    parser.add_argument("--ckpt", type=str, required=True)
    # parser.add_argument("--ckpt_with", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    # args = parser.parse_args()
    cli_args = parser.parse_args()  # 先保存命令行参数，避免被覆盖


  # 冻住 sys.argv，防止 config 解析到 --ckpt
    import importlib
    cfg = importlib.import_module(cli_args.config_mod)
    old_argv = sys.argv
    sys.argv = [old_argv[0]]          # 只留脚本名，避免未知参数
    cfg_args = cfg.get_config()       # 这里不会看到 --ckpt
    sys.argv = old_argv               # 解析完恢复

    # 合并参数
    args_dict = vars(cfg_args)
    args_dict.update({
        "dataset": cli_args.dataset,
        "data_path": cli_args.data_path,
        "device": cli_args.device,
        "ckpt": cli_args.ckpt,  # 加上这一行
    })
    args = argparse.Namespace(**args_dict)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    # 无/有上下文两组
    vecs_wo, _ = extract_user_repr(args, args.ckpt, device) 
    # vecs_w, _  = extract_user_repr(args, args.ckpt_with, device)

    coords_wo = tsne_2d(vecs_wo, seed=0)
    # coords_w  = tsne_2d(vecs_w, seed=0)

    plot_scatter(coords_wo, f"Without context on {args.dataset}", f"tsne_wo_{args.dataset}.png")
    # plot_scatter(coords_w,  f"With context on {args.dataset}",  f"tsne_w_{args.dataset}.png")

if __name__ == "__main__":
    main()
