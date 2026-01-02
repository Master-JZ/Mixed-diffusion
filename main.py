import os
import torch
from tqdm import tqdm
from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer 
from utils import set_seed
from configs.cadirec_config import get_config
import logging
import builtins
import json


def main():
    args = get_config()
    set_seed(args.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    #new
    log_path = setup_logging(args, args_str)
    # print(f"Logging to {log_path}")

    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)


    data_generator = DataGenerator(args)
    trainer = Trainer(args, device, data_generator)
    trainer.train()
 

##new  for log
def setup_logging(args, args_str):
    """
    根据 args.output_dir 和 args_str 创建一个 log 文件，
    1）用多行 JSON 形式记录本次运行的参数
    2）后面所有 logging.info(...) 都会写进这个文件

    不改写全局 print，终端输出保持原状。
    """
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, args_str + ".log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()  # 继续在命令行显示 logging 的内容
        ],
        force=True
    )

    logging.info(f"Logging to {log_path}")
    logging.info("=== Run config (args) ===")
    # 多行 JSON，方便看
    logging.info(
        json.dumps(vars(args), indent=2, ensure_ascii=False, sort_keys=True)
    )

    return log_path


if __name__ == "__main__":
    main()






