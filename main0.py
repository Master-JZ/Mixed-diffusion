import os
import torch
from tqdm import tqdm
from data_generators.data_generator import DataGenerator
from trainers.trainer import Trainer 
from utils import set_seed
from configs.cadirec_config import get_config
import logging
import builtins


def main():
    args = get_config()
    set_seed(args.seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    #new
    log_path = setup_logging(args, args_str)
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)


    data_generator = DataGenerator(args)
    trainer = Trainer(args, device, data_generator)
    trainer.train()
 
def setup_logging(args, args_str):
    """
    根据 args.output_dir 和 args_str 创建一个 log 文件，
    然后把全局 print 替换成 logging.info，
    这样所有 print 同时输出到终端和 log 文件。
    """
    # log 目录：比如 output/logs
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, args_str + ".log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()  # 继续在命令行显示
        ]
    )

    logger = logging.getLogger(__name__)

    # 把内置 print 改写成用 logging.info
    def log_print(*values, **kwargs):
        msg = " ".join(str(v) for v in values)
        logger.info(msg)

    builtins.print = log_print  # 以后所有 print() 都会走到 logging

    logger.info(f"Logging to {log_path}")
    logger.info(f"Args: {args}")

    return log_path


if __name__ == "__main__":
    main()






