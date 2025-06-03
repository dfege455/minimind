# 导入必要的库和模块
import os  # 操作系统接口
import platform  # 获取平台信息
import argparse  # 命令行参数解析
import time  # 时间相关操作
import math  # 数学运算
import warnings  # 警告处理
import pandas as pd  # 数据处理（虽然代码中未直接使用）
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练支持
from torch import optim, nn  # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行模型包装
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器（虽然代码中未直接使用）
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载和分布式采样
from contextlib import nullcontext  # 上下文管理器

from transformers import AutoTokenizer  # HuggingFace的自动分词器

from model.model import MiniMindLM  # 自定义的模型类
from model.LMConfig import LMConfig  # 模型配置类
from model.dataset import PretrainDataset  # 预训练数据集类

warnings.filterwarnings('ignore')  # 忽略警告信息


def Logger(content):
    """分布式环境下的日志打印函数，仅在主进程打印日志"""
    if not ddp or dist.get_rank() == 0:  # 非分布式模式或主进程
        print(content)


def get_lr(current_step, total_steps, lr):
    """自定义学习率调度函数，结合余弦退火和基础学习率"""
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    """单个训练周期的完整流程"""
    loss_fct = nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失函数，不自动求平均
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 将数据移动到指定设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 更新学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度训练上下文
        with ctx:
            # 前向传播
            res = model(X)
            # 计算损失（考虑损失掩码）
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # 应用损失掩码
            loss += res.aux_loss  # 添加辅助损失（如MoE的负载平衡损失）
            loss = loss / args.accumulation_steps  # 梯度累积归一化

        # 反向传播（自动缩放损失）
        scaler.scale(loss).backward()

        # 梯度累积步骤更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 取消梯度缩放以进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 参数更新
            scaler.update()  # 更新缩放器状态

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        # 日志记录
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # 恢复实际损失值
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            # 记录到wandb
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 定期保存模型检查点
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''  # MoE模型特殊标识
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            # 处理分布式模型的状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    """初始化模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 加载预训练分词器
    model = MiniMindLM(lm_config).to(args.device)  # 初始化模型并移动设备
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')  # 打印参数量
    return model, tokenizer


def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")  # 使用NCCL后端初始化进程组
    ddp_rank = int(os.environ["RANK"])  # 全局进程排名
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本地进程排名
    ddp_world_size = int(os.environ["WORLD_SIZE"])  # 总进程数
    DEVICE = f"cuda:{ddp_local_rank}"  # 设置当前设备
    torch.cuda.set_device(DEVICE)  # 绑定设备


# 分布式训练启动命令示例：torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")  # 输出目录
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮次
    parser.add_argument("--batch_size", type=int, default=32)  # 批次大小
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # 初始学习率
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备
    parser.add_argument("--dtype", type=str, default="bfloat16")  # 混合精度类型
    parser.add_argument("--use_wandb", action="store_true")  # 是否使用wandb
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")  # wandb项目名称
    parser.add_argument("--num_workers", type=int, default=1)  # 数据加载工作线程数
    parser.add_argument("--ddp", action="store_true")  # 是否使用分布式训练
    parser.add_argument("--accumulation_steps", type=int, default=8)  # 梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)  # 梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0)  # 预热步数（当前未使用）
    parser.add_argument("--log_interval", type=int, default=100)  # 日志间隔
    parser.add_argument("--save_interval", type=int, default=100)  # 保存间隔
    parser.add_argument('--local_rank', type=int, default=-1)  # 分布式训练的本地rank（自动设置）
    # 模型配置参数
    parser.add_argument('--dim', default=512, type=int)  # 模型维度
    parser.add_argument('--n_layers', default=8, type=int)  # 层数
    parser.add_argument('--max_seq_len', default=512, type=int)  # 最大序列长度
    parser.add_argument('--use_moe', default=False, type=bool)  # 是否使用MoE
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")  # 训练数据路径
    args = parser.parse_args()

    # 模型配置初始化
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)  # 模型保存路径
    os.makedirs(args.save_dir, exist_ok=True)  # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len  # 每迭代处理的token数
    torch.manual_seed(1337)  # 固定随机种子
    device_type = "cuda" if "cuda" in args.device else "cpu"  # 设备类型

    # wandb运行名称
    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 混合精度训练上下文（CPU模式使用普通上下文）
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 分布式训练检测（通过环境变量判断）
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否处于分布式环境
    ddp_local_rank, DEVICE = 0, "cuda:0"  # 默认值

    # 初始化分布式训练
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    # 初始化wandb（仅在主进程）
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    # 数据集和数据加载器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None  # 分布式采样器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  # 锁页内存加速数据传输
        drop_last=False,
        shuffle=False,  # 分布式模式下通过采样器实现shuffle
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 混合精度梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # 优化器

    # 分布式数据并行包装
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}  # 忽略旋转位置编码参数
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
    # 训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)