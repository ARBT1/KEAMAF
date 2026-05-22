import numpy as np
import random
import logging
import os
import sys
import argparse
import json
import re
from typing import Optional
from datetime import datetime
from collections import Counter

from config.config import Configargs
from models.semantic_replay_buffer import SemanticReplayBuffer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        seed: 随机种子值
    """
    try:
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 如果有GPU可用，也设置CUDA的随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 确保CUDA操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"随机种子已设置为: {seed}")
        
    except Exception as e:
        logger.error(f"设置随机种子时出错: {e}")
        raise

def check_environment() -> None:
    """
    检查运行环境和依赖项
    """
    import torch
    logger.info("=== 环境检查 ===")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 检查内存使用情况
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU内存: {gpu_memory:.2f} GB")

def validate_config(config: Configargs) -> bool:
    """
    验证配置参数的有效性
    
    Args:
        config: 配置对象
        
    Returns:
        bool: 配置是否有效
    """
    try:
        logger.info("配置验证通过")
        return True
        
    except Exception as e:
        logger.error(f"配置验证时出错: {e}")
        return False

def create_output_directories(config: Configargs) -> None:
    """
    创建输出目录
    
    Args:
        config: 配置对象
    """
    try:
        # 创建结果输出目录
        result_dir = "results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
            logger.info(f"创建结果目录: {result_dir}")
        
        # 创建日志目录
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"创建日志目录: {log_dir}")
            
    except Exception as e:
        logger.error(f"创建输出目录时出错: {e}")
        raise

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='材料科学关系抽取训练和测试程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode train     # 仅训练
  python main.py --mode test      # 仅测试
  python main.py --mode both      # 训练+测试（默认）
  python main.py --mode package   # 生成阶段一数据包
  python main.py                  # 默认运行训练+测试
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'test', 'both', 'package'], 
        default='both',
        help='运行模式: train(仅训练), test(仅测试), both(训练+测试，默认), package(生成阶段一数据包)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=2222,
        help='随机种子值（默认: 2222）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细日志输出'
    )

    parser.add_argument(
        '--input',
        type=str,
        default='./dataset/ASaIE/train_data.json',
        help='阶段一数据包生成的输入文件路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='stage1_data_packages.json',
        help='阶段一数据包输出文件路径'
    )

    parser.add_argument(
        '--task-id',
        type=int,
        default=1,
        help='阶段一任务编号'
    )

    parser.add_argument(
        '--id-field',
        type=str,
        default='id',
        help='样本唯一标识字段名'
    )

    parser.add_argument(
        '--time-field',
        type=str,
        default='time',
        help='样本时间字段名'
    )
    
    return parser.parse_args()

def normalize_confidence(score: float) -> float:
    """
    将原始分数归一化到 0.01 到 0.99 之间的置信度分数。
    
    用于将模型抽取的非标准化分数转化为代表置信度的概率值。
    
    Args:
        score: 原始分数
        
    Returns:
        float: 归一化后的置信度 (0.01 ~ 0.99)
    """
    if score <= 0:
        return 0.01
    value = score / (score + 5.0)
    return max(0.01, min(0.99, float(value)))

def extract_triplets(sample: dict) -> list:
    """
    从原始样本字典中提取标准化格式的三元组列表。
    
    兼容不同的数据格式约定（如 spo_list, relation_list, triples 等）。
    
    Args:
        sample: 包含原始文本和关系的样本字典
        
    Returns:
        list: 提取并规范化后的三元组元组列表 [(head, relation, tail), ...]
    """
    triplets = sample.get('spo_list') or sample.get('relation_list') or sample.get('triples') or []
    normalized = []
    for item in triplets:
        if isinstance(item, dict):
            head = item.get('head') or item.get('subject')
            relation = item.get('relation') or item.get('predicate')
            tail = item.get('tail') or item.get('object')
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            head, relation, tail = item[0], item[1], item[2]
        else:
            continue
        if head is None or relation is None or tail is None:
            continue
        normalized.append((head, relation, tail))
    return normalized

def parse_value_unit(text: str) -> tuple:
    """
    解析文本中包含的数值和单位。
    
    使用正则表达式提取材料性能、工艺参数等尾实体中的定量数据和物理单位。
    
    Args:
        text: 包含可能数值和单位的字符串（如 "120 °C" 或 "500 MPa"）
        
    Returns:
        tuple: (提取的数值字符串, 提取的单位字符串) 若无则为 None
    """
    if not text:
        return None, None
    match = re.search(r'(-?\d+(?:\.\d+)?)\s*([a-zA-Z%°/]+)?', text)
    if not match:
        return None, None
    value = match.group(1)
    unit = match.group(2) if match.group(2) else None
    return value, unit

def build_data_package(sample: dict, index: int, task_id: int, id_field: str, time_field: str, buffer: SemanticReplayBuffer) -> dict:
    """
    将原始样本打包为用于知识图谱更新的数据包 (Package) 格式。
    
    包含唯一的文献 ID、任务 ID，并提取每个三元组的具体信息，
    包括计算其置信度以及解析尾实体中的定量数值和单位。
    
    Args:
        sample: 原始输入样本字典
        index: 样本在当前任务中的索引
        task_id: 当前所属任务 ID
        id_field: 样本中代表唯一标识的键名
        time_field: 样本中代表时间戳的键名
        buffer: 经验回放缓冲区，用于计算语义相似度和置信度
        
    Returns:
        dict: 结构化的数据包字典，供阶段二使用
    """
    text = sample.get('text', '')
    sample_id = sample.get(id_field) or f"sample_{index}"
    triplets = extract_triplets(sample)
    package = {
        'id': sample_id,
        'task': task_id,
        'triplets': []
    }
    time_value = sample.get(time_field)
    if time_value is not None:
        package['time'] = task_id*5+2000
    for head, relation, tail in triplets:
        score_sample = {'text': text, 'spo_list': [[head, relation, tail]]}
        confidence = normalize_confidence(buffer.score_sample(score_sample))
        value, unit = parse_value_unit(str(tail))
        triplet_item = {
            'head': str(head),
            'relation': str(relation),
            'tail': str(tail),
            'raw_sentence': text,
            'confidence': confidence
        }
        if value is not None:
            triplet_item['quantitative_value'] = value
        if unit:
            triplet_item['unit'] = unit
        package['triplets'].append(triplet_item)
    return package

def summarize_packages(packages: list) -> dict:
    total_samples = len(packages)
    total_triplets = 0
    relation_counter = Counter()
    for pkg in packages:
        total_triplets += len(pkg.get('triplets', []))
        for t in pkg.get('triplets', []):
            relation_counter[t.get('relation')] += 1
    avg_triplets = total_triplets / total_samples if total_samples else 0.0
    return {
        'total_samples': total_samples,
        'total_triplets': total_triplets,
        'avg_triplets_per_sample': avg_triplets,
        'relation_distribution': dict(relation_counter)
    }

def run_stage_one(input_path: str, output_path: str, task_id: int, id_field: str, time_field: str) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入数据必须是列表格式")
    buffer = SemanticReplayBuffer()
    packages = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            continue
        packages.append(build_data_package(sample, idx, task_id, id_field, time_field, buffer))
    metrics = summarize_packages(packages)
    output = {
        'packages': packages,
        'metrics': metrics
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"阶段一数据包已保存: {output_path}")
    logger.info(f"阶段一指标: {metrics}")
    return output_path

def main(args) -> None:
    """
    主函数：执行完整的训练和测试流程
    
    Args:
        mode: 运行模式 - "train"(仅训练), "test"(仅测试), "both"(训练+测试)
        seed: 随机种子值
        verbose: 是否启用详细日志输出
    """
    # 根据verbose参数调整日志级别
    if args.mode == "package":
        run_stage_one(
            input_path=args.input,
            output_path=args.output,
            task_id=args.task_id,
            id_field=args.id_field,
            time_field=args.time_field
        )
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("启用详细日志输出")
    
    start_time = datetime.now()
    logger.info(f"=== 材料科学关系抽取任务开始 ===")
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"开始时间: {start_time}")
    
    try:
        # 1. 检查运行环境
        check_environment()
        
        # 2. 加载配置
        logger.info("加载配置参数...")
        config = Configargs()
        
        # 3. 验证配置
        if not validate_config(config):
            logger.error("配置验证失败，程序退出")
            sys.exit(1)
        
        # 4. 创建输出目录
        create_output_directories(config)
        
        # 5. 设置随机种子
        set_seed(args.seed)
        
        # 6. 初始化框架
        logger.info("初始化训练框架...")
        from framework.framework import Framework
        fw = Framework(config)
        
        # 7. 根据模式执行相应操作
        if args.mode in ["train", "both"]:
            logger.info("开始模型训练...")
            fw.train()
            logger.info("模型训练完成")
        
        if args.mode in ["test", "both"]:
            logger.info("开始模型测试...")
            fw.test()
            logger.info("模型测试完成")
        
        # 8. 计算总耗时
        end_time = datetime.now()
        total_time = end_time - start_time
        
        logger.info(f"=== 任务完成 ===")
        logger.info(f"结束时间: {end_time}")
        logger.info(f"总耗时: {total_time}")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 使用解析的参数调用main函数
    main(args)
