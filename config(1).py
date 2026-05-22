
import os
import json
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Configargs:
    """
    配置参数类：管理模型训练、数据路径和超参数配置
    支持材料科学领域的关系抽取任务配置
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置参数
        
        Args:
            config_file: 可选的配置文件路径，用于从JSON文件加载配置
        """
        # 基础路径配置
        self.project_root = Path(os.getcwd())
        
        # 关系抽取任务配置
        self.num_rel = 13  # 关系数量
        
        # 数据文件路径配置
        self.train_file = "./dataset/ASaIE/train_data.json"
        self.dev_file = "./dataset/ASaIE/dev_data.json"
        self.schema_fn = "./dataset/ASaIE/schema.json"
        self.test_file = "./dataset/ASaIE/dev_data.json"
        self.tags = "./dataset/tag2id.json"

        # 获取当前文件的绝对路径，然后计算出预训练模型的绝对路径
        self.bert_path = r"D:\BaiduNetdiskDownload\知识图谱260407\知识图谱\chinese_roberta_L-12_H-768\chinese_roberta_L-12_H-768"
        self.checkpoint = "checkpoint/ASaREl_ALL_self.pt"  # 保存模型位置
        
        # 结果输出路径
        self.dev_result = "dev_result/type/data_result_ASaIE.json"
        self.test_result = "dev_result/data1.json"
        
        # 模型架构参数
        self.bert_dim = 768  # BERT维度
        self.tag_size = 4  # 标签大小
        self.word_vocab_size = 30522  # BERT-Base的词汇表大小
        self.word_embed_dim = 768  # BERT-Base的嵌入维度
        self.add_layer = 1  # 在BERT的第几层融合词向量
        
        # 训练超参数
        self.batch_size = 2  # 批次大小
        self.max_len = 195  # 最大输入序列长度
        self.learning_rate = 1e-5  # 学习率
        self.epochs = 20  # 训练轮数
        self.eps = 1.0e-08  # Adam优化器的epsilon参数
        self.warm_up_ratio = 0.08  # 预热比例
        self.weight_decay = 1e-5  # 权重衰减
        self.max_grad_norm = 1.0  # 梯度裁剪阈值
        
        # Dropout配置
        self.dropout_prob = 0.1  # 通用dropout概率
        self.entity_pair_dropout = 0.2  # 实体对dropout概率
        
        # 数据集配置
        self.dataset = "ASaIE"  # 数据集名称
        
        # 词嵌入相关配置
        self.max_word_num = 1  # 最大词组数量
        self.data_path = "./dataset/ASaIE/word_v1"  # 构建词表保存位置
        self.overwrite = False  # 是否覆盖保存到指定路径
        self.max_scan_num = 4000000  # 最大扫描数量
        self.pretrain_embed_path = './word_embedding/tencent-ailab-embedding-zh-d200-v0.2.0/tencent-ailab-embedding-zh-d200-v0.2.0.txt'
        
        # 日志配置
        self.log = f"log/{self.dataset}_RoBert_log_ASaIE_1.log"
        self.log_level = "INFO"
        self.log_interval = 350  # 每多少步记录一次日志
        
        # 数据加载参数
        self.num_workers = 0  # Windows系统建议设为0
        self.pin_memory = True
        
        # 学习率调度器参数
        self.scheduler_step = 10
        self.scheduler_gamma = 0.9
        
        # 早停参数
        self.early_stopping_patience = 10
        self.min_delta = 1e-4
        
        # 评估参数
        self.eval_interval = 5  # 每多少个epoch评估一次
        self.save_best_only = True
        
        # 材料科学特定参数
        self.material_constraints = {
            'process_params': ['temperature', 'pressure', 'time', 'atmosphere'],
            'performance_metrics': ['strength', 'hardness', 'conductivity', 'density'],
            'composition_elements': ['Fe', 'C', 'Ni', 'Cr', 'Al', 'Ti', 'Cu', 'Zn']
        }
        
        # 动态任务划分参数
        self.task_split_config = {
            'max_tasks_per_level': 10,
            'min_samples_per_task': 50,
            'complexity_threshold': 0.7,
            'dependency_weight': 0.3
        }
        
        # EWC参数
        self.ewc_config = {
            'lambda_ewc': 1000,
            'fisher_sample_size': 1000,
            'importance_decay': 0.9
        }
        
        # 语义回放参数
        self.replay_config = {
            'buffer_size': 1000,
            'replay_ratio': 0.2,
            'similarity_threshold': 0.8,
            'diversity_weight': 0.3
        }
        
        # 如果提供了配置文件，则从文件加载配置
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # 验证配置
        self._validate_config()
        
        logger.info("配置参数初始化完成")
    
    def _validate_config(self) -> None:
        """
        验证配置参数的有效性
        """
        try:
            # 验证数值参数
            assert self.batch_size > 0, "batch_size必须大于0"
            assert self.learning_rate > 0, "learning_rate必须大于0"
            assert self.epochs > 0, "epochs必须大于0"
            assert self.max_len > 0, "max_len必须大于0"
            assert 0 <= self.dropout_prob <= 1, "dropout_prob必须在[0,1]范围内"
            assert 0 <= self.entity_pair_dropout <= 1, "entity_pair_dropout必须在[0,1]范围内"
            assert self.num_rel > 0, "num_rel必须大于0"
            
            # 验证文件路径（如果文件不存在则给出警告）
            required_files = [self.train_file, self.dev_file, self.schema_fn, self.tags]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.warning(f"必需文件不存在: {file_path}")
            
            # 验证BERT路径
            if not os.path.exists(self.bert_path):
                logger.warning(f"BERT模型路径不存在: {self.bert_path}")
            
            logger.info("配置参数验证完成")
            
        except AssertionError as e:
            logger.error(f"配置参数验证失败: {e}")
            raise
        except Exception as e:
            logger.error(f"配置验证过程中出现错误: {e}")
            raise
    
    def load_from_file(self, config_file: str) -> None:
        """
        从JSON文件加载配置参数
        
        Args:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 更新配置参数
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"从文件更新配置: {key} = {value}")
                else:
                    logger.warning(f"未知配置参数: {key}")
            
            logger.info(f"成功从文件加载配置: {config_file}")
            
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def save_to_file(self, config_file: str) -> None:
        """
        将当前配置保存到JSON文件
        
        Args:
            config_file: 配置文件保存路径
        """
        try:
            # 获取所有配置参数
            config_dict = self.get_config_dict()
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            # 保存配置
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            
            logger.info(f"配置已保存到文件: {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        获取配置参数字典
        
        Returns:
            Dict[str, Any]: 配置参数字典
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict
    
    def update_config(self, **kwargs) -> None:
        """
        批量更新配置参数
        
        Args:
            **kwargs: 要更新的配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"更新配置参数: {key} = {value} (原值: {old_value})")
            else:
                logger.warning(f"尝试设置未知配置参数: {key}")
        
        # 重新验证配置
        self._validate_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型相关的配置参数
        
        Returns:
            Dict[str, Any]: 模型配置参数字典
        """
        model_keys = [
            'bert_dim', 'tag_size', 'word_vocab_size', 'word_embed_dim',
            'add_layer', 'dropout_prob', 'entity_pair_dropout', 'num_rel'
        ]
        
        return {key: getattr(self, key) for key in model_keys if hasattr(self, key)}
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        获取训练相关的配置参数
        
        Returns:
            Dict[str, Any]: 训练配置参数字典
        """
        training_keys = [
            'batch_size', 'max_len', 'learning_rate', 'epochs', 'eps',
            'warm_up_ratio', 'weight_decay', 'max_grad_norm'
        ]
        
        return {key: getattr(self, key) for key in training_keys if hasattr(self, key)}
    
    def __str__(self) -> str:
        """
        返回配置参数的字符串表示
        
        Returns:
            str: 配置参数字符串
        """
        config_str = "=== 配置参数 ===\n"
        config_dict = self.get_config_dict()
        
        # 按类别组织输出
        categories = {
            '数据配置': ['train_file', 'dev_file', 'test_file', 'schema_fn', 'tags', 'dataset'],
            '模型配置': ['bert_path', 'bert_dim', 'tag_size', 'word_vocab_size', 'word_embed_dim', 'add_layer'],
            '训练配置': ['batch_size', 'max_len', 'learning_rate', 'epochs', 'eps', 'warm_up_ratio'],
            '其他配置': []
        }
        
        # 收集未分类的配置项
        categorized_keys = set()
        for cat_keys in categories.values():
            categorized_keys.update(cat_keys)
        
        categories['其他配置'] = [key for key in config_dict.keys() if key not in categorized_keys]
        
        # 按类别输出
        for category, keys in categories.items():
            if keys:
                config_str += f"\n{category}:\n"
                for key in keys:
                    if key in config_dict:
                        config_str += f"  {key}: {config_dict[key]}\n"
        
        return config_str
