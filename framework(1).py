import json
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from collections import defaultdict
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from dataloader import REDataset, collate_fn
from dataloader.dataloader import REDataset, collate_fn

# from models import OneRel
from models.models import OneRel

# from logger import Logger
from logger.logger import Logger

# from processor import LEBertProcessor
from processors.processor import LEBertProcessor

from transformers import BertTokenizer, BertConfig, get_linear_schedule_with_warmup


class Framework():
    """
    训练框架类：负责模型训练、验证和测试的完整流程
    支持材料科学领域的关系抽取任务
    """
    
    def __init__(self, config):
        """
        初始化训练框架
        
        Args:
            config: 配置对象，包含模型和训练参数
        """
        self.config = config
        
        # 加载标签和关系映射
        try:
            with open(self.config.tags, "r", encoding="utf-8") as f:
                self.tag2id = json.load(f)[1]
            with open(self.config.schema_fn, "r", encoding="utf-8") as fs:
                self.id2rel = json.load(fs)[1]
            logger.info(f"成功加载标签映射，标签数量: {len(self.tag2id)}")
            logger.info(f"成功加载关系映射，关系数量: {len(self.id2rel)}")
        except Exception as e:
            logger.error(f"加载标签或关系映射失败: {e}")
            raise
        
        # 初始化损失函数和分词器
        self.loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, do_lower_case=True)
        
        # 设备配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 训练状态跟踪
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': []
        }
        
        # 统计信息
        self.training_stats = defaultdict(list)
        
        # 初始化日志记录器
        self.log = Logger(self.config.log)

    def build_processor(self) -> Tuple[LEBertProcessor, BertConfig]:
        """
        构建数据处理器和BERT配置
        
        Returns:
            Tuple[LEBertProcessor, BertConfig]: 数据处理器和BERT配置对象
        """
        try:
            # 构建数据集词表，并获取对应词向量
            processor = LEBertProcessor(self.config, self.tokenizer)
            logger.info(f"数据处理器构建完成，词嵌入维度: {processor.word_embedding.shape}")
            
            # 在BERT中融入词向量
            bert_config = BertConfig.from_pretrained(self.config.bert_path)
            bert_config.add_layer = self.config.add_layer
            bert_config.word_vocab_size = processor.word_embedding.shape[0]
            bert_config.word_embed_dim = processor.word_embedding.shape[1]  # 2162 200
            
            logger.info(f"BERT配置更新完成，词汇表大小: {bert_config.word_vocab_size}, 词嵌入维度: {bert_config.word_embed_dim}")
            
            return processor, bert_config
            
        except Exception as e:
            logger.error(f"构建数据处理器失败: {e}")
            raise


    def train(self) -> None:
        """
        执行完整的训练流程
        """
        def cal_loss(predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """
            计算带掩码的交叉熵损失
            
            Args:
                predict: 预测结果
                target: 目标标签
                mask: 损失掩码
                
            Returns:
                torch.Tensor: 计算得到的损失值
            """
            loss_ = self.loss_function(predict, target)
            loss = torch.sum(loss_ * mask) / torch.sum(mask)
            return loss

        try:
            # 构建数据处理器和配置
            processor, bert_config = self.build_processor()
            
            # 初始化模型
            model = OneRel(bert_config).to(self.device)
            model.word_embeddings.weight.data.copy_(torch.from_numpy(processor.word_embedding))
            logger.info(f"模型初始化完成，参数数量: {sum(p.numel() for p in model.parameters())}")

            # 准备数据集
            train_dataset = REDataset(self.config, self.config.train_file, processor)
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True, 
                collate_fn=collate_fn,
                num_workers=getattr(self.config, 'num_workers', 0),
                pin_memory=True if self.device.type == 'cuda' else False
            )
            logger.info(f"训练数据加载完成，样本数: {len(train_dataset)}, 批次数: {len(train_dataloader)}")

            dev_dataset = REDataset(self.config, self.config.dev_file, processor)
            dev_dataloader = DataLoader(
                dev_dataset, 
                batch_size=1, 
                collate_fn=collate_fn,
                num_workers=getattr(self.config, 'num_workers', 0),
                pin_memory=True if self.device.type == 'cuda' else False
            )
            logger.info(f"验证数据加载完成，样本数: {len(dev_dataset)}")

            # 初始化优化器
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.config.learning_rate,
                eps=self.config.eps,
                weight_decay=getattr(self.config, 'weight_decay', 1e-5)
            )
            logger.info(f"优化器初始化完成，学习率: {self.config.learning_rate}")

            # 训练状态变量
            global_step = 0
            global_loss = 0.0
            best_epoch = 0
            best_f1_score = 0.0
            best_recall = 0.0
            best_precision = 0.0
            best_accuracy = 0.0
            
            # 开始训练循环
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                logger.info(f"开始训练 Epoch [{epoch+1}/{self.config.epochs}]")
                
                model.train()
                epoch_loss = 0.0
                num_batches = 0
                
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                
                for batch_idx, data in enumerate(progress_bar):
                    try:
                        # 将数据移动到设备
                        data = self._move_batch_to_device(data)
                        
                        # 前向传播
                        output = model(data)
                        
                        # 清零梯度
                        optimizer.zero_grad()
                        
                        # 计算损失
                        loss = cal_loss(output, data["matrix"], data["loss_mask"])
                        
                        # 检查损失是否有效
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"检测到无效损失值: {loss.item()}，跳过此批次")
                            continue
                        
                        # 反向传播
                        loss.backward()
                        
                        # 梯度裁剪
                        if hasattr(self.config, 'max_grad_norm'):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                        
                        # 更新参数
                        optimizer.step()
                        
                        # 累计损失和步数
                        global_loss += loss.item()
                        epoch_loss += loss.item()
                        global_step += 1
                        num_batches += 1
                        
                        # 更新进度条
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{epoch_loss/num_batches:.4f}'
                        })
                        
                        # 定期记录训练状态
                        if global_step % 350 == 0:
                            avg_loss = global_loss / 350
                            self.log.logger.info(f"epoch: {epoch+1} global_step: {global_step} avg_loss: {avg_loss:.4f}")
                            self.training_stats['step_loss'].append(avg_loss)
                            global_loss = 0.0
                            
                    except Exception as e:
                        logger.error(f"训练批次 {batch_idx} 时出错: {e}")
                        continue
                
                # 记录epoch统计信息
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                epoch_time = time.time() - epoch_start_time
                self.training_history['train_loss'].append(avg_epoch_loss)
                
                logger.info(f"Epoch {epoch+1} 完成，平均损失: {avg_epoch_loss:.4f}, 用时: {epoch_time:.2f}秒")

                # 每5个epoch进行一次验证
                if (epoch + 1) % 5 == 0:
                    accuracy, precision, recall, f1_score, predict = self.evaluate(dev_dataloader, model)
                    if f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_recall = recall
                        best_precision = precision
                        best_accuracy = accuracy
                        best_epoch = epoch + 1
                        print("save model ......")
                        self.log.logger.info("save model......")
                        torch.save(model.state_dict(), self.config.checkpoint)
                        json.dump(predict, open(self.config.dev_result, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
                        print("epoch:{} best_epoch:{} best_accuracy:{:5.4f} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(epoch+1, best_epoch, best_accuracy, best_recall, best_precision, best_f1_score))
                        self.log.logger.info("epoch:{} best_epoch:{} best_accuracy:{:5.4f} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(epoch+1, best_epoch, best_accuracy, best_recall, best_precision, best_f1_score))

            print("best_epoch:{} best_accuracy:{:5.4f} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(best_epoch, best_accuracy, best_recall, best_precision, best_f1_score))
            self.log.logger.info("best_epoch:{} best_accuracy:{:5.4f} best_recall:{:5.4f} best_precision:{:5.4f} best_f1_score:{:5.4f}".format(best_epoch, best_accuracy, best_recall, best_precision, best_f1_score))
            
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            raise


    def evaluate(self, dataloader: DataLoader, model: torch.nn.Module) -> Tuple[float, float, float, float, List[Dict]]:
        """
        评估模型性能
        
        Args:
            dataloader: 评估数据加载器
            model: 待评估的模型
            
        Returns:
            Tuple[float, float, float, float, List[Dict]]: 准确率、精确率、召回率、F1分数和预测结果
        """
        logger.info("开始模型评估...")
        self.log.logger.info("开始模型评估...")
        
        model.eval()
        predict_num, gold_num, correct_num = 0, 0, 0
        total_samples = 0  # 总样本数，用于计算准确率
        predict = []
        
        def to_ret(data: List) -> Tuple:
            """
            将列表转换为元组格式
            
            Args:
                data: 输入数据列表
                
            Returns:
                Tuple: 转换后的元组
            """
            ret = []
            for i in data:
                ret.append(tuple(i))
            return tuple(ret)

        with torch.no_grad():
            for data in tqdm(dataloader):
                # [num_rel, seq_len, seq_len]
                pred_triple_matrix = model(data, train=False).cpu()[0]
                number_rel, seq_lens, seq_lens = pred_triple_matrix.shape
                relations, heads, tails = np.where(pred_triple_matrix > 0)

                token = data["token"][0]
                gold = data["triple"][0]
                pair_numbers = len(relations)
                predict_triple = []
                if pair_numbers > 0:
                    for i in range(pair_numbers):
                        r_index = relations[i]
                        h_start_idx = heads[i]
                        t_start_idx = tails[i]
                        if pred_triple_matrix[r_index][h_start_idx][t_start_idx] == self.tag2id["HB-TB"] and i + 1 < pair_numbers:
                            t_end_idx = tails[i + 1]
                            if pred_triple_matrix[r_index][h_start_idx][t_end_idx] == self.tag2id["HB-TE"]:
                                for h_end_index in range(h_start_idx, seq_lens):
                                    if pred_triple_matrix[r_index][h_end_index][t_end_idx] == self.tag2id["HE-TE"]:

                                        subject_head, subject_tail = h_start_idx, h_end_index
                                        object_head, object_tail = t_start_idx, t_end_idx
                                        subject = ''.join(token[subject_head: subject_tail + 1])
                                        object = ''.join(token[object_head: object_tail + 1])
                                        relation = self.id2rel[str(int(r_index))]
                                        if len(subject) > 0 and len(object) > 0:
                                            predict_triple.append((subject, relation, object))
                                        break
                gold = to_ret(gold)
                predict_triple = to_ret(predict_triple)
                gold_num += len(gold)
                predict_num += len(predict_triple)
                correct_num += len(set(gold) & set(predict_triple))
                lack = set(gold) - set(predict_triple)
                new = set(predict_triple) - set(gold)
                predict.append({"text": data["sentence"][0], "gold": gold, "predict": predict_triple,
                                "lack": list(lack), "new": list(new)})
                total_samples += 1  # 统计总样本数

        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        
        # 计算准确率：完全匹配的样本数 / 总样本数
        exact_match_count = sum(1 for item in predict if set(item['gold']) == set(item['predict']))
        accuracy = exact_match_count / (total_samples + 1e-10)
        
        print("predict_num: {} gold_num: {} correct_num: {} total_samples: {} exact_match: {}".format(
            predict_num, gold_num, correct_num, total_samples, exact_match_count))
        self.log.logger.info("predict_num: {} gold_num: {} correct_num: {} total_samples: {} exact_match: {}".format(
            predict_num, gold_num, correct_num, total_samples, exact_match_count))
        model.train()
        return accuracy, precision, recall, f1_score, predict

    def test(self) -> None:
        """
        执行模型测试
        """
        try:
            # 构建数据处理器和配置
            processor, bert_config = self.build_processor()
            
            # 加载测试数据集
            dev_dataset = REDataset(self.config, self.config.dev_file, processor)
            dev_dataloader = DataLoader(
                dev_dataset, 
                shuffle=False,  # 测试时不需要打乱
                batch_size=1, 
                collate_fn=collate_fn, 
                pin_memory=True if self.device.type == 'cuda' else False,
                num_workers=getattr(self.config, 'num_workers', 0)
            )
            logger.info(f"测试数据加载完成，样本数: {len(dev_dataset)}")
            
            # 加载模型
            logger.info("正在加载预训练模型...")
            model = OneRel(bert_config)
            
            # 检查检查点文件是否存在
            if not os.path.exists(self.config.checkpoint):
                raise FileNotFoundError(f"检查点文件不存在: {self.config.checkpoint}")
            
            # 加载模型权重
            checkpoint = torch.load(self.config.checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.to(self.device)
            logger.info(f"模型加载完成，使用检查点: {self.config.checkpoint}")
            
            # 执行评估
            accuracy, precision, recall, f1_score, predict = self.evaluate(dev_dataloader, model)
            
            # 保存测试结果
            os.makedirs(os.path.dirname(self.config.test_result), exist_ok=True)
            with open(self.config.test_result, "w", encoding="utf-8") as f:
                json.dump(predict, f, indent=4, ensure_ascii=False)
            
            # 输出测试结果
            logger.info("=== 测试结果 ===")
            logger.info(f"准确率: {accuracy:.4f}")
            logger.info(f"精确率: {precision:.4f}")
            logger.info(f"召回率: {recall:.4f}")
            logger.info(f"F1分数: {f1_score:.4f}")
            logger.info(f"结果已保存至: {self.config.test_result}")
            
            print("=== 测试完成 ===")
            print(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1_score:.4f}")
            
        except Exception as e:
            logger.error(f"测试过程中出错: {e}")
            raise
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        将批次数据移动到指定设备
        
        Args:
            batch: 批次数据字典
            
        Returns:
            Dict[str, Any]: 移动到设备后的批次数据
        """
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device, non_blocking=True)
            else:
                moved_batch[key] = value
        return moved_batch


