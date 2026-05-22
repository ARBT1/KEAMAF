#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态持续学习知识图谱总控调度引擎 (Pipeline Manager)
严格按照 5 个 Task 的流转顺序，执行 阶段一 -> 阶段二 -> 阶段三
"""

import os
import sys
import json
import logging
from datetime import datetime
from collections import defaultdict

# 导入你的模块
from config.config import Configargs
from real_dynamic_continual_learning import RealDynamicContinualLearningFramework
from main import build_data_package, summarize_packages
from KG.kg import build_cypher, execute_cypher, stage3_pipeline, stage4_pipeline, write_html, write_stats, save_mistakes_to_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置统一的输出目录
OUTPUT_DIR = "output"


def setup_output_dir():
    """
    初始化并清空全局输出目录 (OUTPUT_DIR)。
    在每次执行 Pipeline 前，移除旧的 output 目录并重新创建，
    确保每次运行产生的数据结果都是纯净的。
    """
    import shutil
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)


def main():
    """
    系统全局入口主函数。
    串联了知识图谱持续学习的 4 个核心阶段：
    - 阶段一：基于预训练模型与经验回放缓冲区进行增量数据抽取（抗遗忘）
    - 阶段二：抽取结果转化为 Neo4j 图谱节点并检测业务异常
    - 阶段三：基于更新后的图谱，计算材料节点的 State-aware puTransE 嵌入向量
    - 阶段四：利用生成的向量嵌入为指定目标属性进行材料性能/工艺推荐
    """
    setup_output_dir()
    logger.info("========== 开始启动 动态持续学习知识图谱 总控引擎 ==========")

    # 1. 初始化配置和模型
    config = Configargs()
    # 增加 epochs 提升持续学习的抽取精度 (从5提升到15，让RoBERTa能更充分地拟合)
    epochs_per_task = 20

    model_path = r"D:\BaiduNetdiskDownload\知识图谱260407\知识图谱\chinese_roberta_L-12_H-768\chinese_roberta_L-12_H-768"

    framework = RealDynamicContinualLearningFramework(
        config=config,
        replay_buffer_size=1000,
        replay_ratio=0.3,
        model_name=model_path
    )

    data_path = os.path.join('dataset', 'ASaIE', 'train_data.json')
    num_tasks = 5

    # 2. 将数据切分为 5 个 Task（跨阶段划分）
    train_task_data_list, test_task_data_list = framework.load_and_split_data(
        data_path, use_cross_stage=True, num_tasks=num_tasks
    )

    # === 阶段一 AMCER 抽取性能指标数据结构 ===
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(train_task_data_list),
        "total_samples": sum(len(t) for t in train_task_data_list) + sum(len(t) for t in test_task_data_list),
        "total_train_samples": sum(len(t) for t in train_task_data_list),
        "total_test_samples": sum(len(t) for t in test_task_data_list),
        "valid_tasks": len(train_task_data_list)
    }
    extraction_task_details = []
    cumulative_test_data = []

    # 全局变量：用于累积所有被抽出来的三元组数据包（模拟现实中不断增多的文献库）
    global_packages = []
    all_tasks_kg_metrics = []  # 用于收集所有任务的图谱（KG）指标

    # 3. 核心大循环：遍历 5 个任务
    for task_id, task_data in enumerate(train_task_data_list):
        logger.info(f"\n" + "=" * 50)
        logger.info(f"🚀 开始执行 Task {task_id + 1} / {num_tasks}")
        logger.info("=" * 50)

        # =================================================================
        # 【阶段一】：知识抽取 (持续学习抗遗忘训练)
        # =================================================================
        logger.info(f"--> [阶段一] 开始对抗遗忘训练并抽取 Task {task_id + 1} 的三元组...")
        train_result = framework.train_single_task(
            task_id=task_id,
            task_data=task_data,
            epochs=epochs_per_task
        )

        # 阶段一抽取模型评估：把当前的测试集加入累积测试集
        test_data = test_task_data_list[task_id] if task_id < len(test_task_data_list) else []
        cumulative_test_data.extend(test_data)

        # 执行指标评估，保留评价指标
        eval_metrics = framework.evaluate_cumulative_performance(
            cumulative_test_data=cumulative_test_data,
            task_id=task_id
        )

        # 记录阶段一提取模型的任务详情
        extraction_task_details.append({
            "task_id": task_id,
            "train_data_size": len(task_data),
            "test_data_size": len(test_data),
            "cumulative_test_size": len(cumulative_test_data),
            "train_loss": train_result.get('loss', 0.0),
            "cumulative_metrics": eval_metrics
        })

        # 将当前 Task 的数据抽取为 Package 格式
        current_task_packages = []
        for idx, sample in enumerate(task_data):
            pkg = build_data_package(
                sample=sample,
                index=idx,
                task_id=task_id,
                id_field='id',
                time_field='time',
                buffer=framework.replay_buffer
            )
            current_task_packages.append(pkg)

        global_packages.extend(current_task_packages)
        logger.info(f"Task {task_id + 1} 抽取完成！当前图谱累积数据包数量: {len(global_packages)}")

        # 保存累积的阶段一数据包，供后续阶段使用
        stage1_output_path = os.path.join(OUTPUT_DIR, 'stage1_data_packages_cumulative.json')
        with open(stage1_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "packages": global_packages,
                "metrics": summarize_packages(global_packages),
                "source_samples": len(global_packages),
                "num_tasks": task_id + 1
            }, f, ensure_ascii=False, indent=2)

        stage1_metrics_output_path = os.path.join(OUTPUT_DIR, 'stage1_extraction_metrics_cumulative.json')
        with open(stage1_metrics_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment_info": experiment_info,
                "overall_performance": extraction_task_details[-1]["cumulative_metrics"],
                "task_details": extraction_task_details,
                "best_performance": framework.best_performance
            }, f, ensure_ascii=False, indent=2)

        # =================================================================
        # 【阶段二】：动态图谱构建与更新 (Neo4j 交互与状态升级)
        # =================================================================
        logger.info(f"--> [阶段二] 开始将 Task {task_id + 1} 的知识增量写入 Neo4j 图谱...")

        # 基于【当前所有积累的数据】生成 Cypher。
        # 注意：这里 kg.py 内部的逻辑会生效 -> 发现旧知识被新 Task 再次印证，触发 trend_confirmed 升级！
        cypher_statements, anomalies, graph_data = build_cypher(global_packages)

        # 将最新的 Cypher 写入文件
        cypher_output_path = os.path.join(OUTPUT_DIR, f'stage2_import_task{task_id + 1}.cypher')
        with open(cypher_output_path, 'w', encoding='utf-8') as f:
            for stmt in cypher_statements:
                f.write(stmt.strip() + ";\n")

        # 保存到文件
        try:
            filename= os.path.join(OUTPUT_DIR, f'stage2_mistake_task{task_id + 1}.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_mistakes_to_json(anomalies), f, ensure_ascii=False, indent=2)
            print(f"成功保存 {len(anomalies)} 条异常数据到 {filename}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

        # save_mistakes_to_json(anomalies)

        # 自动将数据导入 Neo4j 图数据库
        logger.info(f"--> [阶段二] 正在将 Task {task_id + 1} 的知识自动导入本地 Neo4j 图数据库...")
        try:
            execute_cypher(
                statements=cypher_statements,
                uri="bolt://localhost:7687",
                user="neo4j",
                password="neo4j1234"
            )
            logger.info("图谱数据成功写入 Neo4j 数据库！")
        except Exception as e:
            logger.error(f"写入 Neo4j 数据库失败，请检查数据库连接及密码配置: {e}")

        # 生成阶段二：图谱可视化 HTML 以及异常统计
        total_events = sum(1 for node in graph_data['nodes'] if node.get('type') == 'Event')
        anomaly_ratio = len(anomalies) / total_events if total_events > 0 else 0
        stats = {
            'total_events': total_events,
            'anomalous_events': len(anomalies),
            'anomaly_ratio': anomaly_ratio
        }
        stats_output_path = os.path.join(OUTPUT_DIR, f'stage2_stats_task{task_id + 1}.json')
        html_output_path = os.path.join(OUTPUT_DIR, f'stage2_graph_task{task_id + 1}.html')
        write_stats(stats, stats_output_path)

        # 将 stats 添加到 extraction_task_details 中
        extraction_task_details[-1]["stage2_stats"] = stats

        # 覆盖 stage2_stats_task*.json 使其格式严格一致
        current_stats_results = {
            "experiment_info": experiment_info,
            "overall_performance": extraction_task_details[-1]["cumulative_metrics"],
            "task_details": extraction_task_details,
            "best_performance": framework.best_performance
        }
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            json.dump(current_stats_results, f, ensure_ascii=False, indent=2)

        write_html(graph_data, stats, html_output_path)

        logger.info(f"阶段二处理完毕，检测到异常规则冲突事件: {len(anomalies)} 个。")

        # =================================================================
        # 【阶段三】：状态感知的表示学习 (State-aware puTransE)
        # =================================================================
        logger.info(f"--> [阶段三] 基于更新后的图谱，重新训练材料特征向量...")

        stage3_emb_path = os.path.join(OUTPUT_DIR, f'stage3_embeddings_task{task_id + 1}.json')
        stage3_met_path = os.path.join(OUTPUT_DIR, f'stage3_metrics_task{task_id + 1}.json')
        stage3_html_path = os.path.join(OUTPUT_DIR,
                                        'stage3_triplets_embedding_final.html') if task_id == num_tasks - 1 else None

        metrics = stage3_pipeline(
            packages=global_packages,
            output_embeddings=stage3_emb_path,
            output_metrics=stage3_met_path,
            # 只有在最后一次任务时，才进行三元组可视化
            output_html=stage3_html_path,
            dim=256,  # 提升维度到 256，进一步增强向量表征能力
            epochs=1000,  # TransE 类图谱模型通常需要几百上千轮才能收敛，这里设为 500
            lr=0.001  # 调低学习率至 0.001，避免在复杂空间中震荡，确保收敛更平稳
        )

        logger.info(f"Task {task_id + 1} 向量更新完毕！核心指标 (Sa-puTransE MRR): {metrics['Sa-puTransE']['MRR']:.4f}")

        # 保存这轮生成的图谱指标
        all_tasks_kg_metrics.append({
            "task_id": task_id,
            "metrics": metrics
        })

        # 将 kg_metrics 添加到 extraction_task_details 中
        extraction_task_details[-1]["kg_metrics"] = metrics

        # 覆盖 stage3_metrics_task*.json 使其格式严格一致
        current_metrics_results = {
            "experiment_info": experiment_info,
            "overall_performance": extraction_task_details[-1]["cumulative_metrics"],
            "task_details": extraction_task_details,
            "best_performance": framework.best_performance
        }
        with open(stage3_met_path, 'w', encoding='utf-8') as f:
            json.dump(current_metrics_results, f, ensure_ascii=False, indent=2)

    logger.info("\n========== 所有 5 个 Task 的持续学习全流程结束！ ==========")

    # === 阶段四 AMCER 抽取性能：按照 real_dynamic_continual_learning_results 的格式输出 ===
    final_extraction_results = {
        "experiment_info": experiment_info,
        "overall_performance": extraction_task_details[-1]["cumulative_metrics"] if extraction_task_details else {},
        "task_details": extraction_task_details,
        "best_performance": framework.best_performance
    }
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    extraction_results_filename = os.path.join(OUTPUT_DIR, f'pipeline_overall_results_{timestamp_str}.json')
    with open(extraction_results_filename, 'w', encoding='utf-8') as f:
        json.dump(final_extraction_results, f, ensure_ascii=False, indent=2)
    logger.info(f"==> [需求要求] 阶段一知识抽取持续学习性能指标，已按照指定格式保存至: {extraction_results_filename}")

    logger.info("--> [阶段四] 基于最终版的图谱进行智能推荐与应用评估...")
    output = stage4_pipeline(
        packages=global_packages,
        embeddings_path=os.path.join(OUTPUT_DIR, f'stage3_embeddings_task{num_tasks}.json'),
        output_path=os.path.join(OUTPUT_DIR, 'stage4_recommendations_final.json'),
        target_property='抗拉强度',
        threshold=246,
        top_k=5
    )

    final_extraction_results["recommendations"] = output.get('recommendations',  []),
    final_extraction_results["event"] = output.get('trend_confirmed_events', []),
    with open(os.path.join(OUTPUT_DIR, 'stage4_recommendations_final.json'), 'w', encoding='utf-8') as f:
        json.dump(final_extraction_results, f, ensure_ascii=False, indent=2)


    logger.info(
        f"阶段四结果已生成: {os.path.join(OUTPUT_DIR, 'stage4_recommendations_final.json')}，共推荐 {len(output.get('recommendations', []))} 个合金。")


if __name__ == '__main__':
    main()
