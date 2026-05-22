import json
import argparse
import os
import re
import hashlib
import math
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
from typing import List, Dict, Any, Optional, Tuple, Union


def get_label(entity):
    """
    根据实体名称推断其在知识图谱中的标签类别。

    通过预定义的关键词匹配规则，将材料科学领域的实体分类为合金(Alloy)、
    元素(Element)、性能(Property)、工艺(Process)和数值(Value)等。

    Args:
        entity (str): 实体名称

    Returns:
        str: 实体的标签类别（如 'Alloy', 'Element', 'Entity' 等）
    """
    if entity is None:
        return 'Entity'
    entity_lower = str(entity).lower()
    if '合金' in str(entity) or 'al-' in entity_lower or 'a390' in entity_lower:
        return 'Alloy'
    if any(element in entity_lower for element in ['si', 'al', 'fe', 'cu', 'bi', 'mg', 'ti', 'cr', 'zn', 'ni']):
        return 'Element'
    if any(test in str(entity) for test in ['组织', '金相', '显微', '热分析', '曲线', '性能', '断口', '图片', '硬度', '强度', '延伸率']):
        return 'Property'
    if any(param in str(entity) for param in ['温度', '时间', '压力', '速度', '含量', '变质', '加热', '保温', '时效', '固溶']):
        return 'Process'
    if any(value in entity_lower for value in ['%', '℃', '°c', 'mpa', 'gpa', 'min', 's', 'μm', 'mm', 'r/min', 'k']):
        return 'Value'
    return 'Entity'


relation_map = {
    "Ele-Alloy": "CONTAINS_ELEMENT",
    "Alloy-Test_n": "HAS_TEST",
    "Con-Ele": "HAS_CONCENTRATION",
    "Par_n-Par_v": "HAS_PARAMETER_VALUE",
    "Exp_r-Par_n": "EXPERIMENT_RESULT",
    "Test_n-Test_f": "TEST_RESULT_IMAGE",
    "Test_n-Par_n": "TEST_PARAMETER",
    "Exp-Par_n": "EXPERIMENT_PARAMETER",
    "Alloy-Exp": "ALLOY_EXPERIMENT",
    "Test_n-Phase": "TEST_PHASE",
    "Test_n-Test_v": "TEST_VALUE"
}


def cypher_value(value):
    """
    将 Python 变量值转换为安全的 Neo4j Cypher 查询字符串格式。

    Args:
        value (Any): 需要转换的值

    Returns:
        str: 转义后的 Cypher 字符串表示
    """
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def parse_numeric(text):
    """
    从字符串中解析出第一个数值（支持负数和小数）。

    主要用于异常检测时的数值比较。

    Args:
        text (str): 包含数字的原始文本

    Returns:
        float | None: 提取出的浮点数，未找到则返回 None
    """
    if not text:
        return None
    match = re.search(r'(-?\d+(?:\.\d+)?)', str(text))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def build_event_id(source_id, index, head, relation, tail):
    """
    基于三元组信息构建全局唯一的事件 ID (MD5哈希值)。

    Args:
        source_id (str): 来源文献或数据包的 ID
        index (int): 在数据包中的索引
        head (str): 头实体名称
        relation (str): 关系类型
        tail (str): 尾实体名称

    Returns:
        str: 32 位 MD5 哈希字符串
    """
    raw = f"{source_id}|{index}|{head}|{relation}|{tail}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()

def parse_numeric(text: str) -> Optional[float]:
    """
    从文本中解析第一个数值

    Args:
        text: 输入文本

    Returns:
        解析到的数值，如果没有找到则返回None
    """
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    matches = re.findall(pattern, text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None

def parse_numeric_with_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    从文本中解析数值和单位

    Args:
        text: 输入文本

    Returns:
        (数值, 单位) 元组
    """
    # 常见单位模式
    unit_patterns = [
        (r'(\d+(?:\.\d+)?)\s*(°C|℃)', '°C'),  # 摄氏度
        (r'(\d+(?:\.\d+)?)\s*K\b', 'K'),  # 开尔文
        (r'(\d+(?:\.\d+)?)\s*MPa', 'MPa'),  # 兆帕
        (r'(\d+(?:\.\d+)?)\s*GPa', 'GPa'),  # 吉帕
        (r'(\d+(?:\.\d+)?)\s*g/cm³', 'g/cm³'),  # 密度
        (r'(\d+(?:\.\d+)?)\s*g/cm3', 'g/cm³'),
        (r'(\d+(?:\.\d+)?)\s*%', '%'),  # 百分比
        (r'(\d+(?:\.\d+)?)\s*10⁻⁶/K', '10⁻⁶/K'),  # 热膨胀系数
        (r'(\d+(?:\.\d+)?)\s*10\^-6/K', '10⁻⁶/K'),
        (r'(\d+(?:\.\d+)?)\s*Å', 'Å'),  # 埃
        (r'(\d+(?:\.\d+)?)\s*S/m', 'S/m'),  # 电导率
    ]

    for pattern, unit in unit_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                value = float(match.group(1))
                return value, unit
            except ValueError:
                pass

    # 如果没有匹配到带单位的模式，尝试解析纯数值
    value = parse_numeric(text)
    return value, None

def detect_anomalies(triplet, raw_sentence):
    """
    基于合金材料科学专家规则进行知识图谱异常检测。

    选择5个最具代表性的合金材料物理规则：
    1. 合金成分合理性规则（元素含量范围）
    2. 相变温度规则（熔点/固相线温度）
    3. 力学性能规则（屈服/抗拉强度）

    Args:
        triplet: 抽取的三元组数据
        raw_sentence: 抽取该三元组的原始句子

    Returns:
        list: 异常检测结果列表，每个元素是一个异常记录字典
    """
    anomalies = []
    text = raw_sentence or ''

    # 规则1: 合金成分合理性规则
    # 检测常见合金元素含量是否在合理范围内
    alloy_elements = {
        'Al': {'min': 0, 'max': 100, 'typical_max': 15, 'unit': '%'},
        'Si': {'min': 0, 'max': 30, 'typical_max': 25, 'unit': '%'},
        'Fe': {'min': 0, 'max': 5, 'typical_max': 2, 'unit': '%'},
        'Cu': {'min': 0, 'max': 10, 'typical_max': 5, 'unit': '%'},
        'Mg': {'min': 0, 'max': 10, 'typical_max': 6, 'unit': '%'},
        'Zn': {'min': 0, 'max': 12, 'typical_max': 8, 'unit': '%'},
        'Mn': {'min': 0, 'max': 2, 'typical_max': 1.5, 'unit': '%'},
        'Ti': {'min': 0, 'max': 0.3, 'typical_max': 0.2, 'unit': '%'},
        'Cr': {'min': 0, 'max': 0.5, 'typical_max': 0.3, 'unit': '%'},
        'Ni': {'min': 0, 'max': 3, 'typical_max': 2, 'unit': '%'},
    }

    for element, specs in alloy_elements.items():
        if element in text or element.lower() in text.lower():
            # 查找元素含量
            patterns = [
                rf'{element}\s*[:：]?\s*(\d+(?:\.\d+)?)\s*%',
                rf'{element}\s*(\d+(?:\.\d+)?)\s*wt%',
                rf'{element}\s*(\d+(?:\.\d+)?)\s*wt\.%',
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        content = float(match.group(1))

                        # 检查是否超出物理极限
                        if content < specs['min'] or content > specs['max']:
                            anomalies.append({
                                'rule_name': 'alloy_composition_range',
                                'rule_description': f'{element}元素含量超出物理极限范围({specs["min"]}-{specs["max"]}{specs["unit"]})',
                                'material': triplet.get('head', '未知材料'),
                                'element': element,
                                'detected_value': f'{content}{specs["unit"]}',
                                'expected_range': f'{specs["min"]}-{specs["max"]}{specs["unit"]}',
                                'severity': 'high',
                                'sentence': text,
                                'triplet': triplet
                            })
                        # 检查是否超出典型范围
                        elif content > specs['typical_max']:
                            anomalies.append({
                                'rule_name': 'alloy_composition_typical',
                                'rule_description': f'{element}元素含量超出典型合金范围(> {specs["typical_max"]}{specs["unit"]})',
                                'material': triplet.get('head', '未知材料'),
                                'element': element,
                                'detected_value': f'{content}{specs["unit"]}',
                                'expected_range': f'≤{specs["typical_max"]}{specs["unit"]}',
                                'severity': 'medium',
                                'sentence': text,
                                'triplet': triplet
                            })
                    except ValueError:
                        pass

    # 规则2: 相变温度规则
    # 检测合金的熔点/固相线温度是否合理
    temperature_keywords = ['熔点', 'melting point', '融点', '液相线', 'liquidus', '固相线', 'solidus']

    if any(keyword in text.lower() for keyword in temperature_keywords):
        temp_value, temp_unit = parse_numeric_with_unit(text)

        if temp_value is not None:
            # 常见合金的熔点范围
            alloy_melting_ranges = {
                '铝': {'min': 500, 'max': 700, 'typical': 660, 'unit': '°C'},
                '铝合金': {'min': 500, 'max': 650, 'typical': 580, 'unit': '°C'},
                '铜': {'min': 1000, 'max': 1200, 'typical': 1085, 'unit': '°C'},
                '黄铜': {'min': 900, 'max': 1000, 'typical': 950, 'unit': '°C'},
                '青铜': {'min': 900, 'max': 1050, 'typical': 980, 'unit': '°C'},
                '钢': {'min': 1400, 'max': 1550, 'typical': 1500, 'unit': '°C'},
                '不锈钢': {'min': 1400, 'max': 1530, 'typical': 1450, 'unit': '°C'},
                '钛': {'min': 1600, 'max': 1700, 'typical': 1668, 'unit': '°C'},
                '镁': {'min': 600, 'max': 700, 'typical': 650, 'unit': '°C'},
            }

            # 单位转换
            if temp_unit == 'K' and '°C' not in text and '℃' not in text:
                temp_value = temp_value - 273.15
                temp_unit = '°C'

            # 通用物理极限
            if temp_value < 0 or temp_value > 4000:
                anomalies.append({
                    'rule_name': 'melting_point_physical_limit',
                    'rule_description': f'熔点超出物理极限范围(0-4000°C)',
                    'material': triplet.get('head', '未知材料'),
                    'property': 'melting_point',
                    'detected_value': f'{temp_value}{temp_unit or "°C"}',
                    'expected_range': '0-4000°C',
                    'severity': 'high',
                    'sentence': text,
                    'triplet': triplet
                })

            # 检查具体合金
            for alloy, specs in alloy_melting_ranges.items():
                if alloy in text:
                    if temp_value < specs['min'] or temp_value > specs['max']:
                        anomalies.append({
                            'rule_name': f'{alloy}_melting_point',
                            'rule_description': f'{alloy}熔点超出合理范围({specs["min"]}-{specs["max"]}{specs["unit"]})',
                            'material': alloy,
                            'property': 'melting_point',
                            'detected_value': f'{temp_value}{temp_unit or specs["unit"]}',
                            'expected_range': f'{specs["min"]}-{specs["max"]}{specs["unit"]}',
                            'severity': 'medium',
                            'sentence': text,
                            'triplet': triplet
                        })
                    break

    # 规则3: 力学性能规则
    # 检测合金的屈服/抗拉强度是否合理
    strength_keywords = ['抗拉强度', 'tensile strength', '屈服强度', 'yield strength', '强度', 'strength']
    strength_units = ['MPa', 'GPa', '兆帕', '吉帕']

    if any(keyword in text.lower() for keyword in strength_keywords) and any(unit in text for unit in strength_units):
        strength_value, strength_unit = parse_numeric_with_unit(text)

        if strength_value is not None:
            # 单位转换
            if strength_unit == 'GPa':
                strength_value = strength_value * 1000
                strength_unit = 'MPa'

            # 常见合金的强度范围
            alloy_strength_ranges = {
                '纯铝': {'min': 40, 'max': 100, 'typical': 70, 'unit': 'MPa'},
                '铝合金': {'min': 100, 'max': 700, 'typical': 300, 'unit': 'MPa'},
                '纯铜': {'min': 200, 'max': 250, 'typical': 220, 'unit': 'MPa'},
                '黄铜': {'min': 300, 'max': 600, 'typical': 450, 'unit': 'MPa'},
                '青铜': {'min': 350, 'max': 700, 'typical': 500, 'unit': 'MPa'},
                '低碳钢': {'min': 300, 'max': 500, 'typical': 400, 'unit': 'MPa'},
                '中碳钢': {'min': 500, 'max': 800, 'typical': 650, 'unit': 'MPa'},
                '高碳钢': {'min': 800, 'max': 1500, 'typical': 1000, 'unit': 'MPa'},
                '不锈钢': {'min': 500, 'max': 2000, 'typical': 800, 'unit': 'MPa'},
                '钛合金': {'min': 800, 'max': 1400, 'typical': 1000, 'unit': 'MPa'},
                '镁合金': {'min': 150, 'max': 400, 'typical': 250, 'unit': 'MPa'},
            }

            # 通用物理极限
            if strength_value < 10 or strength_value > 5000:
                anomalies.append({
                    'rule_name': 'strength_physical_limit',
                    'rule_description': f'强度值超出物理极限范围(10-5000 MPa)',
                    'material': triplet.get('head', '未知材料'),
                    'property': 'strength',
                    'detected_value': f'{strength_value}{strength_unit or "MPa"}',
                    'expected_range': '10-5000 MPa',
                    'severity': 'high',
                    'sentence': text,
                    'triplet': triplet
                })

            # 检查具体合金
            for alloy, specs in alloy_strength_ranges.items():
                if alloy in text:
                    if strength_value < specs['min'] or strength_value > specs['max']:
                        anomalies.append({
                            'rule_name': f'{alloy}_strength',
                            'rule_description': f'{alloy}强度超出合理范围({specs["min"]}-{specs["max"]}{specs["unit"]})',
                            'material': alloy,
                            'property': 'tensile_strength',
                            'detected_value': f'{strength_value}{strength_unit or specs["unit"]}',
                            'expected_range': f'{specs["min"]}-{specs["max"]}{specs["unit"]}',
                            'severity': 'medium',
                            'sentence': text,
                            'triplet': triplet
                        })
                    break

    # # 规则4: 热物理性能规则
    # # 检测合金的热膨胀系数是否合理
    # thermal_keywords = ['热膨胀系数', 'CTE', 'thermal expansion', '膨胀系数']
    #
    # if any(keyword in text.lower() for keyword in thermal_keywords):
    #     cte_value, cte_unit = parse_numeric_with_unit(text)
    #
    #     if cte_value is not None:
    #         # 常见合金的热膨胀系数范围(10⁻⁶/K)
    #         alloy_cte_ranges = {
    #             '铝': {'min': 22, 'max': 25, 'typical': 23.1, 'unit': '10⁻⁶/K'},
    #             '铝合金': {'min': 20, 'max': 25, 'typical': 23, 'unit': '10⁻⁶/K'},
    #             '铜': {'min': 16, 'max': 18, 'typical': 16.5, 'unit': '10⁻⁶/K'},
    #             '黄铜': {'min': 18, 'max': 20, 'typical': 19, 'unit': '10⁻⁶/K'},
    #             '钢': {'min': 10, 'max': 13, 'typical': 11, 'unit': '10⁻⁶/K'},
    #             '不锈钢': {'min': 16, 'max': 18, 'typical': 17, 'unit': '10⁻⁶/K'},
    #             '钛': {'min': 8, 'max': 10, 'typical': 8.6, 'unit': '10⁻⁶/K'},
    #             '镁': {'min': 24, 'max': 26, 'typical': 25, 'unit': '10⁻⁶/K'},
    #             '因瓦合金': {'min': 0.5, 'max': 2, 'typical': 1.2, 'unit': '10⁻⁶/K'},
    #         }
    #
    #
    #         # 通用物理极限
    #         if cte_value < 0 or cte_value > 30:
    #             anomalies.append({
    #                 'rule_name': 'cte_physical_limit',
    #                 'rule_description': f'热膨胀系数超出物理极限范围(0-30×10⁻⁶/K)',
    #                 'material': triplet.get('head', '未知材料'),
    #                 'property': 'coefficient_of_thermal_expansion',
    #                 'detected_value': f'{cte_value}{cte_unit or "10⁻⁶/K"}',
    #                 'expected_range': '0-30×10⁻⁶/K',
    #                 'severity': 'high',
    #                 'sentence': text,
    #                 'triplet': triplet
    #             })
    #
    #         # 检查具体合金
    #         for alloy, specs in alloy_cte_ranges.items():
    #             if alloy in text:
    #                 if cte_value < specs['min'] or cte_value > specs['max']:
    #                     anomalies.append({
    #                         'rule_name': f'{alloy}_cte',
    #                         'rule_description': f'{alloy}热膨胀系数超出合理范围({specs["min"]}-{specs["max"]}{specs["unit"]})',
    #                         'material': alloy,
    #                         'property': 'coefficient_of_thermal_expansion',
    #                         'detected_value': f'{cte_value}{cte_unit or specs["unit"]}',
    #                         'expected_range': f'{specs["min"]}-{specs["max"]}{specs["unit"]}',
    #                         'severity': 'medium',
    #                         'sentence': text,
    #                         'triplet': triplet
    #                     })
    #                 break

    return anomalies

def save_mistakes_to_json(anomalies: List[Dict[str, Any]], filename: str = "mistake.json") -> None:
    """
    将检测到的异常数据保存到JSON文件

    Args:
        anomalies: 异常检测结果列表
        filename: 保存的文件名
    """
    if not anomalies:
        print(f"未检测到异常数据，不生成 {filename} 文件")
        return

    # 准备保存的数据结构
    mistake_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_mistakes": len(anomalies),
            "description": "合金材料知识图谱异常检测结果"
        },
        "mistakes": []
    }

    # 对异常进行分类统计
    rule_stats = {}
    severity_stats = {"high": 0, "medium": 0, "low": 0}

    for anomaly in anomalies:
        # 简化triplet数据，避免序列化问题
        simplified_anomaly = anomaly.copy()
        if 'triplet' in simplified_anomaly:
            # 确保triplet可以被JSON序列化
            simplified_anomaly['triplet'] = dict(simplified_anomaly['triplet'])

        mistake_data["mistakes"].append(simplified_anomaly)

        # 统计规则出现次数
        rule_name = anomaly.get('rule_name', 'unknown')
        rule_stats[rule_name] = rule_stats.get(rule_name, 0) + 1

        # 统计严重程度
        severity = anomaly.get('severity', 'medium')
        severity_stats[severity] = severity_stats.get(severity, 0) + 1

    # 添加统计信息
    mistake_data["statistics"] = {
        "rules": rule_stats,
        "severity": severity_stats
    }
    return mistake_data

    # # 保存到文件
    # try:
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         json.dump(mistake_data, f, ensure_ascii=False, indent=2)
    #     print(f"成功保存 {len(anomalies)} 条异常数据到 {filename}")
    # except Exception as e:
    #     print(f"保存文件时出错: {e}")


def load_packages(path):
    """
    从指定 JSON 文件加载由抽取模型生成的阶段一知识数据包。

    Args:
        path (str): 数据包 JSON 文件的路径

    Returns:
        list: 包含所有知识数据包字典的列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('packages', [])


def build_cypher(packages):
    """
    将提取的三元组数据包转换为 Neo4j Cypher 插入/更新语句。

    该过程涉及：
    1. 实体与关系的去重与规范化
    2. 基于事件节点 (Event) 的具体语境构建，支持属性挂载
    3. 异常规则的检测并标记异常事件
    4. 构建用于可视化的网络图节点和边数据

    Args:
        packages (list): 数据包列表

    Returns:
        tuple: (
            cypher_statements: list[str] -> Cypher 执行语句列表,
            anomalies: list[dict] -> 检测到的异常事件列表,
            graph_data: dict -> 前端可视化的节点(nodes)和边(links)数据字典
        )
    """
    cypher_statements = []
    cypher_statements.append("CREATE INDEX event_id IF NOT EXISTS FOR (e:Event) ON (e.id);")
    cypher_statements.append("CREATE INDEX literature_id IF NOT EXISTS FOR (l:Literature) ON (l.id);")
    cypher_statements.append("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);")

    events_by_key = defaultdict(list)
    anomalies = []
    graph_nodes = {}
    graph_links = []

    for pkg in packages:
        source_id = pkg.get('id')
        task_id = pkg.get('task')
        timestamp = pkg.get('time')
        triplets = pkg.get('triplets', [])
        for idx, tri in enumerate(triplets):
            head = tri.get('head')
            relation = tri.get('relation')
            tail = tri.get('tail')
            raw_sentence = tri.get('raw_sentence')
            confidence = tri.get('confidence')
            quantitative_value = tri.get('quantitative_value')
            unit = tri.get('unit')
            event_id = build_event_id(source_id, idx, head, relation, tail)
            description = f"{head}-{relation}-{tail}"

            event_props = {
                'id': event_id,
                'description': description,
                'quantitative_value': quantitative_value,
                'unit': unit,
                'knowledge_state': 'reported',
                'confidence': confidence,
                'derived_from_sentence': raw_sentence,
                'timestamp': timestamp,
                'task': task_id,
                'source_id': source_id,
                'relation': relation,
                'head': head,
                'tail': tail
            }

            labels = {
                'head': get_label(head),
                'tail': get_label(tail)
            }

            cypher_statements.append(
                f"MERGE (e:Event {{id: {cypher_value(event_id)}}}) "
                f"SET e.description={cypher_value(description)}, "
                f"e.quantitative_value={cypher_value(quantitative_value)}, "
                f"e.unit={cypher_value(unit)}, "
                f"e.knowledge_state='reported', "
                f"e.confidence={cypher_value(confidence)}, "
                f"e.derived_from_sentence={cypher_value(raw_sentence)}, "
                f"e.timestamp={cypher_value(timestamp)}, "
                f"e.task={cypher_value(task_id)}, "
                f"e.source_id={cypher_value(source_id)}, "
                f"e.relation={cypher_value(relation)}, "
                f"e.head={cypher_value(head)}, "
                f"e.tail={cypher_value(tail)}"
            )

            cypher_statements.append(
                f"MERGE (h:{labels['head']} {{name: {cypher_value(head)}}})"
            )
            cypher_statements.append(
                f"MERGE (t:{labels['tail']} {{name: {cypher_value(tail)}}})"
            )

            cypher_statements.append(
                f"MERGE (e:Event {{id: {cypher_value(event_id)}}})-[:CONTEXT_FOR {{role: 'subject'}}]->(h:{labels['head']} {{name: {cypher_value(head)}}})"
            )

            if labels['tail'] in ['Property', 'Test', 'Microstructure']:
                cypher_statements.append(
                    f"MERGE (e:Event {{id: {cypher_value(event_id)}}})-[:INSTANCE_OF]->(t:{labels['tail']} {{name: {cypher_value(tail)}}})"
                )
            else:
                cypher_statements.append(
                    f"MERGE (e:Event {{id: {cypher_value(event_id)}}})-[:UNDER_CONDITION]->(t:{labels['tail']} {{name: {cypher_value(tail)}}})"
                )

            if source_id:
                cypher_statements.append(
                    f"MERGE (l:Literature {{id: {cypher_value(source_id)}}})"
                )
                cypher_statements.append(
                    f"MERGE (e:Event {{id: {cypher_value(event_id)}}})-[:CITES]->(l:Literature {{id: {cypher_value(source_id)}}})"
                )

            mapped_relation = relation_map.get(relation, relation.replace('-', '_').upper())
            cypher_statements.append(
                f"MERGE (h:{labels['head']} {{name: {cypher_value(head)}}})-[:{mapped_relation} {{source_event: {cypher_value(event_id)}}}]->(t:{labels['tail']} {{name: {cypher_value(tail)}}})"
            )

            key = (labels['head'], str(head), str(relation), str(tail))
            events_by_key[key].append(event_id)

            rules_hit = detect_anomalies(tri, raw_sentence)
            if rules_hit:
                anomalies.append({'event_id': event_id, 'rules': rules_hit})
                cypher_statements.append(
                    f"MERGE (e:Event {{id: {cypher_value(event_id)}}}) "
                    f"SET e.anomalous=true, e.violation_rules={cypher_value(rules_hit)}"
                )

            event_node_id = f"event:{event_id}"
            head_node_id = f"entity:{head}"
            tail_node_id = f"entity:{tail}"
            if source_id:
                lit_node_id = f"lit:{source_id}"
                if lit_node_id not in graph_nodes:
                    graph_nodes[lit_node_id] = {
                        'id': lit_node_id,
                        'label': str(source_id),
                        'type': 'Literature'
                    }
                graph_links.append({
                    'source': event_node_id,
                    'target': lit_node_id,
                    'type': 'CITES'
                })

            if event_node_id not in graph_nodes:
                graph_nodes[event_node_id] = {
                    'id': event_node_id,
                    'label': description,
                    'type': 'Event',
                    'confidence': confidence,
                    'anomalous': bool(rules_hit)
                }
            if head_node_id not in graph_nodes:
                graph_nodes[head_node_id] = {
                    'id': head_node_id,
                    'label': str(head),
                    'type': labels['head']
                }
            if tail_node_id not in graph_nodes:
                graph_nodes[tail_node_id] = {
                    'id': tail_node_id,
                    'label': str(tail),
                    'type': labels['tail']
                }

            graph_links.append({
                'source': event_node_id,
                'target': head_node_id,
                'type': 'CONTEXT_FOR'
            })
            graph_links.append({
                'source': event_node_id,
                'target': tail_node_id,
                'type': 'INSTANCE_OF' if labels['tail'] in ['Property', 'Test', 'Microstructure'] else 'UNDER_CONDITION'
            })
            mapped_relation = relation_map.get(relation, relation.replace('-', '_').upper())
            graph_links.append({
                'source': head_node_id,
                'target': tail_node_id,
                'type': mapped_relation
            })

    for key, event_ids in events_by_key.items():
        if len(event_ids) >= 3:
            ids_list = "[" + ",".join(cypher_value(eid) for eid in event_ids) + "]"
            cypher_statements.append(
                f"MATCH (e:Event) WHERE e.id IN {ids_list} SET e.knowledge_state='trend_confirmed'"
            )
            cypher_statements.append(
                f"UNWIND {ids_list} AS eid1 UNWIND {ids_list} AS eid2 "
                f"WITH eid1, eid2 WHERE eid1 < eid2 "
                f"MATCH (e1:Event {{id: eid1}}), (e2:Event {{id: eid2}}) "
                f"MERGE (e1)-[:CORROBORATES]->(e2)"
            )

    for key, event_ids in events_by_key.items():
        if len(event_ids) >= 5:
            ids_list = "[" + ",".join(cypher_value(eid) for eid in event_ids) + "]"
            cypher_statements.append(
                f"MATCH (e:Event) WHERE e.id IN {ids_list} SET e.knowledge_state='validated'"
            )
            cypher_statements.append(
                f"UNWIND {ids_list} AS eid1 UNWIND {ids_list} AS eid2 "
                f"WITH eid1, eid2 WHERE eid1 < eid2 "
                f"MATCH (e1:Event {{id: eid1}}), (e2:Event {{id: eid2}}) "
                f"MERGE (e1)-[:CORROBORATES]->(e2)"
            )

    graph_data = {
        'nodes': list(graph_nodes.values()),
        'links': graph_links
    }
    return cypher_statements, anomalies, graph_data


def write_cypher(statements, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for stmt in statements:
            f.write(stmt.strip() + ";\n")


def write_stats(stats, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def write_html(graph_data, stats, output_path):
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>阶段二图谱可视化</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Microsoft YaHei",Arial,Helvetica,sans-serif;background:#f3f6fb;color:#1f2937}}
#info{{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:16px;
  padding:14px 18px;
  background:linear-gradient(135deg,#ffffff,#eef4ff);
  border-bottom:1px solid #d8e1f0;
  box-shadow:0 2px 10px rgba(15,23,42,0.06);
}}
.panel-title{{font-size:18px;font-weight:700;color:#0f172a;margin-bottom:4px}}
.panel-subtitle{{font-size:12px;color:#64748b}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;justify-content:flex-end}}
.stat-card{{
  min-width:130px;
  padding:10px 12px;
  background:#ffffff;
  border:1px solid #dbe4f3;
  border-radius:12px;
  box-shadow:0 6px 18px rgba(148,163,184,0.12);
}}
.stat-label{{font-size:12px;color:#64748b;margin-bottom:4px}}
.stat-value{{font-size:20px;font-weight:700;color:#0f172a}}
#chart{{width:100vw;height:calc(100vh - 88px);display:block;cursor:grab;background:radial-gradient(circle at top,#f8fbff 0%,#edf3fb 55%,#e6eef8 100%)}}
#chart:active{{cursor:grabbing}}
.viewport .link{{stroke:#94a3b8;stroke-opacity:0.5}}
.viewport .node{{stroke:#fff;stroke-width:1.6px}}
.viewport text{{font-size:11px;fill:#334155;paint-order:stroke;stroke:#ffffff;stroke-width:3px;stroke-linejoin:round}}
.legend-note{{position:fixed;right:16px;bottom:16px;padding:10px 12px;background:rgba(255,255,255,0.92);border:1px solid #dbe4f3;border-radius:10px;color:#475569;font-size:12px;box-shadow:0 6px 16px rgba(15,23,42,0.08)}}
</style>
</head>
<body>
<div id="info">
  <div>
    <div class="panel-title">阶段二图谱可视化</div>
    <div class="panel-subtitle">滚轮缩放，按住空白处拖动画布，拖动节点可调整局部布局</div>
  </div>
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">节点数</div>
      <div class="stat-value">{len(graph_data['nodes'])}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">关系数</div>
      <div class="stat-value">{len(graph_data['links'])}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">异常事件</div>
      <div class="stat-value">{stats['anomalous_events']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">异常比例</div>
      <div class="stat-value">{stats['anomaly_ratio']:.2%}</div>
    </div>
  </div>
</div>
<svg id="chart"></svg>
<div class="legend-note">红色事件节点表示异常知识事件</div>
<script>
const graph = {json.dumps(graph_data, ensure_ascii=False)};
const width = window.innerWidth;
const height = window.innerHeight - 88;
const svg = d3.select("#chart").attr("width", width).attr("height", height);
svg.insert("rect", ":first-child")
  .attr("width", width)
  .attr("height", height)
  .attr("fill", "transparent");
const viewport = svg.append("g").attr("class", "viewport");
const zoomLayer = viewport.append("g").attr("class", "zoom-layer");
const color = d3.scaleOrdinal()
  .domain(["Event","Alloy","Element","Property","Process","Value","Entity","Literature"])
  .range(["#fb7185","#3b82f6","#22c55e","#f59e0b","#8b5cf6","#f97316","#94a3b8","#38bdf8"]);

const zoom = d3.zoom()
  .scaleExtent([0.2, 6])
  .on("zoom", (event) => {{
    zoomLayer.attr("transform", event.transform);
  }});

svg.call(zoom).on("dblclick.zoom", null);

svg.call(
  zoom.transform,
  d3.zoomIdentity.translate(width * 0.12, height * 0.08).scale(0.92)
);

const simulation = d3.forceSimulation(graph.nodes)
  .force("link", d3.forceLink(graph.links).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-120))
  .force("center", d3.forceCenter(width / 2, height / 2));

const link = zoomLayer.append("g")
  .attr("class","links")
  .selectAll("line")
  .data(graph.links)
  .enter().append("line")
  .attr("class","link")
  .attr("stroke-width", d => d.type === "CITES" ? 1 : 1.4);

const node = zoomLayer.append("g")
  .attr("class","nodes")
  .selectAll("circle")
  .data(graph.nodes)
  .enter().append("circle")
  .attr("class","node")
  .attr("r", d => d.type === "Event" ? 7 : (d.type === "Literature" ? 5 : 4.5))
  .attr("fill", d => d.type === "Event" && d.anomalous ? "#dc2626" : color(d.type || "Entity"))
  .call(
    d3.drag()
      .on("start", (event, d) => {{
        if (event.sourceEvent) event.sourceEvent.stopPropagation();
        dragstarted(event, d);
      }})
      .on("drag", dragged)
      .on("end", dragended)
  );

const label = zoomLayer.append("g")
  .selectAll("text")
  .data(graph.nodes)
  .enter().append("text")
  .attr("dx", 6)
  .attr("dy", 4)
  .text(d => d.label ? String(d.label).slice(0, 18) : "");

node.append("title").text(d => d.label);

simulation.on("tick", () => {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);
  node
    .attr("cx", d => d.x)
    .attr("cy", d => d.y);
  label
    .attr("x", d => d.x)
    .attr("y", d => d.y);
}});

function dragstarted(event, d) {{
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}}
function dragged(event, d) {{
  d.fx = event.x;
  d.fy = event.y;
}}
function dragended(event, d) {{
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}}
</script>
</body>
</html>"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def execute_cypher(statements, uri, user, password):
    try:
        from neo4j import GraphDatabase
    except Exception:
        raise RuntimeError("未安装neo4j驱动，无法执行导入")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for stmt in statements:
            session.run(stmt)
    driver.close()


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def parse_year(value, fallback_year=2023):
    if value is None:
        return fallback_year
    text = str(value)
    match = re.search(r'(19|20)\d{2}', text)
    if match:
        return int(match.group(0))
    return fallback_year


def build_training_samples(packages):
    key_counts = defaultdict(int)
    for pkg in packages:
        for tri in pkg.get('triplets', []):
            key = (tri.get('head'), tri.get('relation'), tri.get('tail'))
            key_counts[key] += 1
    samples = []
    for pkg in packages:
        source_id = pkg.get('id')
        timestamp = pkg.get('time')
        sentence = pkg.get('raw_sentence')
        for idx, tri in enumerate(pkg.get('triplets', [])):
            head = tri.get('head')
            relation = tri.get('relation')
            tail = tri.get('tail')
            event_id = build_event_id(source_id, idx, head, relation, tail)
            key = (head, relation, tail)
            if key_counts[key] > 5 :
                state = 'validated'
            elif key_counts[key] < 2:
                state = 'reported'
            else:
                state = 'trend_confirmed'
            # state = 'trend_confirmed' if key_counts[key] >= 3 else 'reported'
            year = parse_year(timestamp)
            samples.append({
                'event_id': event_id,
                'head': head,
                'relation': relation,
                'tail': tail,
                'state': state,
                'year': year,
                'sentence': sentence
            })
    return samples


def build_spaces(triples, max_spaces=9):
    # 使用NetworkX构建图
    G = nx.Graph()
    for tri in triples:
        G.add_edge(tri['head'], tri['tail'])

    # 使用Louvain算法进行社区发现
    try:
        partition = community_louvain.best_partition(G)
    except Exception as e:
        print(f"Louvain算法执行失败: {e}，回退到连通分量")
        # 降级方案：连通分量
        partition = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                partition[node] = i

    # 将三元组分配到社区
    component_map = defaultdict(list)
    for tri in triples:
        # 以头实体的社区作为三元组的社区
        comm_id = partition.get(tri['head'], 0)
        component_map[comm_id].append(tri)

    components = sorted(component_map.items(), key=lambda x: len(x[1]), reverse=True)
    space_map = {}
    space_id = 0
    for idx, (root, group) in enumerate(components):
        if idx < max_spaces - 1:
            for tri in group:
                space_map[tri['event_id']] = space_id
            space_id += 1
        else:
            for tri in group:
                space_map[tri['event_id']] = max_spaces - 1
    return space_map


def state_weight(state):
    """
    根据知识三元组在图谱中的可靠度状态赋予权重。

    在状态感知的嵌入学习 (State-aware puTransE) 阶段中，越被印证的知识其更新权重越高。

    Args:
        state (str): 知识节点的状态标志

    Returns:
        float: 对应状态赋予的浮点权重值
    """
    mapping = {
        'validated': 2.5,
        'trend_confirmed': 1.5,
        'reported': 1.0,
        'superseded': 0.8,
        'contradicted': 0.5
    }
    return mapping.get(state, 1.0)


def time_weight(year, current_year=2026, decay=0.1):
    """
    计算基于时间衰减的图谱知识权重。

    用于处理材料科学研究中旧知识过时问题，距离 current_year 越远的数据权重越低。

    Args:
        year (int): 知识文献的发表年份
        current_year (int): 当前年份基准
        decay (float): 指数衰减系数

    Returns:
        float: 计算得到的时间权重值
    """
    return math.exp(-decay * (current_year - year))


def init_embeddings(num, dim):
    """
    随机初始化节点的向量表示矩阵。

    Args:
        num (int): 节点数量
        dim (int): 向量维度

    Returns:
        numpy.ndarray: 初始化的 Float32 矩阵 [num, dim]
    """
    return np.random.uniform(-0.1, 0.1, size=(num, dim)).astype(np.float32)


def train_transe(triples, spaces, dim=64, lr=0.01, margin=1.0, epochs=20, use_state=True, use_time=True):
    """
    训练 State-aware puTransE 模型，学习图谱实体的嵌入向量表示。

    基于平移假设 (h + r ≈ t) 以及知识的状态(trend_confirmed)与时效性进行带权重的梯度下降。
    支持将大图谱拆分到不同的局部网络社区(spaces)进行模块化学习以提升效率。

    Args:
        triples (list): 三元组数据列表
        spaces (dict): 各个事件归属的网络社区空间映射
        dim (int): 嵌入向量的维度大小
        lr (float): 学习率
        margin (float): Hinge Loss 的边界边距
        epochs (int): 训练迭代轮数
        use_state (bool): 是否启用状态感知的权重增强
        use_time (bool): 是否启用基于时间衰减的权重机制

    Returns:
        dict: 包含各局部空间(Space)训练好的节点、关系词典和嵌入矩阵 (Embeddings)
    """
    space_triples = defaultdict(list)
    for tri in triples:
        space_triples[spaces.get(tri['event_id'], 0)].append(tri)
    space_models = {}
    for space_id, items in space_triples.items():
        entities = sorted({tri['head'] for tri in items} | {tri['tail'] for tri in items})
        relations = sorted({tri['relation'] for tri in items})
        ent2id = {e: i for i, e in enumerate(entities)}
        rel2id = {r: i for i, r in enumerate(relations)}
        e_emb = init_embeddings(len(entities), dim)
        r_emb = init_embeddings(len(relations), dim)
        for _ in range(epochs):
            random.shuffle(items)
            for tri in items:
                h = ent2id[tri['head']]
                t = ent2id[tri['tail']]
                r = rel2id[tri['relation']]
                neg_t = random.randint(0, len(entities) - 1)
                while neg_t == t and len(entities) > 1:
                    neg_t = random.randint(0, len(entities) - 1)

                h_vec = e_emb[h]
                t_vec = e_emb[t]
                r_vec = r_emb[r]
                neg_vec = e_emb[neg_t]

                score_pos = np.sum(np.abs(h_vec + r_vec - t_vec))
                score_neg = np.sum(np.abs(h_vec + r_vec - neg_vec))
                weight = 1.0
                if use_state:
                    weight *= state_weight(tri['state'])
                if use_time:
                    weight *= time_weight(tri['year'])
                loss = margin + score_pos - score_neg
                if loss > 0:
                    grad_pos = np.sign(h_vec + r_vec - t_vec)
                    grad_neg = np.sign(h_vec + r_vec - neg_vec)
                    e_emb[h] -= lr * weight * grad_pos
                    r_emb[r] -= lr * weight * grad_pos
                    e_emb[t] += lr * weight * grad_pos
                    e_emb[h] += lr * weight * grad_neg
                    r_emb[r] += lr * weight * grad_neg
                    e_emb[neg_t] -= lr * weight * grad_neg
            norms = np.linalg.norm(e_emb, axis=1, keepdims=True) + 1e-8
            e_emb = e_emb / norms
        space_models[space_id] = {
            'ent2id': ent2id,
            'rel2id': rel2id,
            'embeddings': e_emb,
            'rel_embeddings': r_emb,
            'triples': items
        }
    return space_models


def build_space_weights(space_models):
    weights = {}
    for space_id, model in space_models.items():
        stat = defaultdict(list)
        for tri in model['triples']:
            stat[(tri['head'], tri['relation'])].append(state_weight(tri['state']))
        weights[space_id] = {k: float(sum(v) / len(v)) for k, v in stat.items()}
    return weights


def compute_score(model, h, r, t):
    h_id = model['ent2id'].get(h)
    t_id = model['ent2id'].get(t)
    r_id = model['rel2id'].get(r)
    if h_id is None or t_id is None or r_id is None:
        return None
    return float(np.sum(np.abs(model['embeddings'][h_id] + model['rel_embeddings'][r_id] - model['embeddings'][t_id])))


def evaluate(triples, space_models, space_weights):
    hits1 = 0
    hits3 = 0
    hits10 = 0
    mrr = 0.0
    total = 0
    for tri in triples:
        h = tri['head']
        r = tri['relation']
        t = tri['tail']
        candidate_spaces = []
        for space_id, weights in space_weights.items():
            if (h, r) in weights:
                candidate_spaces.append(space_id)
        if not candidate_spaces:
            candidate_spaces = [0] if 0 in space_models else list(space_models.keys())
        candidate_entities = set()
        for space_id in candidate_spaces:
            candidate_entities.update(space_models[space_id]['ent2id'].keys())
        scores = []
        for tail in candidate_entities:
            total_score = 0.0
            total_weight = 0.0
            for space_id in candidate_spaces:
                model = space_models[space_id]
                score = compute_score(model, h, r, tail)
                if score is None:
                    continue
                weight = space_weights[space_id].get((h, r), 1.0)
                total_score += weight * score
                total_weight += weight
            if total_weight == 0:
                continue
            scores.append((tail, total_score / total_weight))
        if not scores:
            continue
        scores.sort(key=lambda x: x[1])
        rank = 1
        for candidate, _ in scores:
            if candidate == t:
                break
            rank += 1
        total += 1
        mrr += 1.0 / rank
        if rank <= 1:
            hits1 += 1
        if rank <= 3:
            hits3 += 1
        if rank <= 10:
            hits10 += 1
    if total == 0:
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'Hits@10': 0.0}
    return {
        'MRR': mrr / total,
        'Hits@1': hits1 / total,
        'Hits@3': hits3 / total,
        'Hits@10': hits10 / total
    }


def build_global_embeddings(space_models):
    embeddings = defaultdict(list)
    types = {}
    for model in space_models.values():
        for ent, idx in model['ent2id'].items():
            embeddings[ent].append(model['embeddings'][idx])
            types[ent] = get_label(ent)
    global_emb = {ent: np.mean(vecs, axis=0).tolist() for ent, vecs in embeddings.items()}
    return global_emb, types


def pca_2d(embeddings):
    keys = list(embeddings.keys())
    if not keys:
        return {}
    matrix = np.array([embeddings[k] for k in keys], dtype=np.float32)
    matrix -= matrix.mean(axis=0, keepdims=True)

    # 防止因样本数或特征维度过少导致的SVD错误
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        # 如果实体或特征太少，随机赋予坐标
        return {k: [random.uniform(-1, 1), random.uniform(-1, 1)] for k in keys}

    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    coords = np.dot(matrix, v[:2].T)
    return {k: coords[i].tolist() for i, k in enumerate(keys)}


def write_triplet_embedding_html(entity_emb_2d, relation_emb, entity_types, triples, output_path):
    # 将实体和三元组关系一起在可视化中体现
    nodes = [
        {
            'id': ent,
            'x': coord[0],
            'y': coord[1],
            'type': entity_types.get(ent, 'Entity')
        }
        for ent, coord in entity_emb_2d.items()
    ]

    # 构建边（即三元组关系）
    links = []
    for tri in triples:
        head = tri['head']
        tail = tri['tail']
        rel = tri['relation']
        if head in entity_emb_2d and tail in entity_emb_2d:
            links.append({
                'source': head,
                'target': tail,
                'relation': rel,
                'state': tri.get('state', 'reported')
            })

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>阶段三：三元组图谱嵌入可视化 (终极快照)</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Microsoft YaHei",Arial,Helvetica,sans-serif;background:#f3f6fb;color:#1f2937}}
#info{{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:16px;
  padding:14px 18px;
  background:linear-gradient(135deg,#ffffff,#eef4ff);
  border-bottom:1px solid #d8e1f0;
  box-shadow:0 2px 10px rgba(15,23,42,0.06);
}}
.panel-title{{font-size:18px;font-weight:700;color:#0f172a;margin-bottom:4px}}
.panel-subtitle{{font-size:12px;color:#64748b}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;justify-content:flex-end}}
.stat-card{{
  min-width:140px;
  padding:10px 12px;
  background:#ffffff;
  border:1px solid #dbe4f3;
  border-radius:12px;
  box-shadow:0 6px 18px rgba(148,163,184,0.12);
}}
.stat-label{{font-size:12px;color:#64748b;margin-bottom:4px}}
.stat-value{{font-size:20px;font-weight:700;color:#0f172a}}
#chart{{width:100vw;height:calc(100vh - 88px);display:block;cursor:grab;background:radial-gradient(circle at top,#f8fbff 0%,#edf3fb 55%,#e6eef8 100%)}}
#chart:active{{cursor:grabbing}}
.link {{ stroke: #94a3b8; stroke-opacity: 0.18; fill: none; }}
.link-trend {{ stroke: #ef4444; stroke-opacity: 0.32; stroke-width: 2px; }}
.node circle {{ stroke: #fff; stroke-width: 1.8px; transition: r 0.15s ease, opacity 0.15s ease; }}
.node text {{ pointer-events: none; font-size: 10px; fill:#334155; paint-order:stroke; stroke:#fff; stroke-width:3px; stroke-linejoin:round; }}
.node.faded circle, .link.faded, .link-trend.faded {{ opacity: 0.08; }}
.node.active circle {{ stroke:#0f172a; stroke-width:2.1px; }}
.link-label {{ pointer-events:none; font-size:9px; fill:#64748b; paint-order:stroke; stroke:#fff; stroke-width:3px; opacity:0; }}
.legend-note{{position:fixed;right:16px;bottom:16px;padding:10px 12px;background:rgba(255,255,255,0.92);border:1px solid #dbe4f3;border-radius:10px;color:#475569;font-size:12px;box-shadow:0 6px 16px rgba(15,23,42,0.08)}}
.hover-panel{{position:fixed;left:16px;bottom:16px;max-width:340px;padding:12px 14px;background:rgba(255,255,255,0.94);border:1px solid #dbe4f3;border-radius:12px;color:#334155;font-size:12px;line-height:1.5;box-shadow:0 10px 20px rgba(15,23,42,0.08)}}
.hover-title{{font-size:13px;font-weight:700;color:#0f172a;margin-bottom:4px}}
</style>
</head>
<body>
<div id="info">
  <div>
    <div class="panel-title">阶段三嵌入可视化</div>
    <div class="panel-subtitle">PCA 初始分布 + 自动避让布局，滚轮缩放，按住空白处拖动画布，拖动节点可局部整理</div>
  </div>
  <div class="stats">
    <div class="stat-card">
      <div class="stat-label">实体数</div>
      <div class="stat-value">{len(nodes)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">三元组关系数</div>
      <div class="stat-value">{len(links)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">趋势关系</div>
      <div class="stat-value">{sum(1 for link in links if link.get('state') == 'trend_confirmed'or'validated')}</div>
    </div>
  </div>
</div>
<svg id="chart"></svg>
<div class="legend-note">红色加粗连线代表处于 trend_confirmed 状态的知识三元组</div>
<div class="hover-panel" id="hoverPanel">
  <div class="hover-title">查看提示</div>
  <div>当前图不是固定死图。节点会基于嵌入位置自动避让，悬停节点可高亮邻接关系，放大后会显示更多标签。</div>
</div>
<script>
const nodesData = {json.dumps(nodes, ensure_ascii=False)};
const linksData = {json.dumps(links, ensure_ascii=False)};
const width = window.innerWidth;
const height = window.innerHeight - 88;
const hoverPanel = document.getElementById("hoverPanel");

const svg = d3.select("#chart").attr("width", width).attr("height", height);
svg.append("rect")
  .attr("width", width)
  .attr("height", height)
  .attr("fill", "transparent");

const viewport = svg.append("g").attr("class", "viewport");
const zoomLayer = viewport.append("g").attr("class", "zoom-layer");

const zoom = d3.zoom()
  .scaleExtent([0.25, 8])
  .on("zoom", (event) => {{
    zoomLayer.attr("transform", event.transform);
    updateLabelVisibility(event.transform.k);
  }});

svg.call(zoom).on("dblclick.zoom", null);

// 为连线添加箭头标记
svg.append("defs").append("marker")
    .attr("id", "arrow")
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 15)
    .attr("refY", 0)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
  .append("path")
    .attr("fill", "#999")
    .attr("d", "M0,-5L10,0L0,5");

const color = d3.scaleOrdinal()
  .domain(["Alloy","Element","Property","Process","Value","Entity"])
  .range(["#4dabf7","#69db7c","#ffd43b","#845ef7","#ffa94d","#ced4da"]);

const xExtent = d3.extent(nodesData, d => d.x);
const yExtent = d3.extent(nodesData, d => d.y);
const x = d3.scaleLinear().domain(xExtent).range([60, width - 60]);
const y = d3.scaleLinear().domain(yExtent).range([height - 60, 60]);
const degreeMap = new Map();
linksData.forEach(link => {{
  degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
  degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
}});

nodesData.forEach(node => {{
  node.initialX = x(node.x);
  node.initialY = y(node.y);
  node.x = node.initialX;
  node.y = node.initialY;
  node.degree = degreeMap.get(node.id) || 0;
  node.radius = Math.max(3.5, Math.min(10, 3.8 + Math.sqrt(node.degree) * 0.9));
  node.alwaysShowLabel = node.degree >= 8 || node.type === "Alloy" || node.type === "Property";
}});

const linkedById = new Set(
  linksData.map(d => `${{d.source}}->${{d.target}}`).concat(linksData.map(d => `${{d.target}}->${{d.source}}`))
);

function isNeighbor(a, b) {{
  return a.id === b.id || linkedById.has(`${{a.id}}->${{b.id}}`);
}}

const link = zoomLayer.append("g")
  .selectAll("line")
  .data(linksData)
  .join("line")
  .attr("class", d => d.state === "trend_confirmed" ? "link-trend" : "link")
  .attr("marker-end", "url(#arrow)");

const linkText = zoomLayer.append("g")
  .selectAll("text")
  .data(linksData)
  .join("text")
  .attr("class", "link-label")
  .text(d => d.relation);

const node = zoomLayer.append("g")
  .selectAll("g")
  .data(nodesData)
  .join("g")
  .attr("class", "node")
  .call(
    d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended)
  )
  .on("mouseenter", handleMouseEnter)
  .on("mouseleave", handleMouseLeave);

node.append("circle")
  .attr("r", d => d.radius)
  .attr("fill", d => color(d.type));

node.append("title")
  .text(d => `${{d.id}} (${{d.type}})`);

node.append("text")
  .attr("dy", d => -d.radius - 4)
  .attr("text-anchor", "middle")
  .text(d => d.id);

const simulation = d3.forceSimulation(nodesData)
  .force("link", d3.forceLink(linksData).id(d => d.id).distance(38).strength(0.08))
  .force("charge", d3.forceManyBody().strength(-36))
  .force("collide", d3.forceCollide().radius(d => d.radius + 5).iterations(2))
  .force("x", d3.forceX(d => d.initialX).strength(0.14))
  .force("y", d3.forceY(d => d.initialY).strength(0.14))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .alpha(0.9)
  .alphaDecay(0.04)
  .on("tick", ticked);

function ticked() {{
  link
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);

  linkText
    .attr("x", d => (d.source.x + d.target.x) / 2)
    .attr("y", d => (d.source.y + d.target.y) / 2);
}}

function updateLabelVisibility(scale = 1) {{
  node.select("text")
    .style("display", d => (scale >= 1.8 || (scale >= 1.1 && d.alwaysShowLabel)) ? "block" : "none");

  linkText
    .style("opacity", scale >= 2.2 ? 0.75 : 0);
}}

function handleMouseEnter(event, hovered) {{
  node
    .classed("faded", d => !isNeighbor(hovered, d))
    .classed("active", d => d.id === hovered.id);

  link.classed("faded", d => !(d.source.id === hovered.id || d.target.id === hovered.id));
  linkText.style("opacity", d => (d.source.id === hovered.id || d.target.id === hovered.id) ? 0.9 : 0);

  node.select("text")
    .style("display", d => isNeighbor(hovered, d) ? "block" : null);

  hoverPanel.innerHTML = `
    <div class="hover-title">${{hovered.id}}</div>
    <div>类型：${{hovered.type}}</div>
    <div>关联数量：${{hovered.degree}}</div>
    <div>提示：拖动这个节点可以手动整理局部拥挤区域。</div>
  `;
}}

function handleMouseLeave() {{
  node.classed("faded", false).classed("active", false);
  link.classed("faded", false);
  hoverPanel.innerHTML = `
    <div class="hover-title">查看提示</div>
    <div>当前图不是固定死图。节点会基于嵌入位置自动避让，悬停节点可高亮邻接关系，放大后会显示更多标签。</div>
  `;
  updateLabelVisibility(currentZoomScale());
}}

function currentZoomScale() {{
  return d3.zoomTransform(svg.node()).k;
}}

function dragstarted(event, d) {{
  if (!event.active) simulation.alphaTarget(0.18).restart();
  d.fx = d.x;
  d.fy = d.y;
}}

function dragged(event, d) {{
  d.fx = event.x;
  d.fy = event.y;
}}

function dragended(event, d) {{
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}}

const xSpan = Math.max((xExtent[1] ?? 1) - (xExtent[0] ?? 0), 1e-6);
const ySpan = Math.max((yExtent[1] ?? 1) - (yExtent[0] ?? 0), 1e-6);
const fitScale = Math.min((width - 140) / (xSpan * 120), (height - 120) / (ySpan * 120), 2.2);
svg.call(
  zoom.transform,
  d3.zoomIdentity.translate(width * 0.12, height * 0.1).scale(Math.max(0.75, fitScale))
);
updateLabelVisibility(Math.max(0.75, fitScale));

</script>
</body>
</html>"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def stage3_pipeline(packages, output_embeddings, output_metrics, output_html, dim=64, epochs=20, lr=0.01):
    samples = build_training_samples(packages)
    random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    test_samples = samples[split:] if split < len(samples) else samples

    spaces_single = {tri['event_id']: 0 for tri in samples}
    spaces_multi = build_spaces(samples)

    models_transe = train_transe(train_samples, spaces_single, dim=dim, lr=lr, epochs=epochs, use_state=False,
                                 use_time=False)
    models_putranse = train_transe(train_samples, spaces_multi, dim=dim, lr=lr, epochs=epochs, use_state=False,
                                   use_time=False)
    models_sa_putranse = train_transe(train_samples, spaces_multi, dim=dim, lr=lr, epochs=epochs, use_state=True,
                                      use_time=True)
    models_wo_state = train_transe(train_samples, spaces_multi, dim=dim, lr=lr, epochs=epochs, use_state=False,
                                   use_time=True)
    models_wo_time = train_transe(train_samples, spaces_multi, dim=dim, lr=lr, epochs=epochs, use_state=True,
                                  use_time=False)

    metrics = {
        'TransE': evaluate(test_samples, models_transe, build_space_weights(models_transe)),
        'puTransE': evaluate(test_samples, models_putranse, build_space_weights(models_putranse)),
        'Sa-puTransE': evaluate(test_samples, models_sa_putranse, build_space_weights(models_sa_putranse)),
        'Ours_w_o_state': evaluate(test_samples, models_wo_state, build_space_weights(models_wo_state)),
        'Ours_w_o_time': evaluate(test_samples, models_wo_time, build_space_weights(models_wo_time))
    }

    global_emb, types = build_global_embeddings(models_sa_putranse)

    # 提取全局关系嵌入 (求各空间的平均值)
    rel_embeddings = defaultdict(list)
    for model in models_sa_putranse.values():
        for rel, idx in model['rel2id'].items():
            rel_embeddings[rel].append(model['rel_embeddings'][idx])
    global_rel_emb = {rel: np.mean(vecs, axis=0).tolist() for rel, vecs in rel_embeddings.items()}

    embedding_2d = pca_2d(global_emb)

    with open(output_embeddings, 'w', encoding='utf-8') as f:
        json.dump({'embeddings': global_emb, 'rel_embeddings': global_rel_emb, 'types': types}, f, ensure_ascii=False,
                  indent=2)
    with open(output_metrics, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if output_html is not None:
        write_triplet_embedding_html(embedding_2d, global_rel_emb, types, samples, output_html)
    return metrics


def stage4_pipeline(packages, embeddings_path, output_path, target_property='抗拉强度', threshold=150, top_k=10):
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        emb_data = json.load(f)
    embeddings = emb_data.get('embeddings', {})
    types = emb_data.get('types', {})

    samples = build_training_samples(packages)
    candidates = []
    for tri in samples:
        if str(tri['head']) != target_property:
            continue
        value = parse_numeric(tri['tail'])
        if value is None:
            continue
        if value >= threshold and tri['state'] == 'trend_confirmed':
            candidates.append(tri)

    # 按照需求文档增加 Cypher 查询回溯生成的逻辑
    cypher_query = f"""
    // 阶段四：寻找满足性能阈值且 knowledge_state 至少为 trend_confirmed 的 Event 节点，回溯其完整的工艺-成分链
    MATCH (e:Event)-[:INSTANCE_OF]->(p:Property {{name: '{target_property}'}})
    WHERE e.knowledge_state IN ['trend_confirmed', 'validated']
      AND toFloat(e.quantitative_value) >= {threshold}
    // 回溯其上下文合金主体
    MATCH (e)-[:CONTEXT_FOR {{role: 'subject'}}]->(alloy:Alloy)
    // 回溯其工艺条件
    OPTIONAL MATCH (e)-[:UNDER_CONDITION]->(proc:Process)
    // 回溯合金成分
    OPTIONAL MATCH (alloy)-[comp:HAS_CONCENTRATION]->(element:Element)
    RETURN alloy.name AS Alloy, 
           collect(DISTINCT proc.name) AS Processes,
           collect(DISTINCT {{element: element.name, value: comp.source_event}}) AS Compositions,
           e.quantitative_value AS Value, 
           e.knowledge_state AS State
    ORDER BY toFloat(e.quantitative_value) DESC
    LIMIT {top_k};
    """

    target_vec = None
    if target_property in embeddings:
        target_vec = np.array(embeddings[target_property], dtype=np.float32)
    else:
        prop_vecs = [np.array(v, dtype=np.float32) for k, v in embeddings.items() if types.get(k) == 'Property']
        if prop_vecs:
            target_vec = np.mean(prop_vecs, axis=0)
    recommendations = []
    if target_vec is not None:
        for ent, vec in embeddings.items():
            if types.get(ent) != 'Alloy':
                continue
            vec_np = np.array(vec, dtype=np.float32)
            sim = float(np.dot(target_vec, vec_np) / (np.linalg.norm(target_vec) * np.linalg.norm(vec_np) + 1e-8))
            recommendations.append({'alloy': ent, 'similarity': sim})
        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        recommendations = recommendations[:top_k]

    output = {
        'target_property': target_property,
        'threshold': threshold,
        'trend_confirmed_events': candidates,
        'recommendations': recommendations,
        'cypher_query_for_retrieval': cypher_query.strip()
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def main():
    parser = argparse.ArgumentParser(description='阶段二到阶段四流程')
    parser.add_argument('--stage', type=str, default='stage2', choices=['stage2', 'stage3', 'stage4', 'all'])
    parser.add_argument('--input', type=str, default='../stage1_data_packages.json')
    parser.add_argument('--output-cypher', type=str, default='stage2_import.cypher')
    parser.add_argument('--output-stats', type=str, default='stage2_stats.json')
    parser.add_argument('--output-html', type=str, default='stage2_graph.html')
    parser.add_argument('--stage3-embeddings', type=str, default='stage3_embeddings.json')
    parser.add_argument('--stage3-metrics', type=str, default='stage3_metrics.json')
    parser.add_argument('--stage3-html', type=str, default='stage3_embedding.html')
    parser.add_argument('--stage3-dim', type=int, default=64)
    parser.add_argument('--stage3-epochs', type=int, default=20)
    parser.add_argument('--stage3-lr', type=float, default=0.01)
    parser.add_argument('--stage4-output', type=str, default='stage4_recommendations.json')
    parser.add_argument('--stage4-property', type=str, default='UTS')
    parser.add_argument('--stage4-threshold', type=float, default=300)
    parser.add_argument('--stage4-topk', type=int, default=10)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j-user', type=str, default='neo4j')
    parser.add_argument('--neo4j-password', type=str, default='neo4j')
    args = parser.parse_args()

    packages = load_packages(args.input)

    if args.stage in ['stage2', 'all']:
        cypher_statements, anomalies, graph_data = build_cypher(packages)
        write_cypher(cypher_statements, args.output_cypher)
        stats = {
            'packages': len(packages),
            'events': sum(len(p.get('triplets', [])) for p in packages),
            'anomalous_events': len(anomalies),
            'anomaly_ratio': (len(anomalies) / max(sum(len(p.get('triplets', [])) for p in packages), 1)),
            'generated_at': datetime.now().isoformat()
        }
        write_stats(stats, args.output_stats)
        write_html(graph_data, stats, args.output_html)
        if args.execute:
            execute_cypher(cypher_statements, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        print(f"导入脚本已生成: {args.output_cypher}")
        print(f"统计信息已生成: {args.output_stats}")
        print(f"可视化文件已生成: {args.output_html}")
        print(f"异常事件数量: {stats['anomalous_events']}，比例: {stats['anomaly_ratio']:.4f}")

    if args.stage in ['stage3', 'all']:
        metrics = stage3_pipeline(
            packages,
            output_embeddings=args.stage3_embeddings,
            output_metrics=args.stage3_metrics,
            output_html=args.stage3_html,
            dim=args.stage3_dim,
            epochs=args.stage3_epochs,
            lr=args.stage3_lr
        )
        print(f"阶段三指标已生成: {args.stage3_metrics}")
        print(f"阶段三嵌入已生成: {args.stage3_embeddings}")
        print(f"阶段三可视化已生成: {args.stage3_html}")
        print(metrics)

    if args.stage in ['stage4', 'all']:
        output = stage4_pipeline(
            packages,
            embeddings_path=args.stage3_embeddings,
            output_path=args.stage4_output,
            target_property=args.stage4_property,
            threshold=args.stage4_threshold,
            top_k=args.stage4_topk
        )
        print(f"阶段四结果已生成: {args.stage4_output}")
        print(f"推荐数量: {len(output.get('recommendations', []))}")


if __name__ == '__main__':
    main()
