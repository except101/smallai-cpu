"""Small, original fixtures. These are not a general Chinese training corpus."""

import json
from pathlib import Path

from .format import tool_call_message
from .tools import TOOL_SCHEMAS, calculate, compact_json


PARAGRAPHS = [
    "地球围绕太阳运行，月球围绕地球运行。地球自转使不同地区交替面向太阳，形成昼夜。描述一个天体的运动时，需要说明观察的参照物。",
    "植物的叶片通常含有叶绿素。光合作用利用光能，把二氧化碳和水转化为有机物，并释放氧气。植物生长还需要适宜的温度、水分和矿物质。",
    "在通常的大气压力下，水的冰点约为零摄氏度，沸点约为一百摄氏度。压力变化会影响沸点，因此高山上的水可能在较低温度时沸腾。",
    "一天有二十四小时，一小时有六十分钟，一分钟有六十秒。时间差的计算需要先统一单位。跨时区比较两个时间，还应当考虑各自的时区。",
    "长度可以用米、厘米和毫米表示。一米等于一百厘米，一厘米等于十毫米。面积和体积的单位换算需要分别考虑平方和立方关系。",
    "阅读一篇文章时，可以先找出它讨论的问题，再观察各段提供了哪些事实和理由。概括文章时应保留主要意思，避免把自己的猜测当成原文的观点。",
    "学习一项新技能可以分成理解、练习和反馈几个步骤。把较大的任务分成可以检查的小任务，能够帮助我们及时发现错误，并调整下一步的练习。",
    "一张表格由行和列构成。表头说明每列数据的含义。记录数据时应当保持单位一致，缺少的信息可以明确标记，不能随意编造数值补齐。",
    "计算平均数时，先求出所有数的总和，再除以数的个数。平均数可能受到极端值影响，因此描述一组数据时，有时还需要观察中位数和分布。",
    "分数表示整体的一部分。分母表示把整体平均分成多少份，分子表示其中的几份。不同分母的分数相加，可以先通分，再进行计算。",
    "字符串是一串字符，整数表示没有小数部分的数。程序处理数据时应当区分不同类型，例如文本中的数字字符和可以参加算术运算的数字。",
    "函数接收参数并返回结果。函数名说明它的用途，参数说明它处理的输入。调用函数前需要检查参数的类型，并考虑输入缺失或不合法的情况。",
    "JSON 是一种常见的数据交换格式。对象使用花括号，数组使用方括号，字符串使用双引号。对象中的键和值通过冒号连接，各项之间使用逗号分隔。",
    "工具调用分为选择工具、填写参数、执行工具和解释结果几个步骤。助手需要等待工具返回真实结果，再根据结果回答。工具失败时，应明确说明失败原因。",
    "当一个问题缺少必要条件时，直接猜测可能给出错误答案。助手可以先说明缺少哪项信息，再提出一个简短的问题，例如询问需要查询哪个时区的时间。",
    "语言模型把文本转换成 token，并学习根据前面的 token 预测后续 token。训练 loss 下降说明预测训练目标有所改善，但还需要独立测试实际任务表现。",
    "训练集用于调整模型参数，验证集用于观察模型对未参与训练的数据的表现。应在切分文本之前处理完全重复的样本，避免同一篇文本同时进入两边。",
    "随机初始化表示模型参数从随机数开始，没有加载已经训练好的语言模型权重。这样的模型需要先学习语言规律，再学习遵循指令和使用工具。",
    "分词器把文字编码成整数序列，也能把整数序列解码成文字。训练后的模型与它的分词器必须保持对应，改变词表会改变相同整数所代表的含义。",
    "模型参数量与训练数据量是不同的概念。参数描述模型的内部规模，数据提供学习材料。体积较小的模型仍然可能需要长期训练才能获得较好的通用能力。",
    "计算机的内存保存程序运行时需要的数据，硬盘用于较长期地保存文件。训练模型时，参数、梯度和优化器状态都会占用内存，不能只计算模型文件大小。",
    "检查程序时，可以先验证一个小例子，再测试边界条件。例如计算器除了验证加法，还应检查除以零、无效字符和不支持的运算是否被正确处理。",
    "备份文件时，应记录文件的来源、日期和用途。训练 checkpoint 可以保存模型参数、优化器状态和训练进度，便于在中断后继续同一项训练任务。",
    "一个合理的计划应说明目标、资源和检查方式。遇到资源限制时，可以先完成小规模实验，测量实际速度和效果，再判断是否有必要扩大任务。",
    "声音通常通过介质中的振动传播。不同介质中的传播速度可能不同。讨论声音的频率、响度和音色时，需要区分这些概念各自描述的性质。",
    "太阳能、风能和水能属于常见的可再生能源形式。不同能源的利用方式受到地理、天气和基础设施影响，比较方案时需要明确使用场景和评价指标。",
    "地图用符号和比例尺表示地理信息。读取地图时，可以先找到图例和方向，再根据比例尺估算实际距离。不同投影方式会在距离、面积或形状上产生差异。",
    "中文标点帮助表达句子之间的关系。句号通常用于结束陈述，问号通常用于提出问题。写作时应使句子结构清楚，避免让同一个代词指向多个不同对象。",
    "一个清楚的回答应先回应问题，再提供必要解释。当事实不确定时，可以说明不确定的部分，并区分已经知道的信息与需要进一步查询的信息。",
    "观察实验结果时，需要记录实验条件。改变多个条件后，即使结果发生变化，也不容易判断是哪一个条件造成的。对照实验可以帮助我们分析这种关系。",
    "文件扩展名通常提示文件的格式，但不保证内容正确。读取 JSONL 文件时，应逐行检查每条记录；发现错误时，指出文件名和行号能够帮助定位问题。",
    "北京时间采用 UTC 加八小时。比较北京时间与协调世界时，可以先换算成相同的时区。查询当前时间需要读取时钟，训练语料中的旧时间不能代替实时查询。",
]

GENERAL_PAIRS = [
    ("你好", "你好！有什么我可以帮你的吗？"),
    ("你是谁？", "我是一个正在学习中文对话和工具调用的小型实验助手。"),
    ("谢谢", "不客气。"),
    ("再见", "再见！"),
    ("什么是函数？", "函数接收输入参数，完成一项任务，并返回结果。"),
    ("什么是 JSON？", "JSON 是一种数据交换格式，常用对象、数组、字符串和数字表示信息。"),
    ("怎样开始学编程？", "先选择一种语言，学习变量、条件和循环，再通过小程序练习。"),
    ("怎样安排学习时间？", "先列出任务，按重要程度安排时间，并留出休息和复习的空隙。"),
    ("帮我写一句鼓励的话。", "从今天能完成的一小步开始，练习会逐渐带来进步。"),
    ("把“我正在学习”翻译成英文。", "I am learning."),
    ("请用一句话解释分词。", "分词是把文本转换成模型可以处理的 token 序列。"),
    ("请简单解释训练集和验证集。", "训练集用于学习参数，验证集用于检查模型对未参与训练的数据的表现。"),
    ("查询时间需要什么信息？", "需要确定时区，例如北京时间或 UTC。"),
    ("帮我算一下。", "请提供需要计算的算式。"),
    ("现在几点？", "你想查询北京时间，还是 UTC 时间？"),
    ("计算器工具有什么作用？", "它接收算式并返回计算结果。"),
    ("我今天有点累。", "可以先休息一会儿，喝点水，再决定要不要继续手头的事。"),
    ("给我一个阅读建议。", "先选一篇感兴趣的短文，读完后用自己的话概括主要内容。"),
    ("请介绍地球和月球的关系。", "月球是地球的天然卫星，围绕地球运行。"),
    ("怎样让回答更清楚？", "先说主要结论，再用简短的解释或例子补充。"),
]


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_demo(output):
    output = Path(output)
    names = ("corpus.jsonl", "sft.jsonl", "eval_tools.jsonl")
    if any((output / name).exists() for name in names):
        raise ValueError(f"示例文件已存在：{output}。不会覆盖你的语料")
    output.mkdir(parents=True, exist_ok=True)
    corpus = [{"text": paragraph} for paragraph in PARAGRAPHS]
    for number in range(1, 101):
        a, b = number + 2, number % 9 + 1
        corpus.append({"text": f"算术练习第{number}题：{a} 加 {b} 等于 {a + b}，{a} 乘 {b} 等于 {a * b}。"
                       "计算前先读清楚问题，确定运算符和数字。需要多步计算时，可以先计算括号中的部分，再按照运算顺序得到结果。"})
    corpus.append({"text": compact_json(TOOL_SCHEMAS)})
    conversations = [{"tools": TOOL_SCHEMAS, "messages": [
        {"role": "user", "content": question}, {"role": "assistant", "content": answer}]}
        for question, answer in GENERAL_PAIRS]
    for number in range(1, 101):
        a, b = number % 31 + 1, number % 13 + 1
        operation = ("+", "-", "*")[number % 3]
        expression = f"{a} {operation} {b}"
        value = calculate(expression)
        call = {"name": "calculator", "arguments": {"expression": expression}}
        conversations.append({"tools": TOOL_SCHEMAS, "messages": [
            {"role": "user", "content": f"请计算 {expression}。"}, tool_call_message(call),
            {"role": "tool", "name": "calculator", "tool_call_id": "call_0", "content": compact_json({"result": value})},
            {"role": "assistant", "content": f"{expression} = {value}。"}]})
    for zone, prompts, timestamp in [
        ("Asia/Shanghai", ["请查询当前北京时间。", "北京现在是什么时间？", "告诉我此刻的北京时间。", "帮我看一下北京时间。"], "2026-01-01T12:00:00+08:00"),
        ("UTC", ["请查询当前 UTC 时间。", "UTC 现在几点？", "告诉我此刻的 UTC 时间。", "帮我看一下 UTC 时间。"], "2026-01-01T04:00:00+00:00"),
    ]:
        for prompt in prompts:
            call = {"name": "get_current_time", "arguments": {"timezone": zone}}
            conversations.append({"tools": TOOL_SCHEMAS, "messages": [
                {"role": "user", "content": prompt}, tool_call_message(call),
                {"role": "tool", "name": call["name"], "tool_call_id": "call_0",
                 "content": compact_json({"timezone": zone, "time": timestamp})},
                {"role": "assistant", "content": f"当前 {zone} 时间是 {timestamp}。"}]})
    evaluation = []
    for index, expression in enumerate(["137 + 29", "203 - 78", "41 * 17", "84 / 7", "(23 + 19) * 4", "2 ** 5"]):
        evaluation.append({"id": f"calc-{index}", "tools": TOOL_SCHEMAS,
                           "messages": [{"role": "user", "content": f"请计算 {expression}。"}],
                           "expected_call": {"name": "calculator", "arguments": {"expression": expression}},
                           "expected_result": {"result": calculate(expression)}})
    for zone in ("UTC", "Asia/Shanghai"):
        evaluation.append({"id": f"time-{zone}", "tools": TOOL_SCHEMAS,
                           "messages": [{"role": "user", "content": f"查询此时 {zone} 的时间。"}],
                           "expected_call": {"name": "get_current_time", "arguments": {"timezone": zone}}})
    for index, prompt in enumerate(["早上好。", "解释一下什么是工具调用。", "给我一句简短的祝福。", "帮我查询时间，但我还没决定时区。"]):
        evaluation.append({"id": f"no-tool-{index}", "tools": TOOL_SCHEMAS,
                           "messages": [{"role": "user", "content": prompt}], "expected_call": None})
    write_jsonl(output / names[0], corpus)
    write_jsonl(output / names[1], conversations)
    write_jsonl(output / names[2], evaluation)
    (output / "NOTICE.txt").write_text(
        "这些是自写的流程测试样本，不是通用中文语料。时间返回值为虚构训练样例。\n"
        "训练集和测试集共享任务模板，测试只能用于开发检查，不能证明泛化或 MiniCPM 水平。\n",
        encoding="utf-8")
    return {"corpus_documents": len(corpus), "sft_conversations": len(conversations), "evaluation_cases": len(evaluation)}
