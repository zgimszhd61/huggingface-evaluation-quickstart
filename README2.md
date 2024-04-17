# 首先，安装必要的库
!pip install transformers
!pip install evaluate
!pip install git+https://github.com/openai/human-eval.git

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import human_eval

# 加载模型和分词器
model_name = "mistral-community/Mixtral-8x22B-v0.1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 下载 HumanEval 数据集
tasks = human_eval.data.get_tasks()

# 定义生成代码的函数
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_length = model.config.n_positions
    outputs = model.generate(**inputs, max_length=max_length)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code

# 定义评估函数
def evaluate_model(tasks, gen_func):
    results = []
    for task in tasks:
        prompt = task['prompt']
        canonical_solution = task['canonical_solution']
        completion = gen_func(prompt)
        results.append((prompt, canonical_solution, completion))
    return results

# 进行评估
results = evaluate_model(tasks, generate_code)

# 打印结果
for prompt, canonical, completion in results[:5]:  # 只打印前5个结果
    print(f"Prompt:\n{prompt}\n")
    print(f"Canonical Solution:\n{canonical}\n")
    print(f"Generated Completion:\n{completion}\n")
    print("="*50)
