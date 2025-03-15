import torch
import torch.nn as nn
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 加载 tokenizer 和 model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# 模拟嵌入模型，确保类型为 float32
embed_model = lambda x: torch.tensor(np.random.rand(512), dtype=torch.float32)

# 嵌入转换函数
def emb_to_seq(e, W1, W2, s=3):
    e = e.to(W1.dtype).to(W1.device)  # 确保类型和设备一致
    hidden = torch.matmul(e, W1)  # 线性变换
    hidden = torch.relu(hidden)   # 非线性激活
    seq = torch.matmul(hidden, W2)  # 投影回嵌入空间
    return seq.repeat(s, 1)  # 扩展到长度 s

# 输入序列拼接
def prepare_input(e, e_hat, x_tokens, W1, W2, embed_layer):
    # 将 x_tokens 转换为嵌入表示
    x_embeds = embed_layer(x_tokens)  # 形状 (batch_size, seq_len, embed_dim)
    e_seq = emb_to_seq(e, W1, W2)
    # 确保 x_embeds 的维度与 e_seq 匹配，调整为 (s, embed_dim)
    x_embeds = x_embeds.squeeze(0).transpose(0, 1)[:3, :]  # 取前 s=3 个 token 的嵌入并调整大小
    x_embeds = x_embeds.repeat(1, 1)  # 保持原始长度
    e_seq = emb_to_seq(e, W1, W2)
    e_hat_seq = emb_to_seq(e_hat, W1, W2)
    diff_seq = emb_to_seq(e - e_hat, W1, W2)
    input_seq = torch.cat([e_seq, e_hat_seq, diff_seq, x_embeds], dim=0)
    return input_seq

# 迭代优化
def vec2text(e, max_steps=10, beam_size=5):
    # 初始假设 x^{(0)}
    x = "initial guess"  # 初始文本
    x_tokens = tokenizer(x, return_tensors="pt")["input_ids"]
    
    W1 = torch.randn(512, 256, dtype=torch.float32)  # 确保类型为 float32
    W2 = torch.randn(256, 512, dtype=torch.float32)
    embed_layer = model.get_input_embeddings()  # 获取 T5 的嵌入层
    
    for t in range(max_steps):
        # 计算当前嵌入
        e_hat = embed_model(x)
        
        # 准备输入
        input_seq = prepare_input(e, e_hat, x_tokens, W1, W2, embed_layer)
        
        # 使用 T5 生成新文本
        outputs = model.generate(input_seq.unsqueeze(0), num_beams=beam_size, max_length=50)  # 添加 batch 维度
        x = tokenizer.decode(outputs[0], skip_special_tokens=True)
        x_tokens = tokenizer(x, return_tensors="pt")["input_ids"]
        
        # 检查嵌入距离
        dist = torch.norm(e - e_hat)
        print(f"Step {t}, Text: {x}, Distance: {dist.item()}")
        if dist < 0.01:  # 停止条件
            break
    
    return x

# 测试
target_text = "Hello world"
e = embed_model(target_text)
generated_text = vec2text(e)
print("Final generated text:", generated_text)