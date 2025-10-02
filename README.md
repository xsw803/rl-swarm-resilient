# RL Swarm Resilient

增强版RL Swarm，支持断线自动重连、代理设置和本地模型加载，解决网络连接不稳定环境下的运行问题。

## 主要特性

- **断线自动重连**：使用指数退避策略自动重试网络操作
- **本地模型支持**：可从本地路径加载模型，无需联网
- **代理支持**：自动检测并应用系统代理配置
- **详细日志**：完整的操作日志和错误提示
- **友好错误处理**：提供解决方案建议

## 使用方法

### 设置环境变量

```bash
# 设置本地模型路径（可选）
export LOCAL_MODEL_PATH=/path/to/local/model

# 设置代理（可选）
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# 设置模型超时时间
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

### 运行程序

```bash
python swarm_launcher.py
```

## 常见问题

### 无法连接Hugging Face
- 设置代理环境变量
- 预先下载模型到本地并设置LOCAL_MODEL_PATH

### P2P连接失败
- 检查网络连接
- 确保防火墙未阻止P2P通信
- 验证代理设置是否正确