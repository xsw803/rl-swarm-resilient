import os
import time
import socket
import logging
import requests
from functools import wraps
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hivemind import HivemindBackend, HivemindRendezvous
import hydra
from omegaconf import DictConfig
import sys

# 添加代理支持
def setup_proxy():
    # 尝试从环境变量获取代理配置
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        
        # 为requests设置代理
        requests.Session().proxies.update(proxies)
        logging.info(f"设置代理: {proxies}")
        
        # 为Hugging Face设置代理
        os.environ['HTTP_PROXY'] = http_proxy or ''
        os.environ['HTTPS_PROXY'] = https_proxy or ''

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/swarm_launcher.log"),
        logging.StreamHandler()
    ]
)

def get_logger():
    return logging.getLogger(__name__)

# 重试配置
MAX_RETRIES = 10
RETRY_DELAY = 5  # 初始延迟时间（秒）
MAX_RETRY_DELAY = 60  # 最大延迟时间（秒）

def with_retry(func, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY, max_delay=MAX_RETRY_DELAY, error_types=None):
    """
    重试装饰器，用于包装可能因网络问题失败的函数
    
    Args:
        func: 要重试的函数
        max_retries: 最大重试次数
        base_delay: 初始延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        error_types: 要捕获并重试的错误类型列表
    
    Returns:
        原函数的结果
    """
    if error_types is None:
        error_types = (requests.exceptions.RequestException, ConnectionError, TimeoutError, socket.timeout)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        delay = base_delay
        last_exception = None
        
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except tuple(error_types) as e:
                last_exception = e
                retries += 1
                if retries > max_retries:
                    get_logger().error(f"达到最大重试次数 {max_retries}，操作失败: {str(e)}")
                    raise
                
                get_logger().warning(f"操作失败，{delay}秒后第 {retries}/{max_retries} 次重试: {str(e)}")
                time.sleep(delay)
                # 指数退避，但不超过最大延迟
                delay = min(delay * 2, max_delay)
        
        raise last_exception
    
    return wrapper

def check_local_model(model_path):
    """
    检查本地模型是否存在
    
    Args:
        model_path: 模型路径
    
    Returns:
        bool: 如果模型存在返回True，否则返回False
    """
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    return True

@with_retry
def load_model_with_retry(model_name, cache_dir=None, local_only=False, **kwargs):
    """
    尝试加载模型，支持从本地加载
    
    Args:
        model_name: 模型名称或本地路径
        cache_dir: 缓存目录
        local_only: 是否只从本地加载
        **kwargs: 其他参数
    
    Returns:
        模型和分词器
    """
    logger = get_logger()
    
    # 检查是否是本地路径
    if os.path.exists(model_name):
        logger.info(f"从本地路径加载模型: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            return model, tokenizer
        except Exception as e:
            logger.error(f"从本地路径加载模型失败: {e}")
            raise
    
    # 如果设置为只从本地加载且不是本地路径，则抛出异常
    if local_only:
        logger.error(f"本地模型路径不存在: {model_name}")
        raise ValueError(f"本地模型路径不存在: {model_name}")
    
    # 尝试从Hugging Face加载模型
    logger.info(f"尝试从Hugging Face加载模型: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, **kwargs)
        logger.info(f"成功加载模型: {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"从Hugging Face加载模型失败: {e}")
        # 提供更友好的错误提示
        logger.info("\n提示: 无法连接到Hugging Face下载模型。您可以：")
        logger.info("1. 设置代理环境变量: export HTTP_PROXY=http://proxy:port export HTTPS_PROXY=http://proxy:port")
        logger.info("2. 预先下载模型到本地，然后通过LOCAL_MODEL_PATH环境变量指定路径")
        raise

def init_hivemind(cfg, max_retries=5, retry_delay=10):
    """
    初始化Hivemind后端
    
    Args:
        cfg: 配置对象
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        HivemindBackend实例
    """
    logger = get_logger()
    retries = 0
    
    while retries < max_retries:
        try:
            logger.info("初始化Hivemind后端...")
            backend = HivemindBackend(
                dht_rendezvous=cfg.rendezvous.dht_rendezvous,
                host_maddrs=cfg.backend.host_maddrs,
                use_relay=cfg.backend.use_relay,
                initial_peers=cfg.backend.initial_peers,
                identity_path=os.environ.get("IDENTITY_PATH", None),
            )
            
            # 等待后端完全初始化
            logger.info("等待Hivemind后端初始化完成...")
            backend.wait_until_ready(timeout=30)
            logger.info("Hivemind后端初始化成功")
            return backend
            
        except Exception as e:
            retries += 1
            logger.error(f"Hivemind后端初始化失败 (尝试 {retries}/{max_retries}): {e}")
            
            if retries >= max_retries:
                logger.error("达到最大重试次数，Hivemind后端初始化失败")
                logger.info("\n提示: 无法连接到P2P引导节点。请检查：")
                logger.info("1. 网络连接是否正常")
                logger.info("2. 防火墙是否阻止了P2P连接")
                logger.info("3. 代理设置是否正确")
                raise
            
            logger.info(f"{retry_delay}秒后重试...")
            time.sleep(retry_delay)

@hydra.main(version_base=None, config_path="../../rgym_exp/config", config_name="rg-swarm.yaml")
def main(cfg: DictConfig):
    logger = get_logger()
    logger.info("开始启动RL Swarm...")
    
    # 设置代理
    setup_proxy()
    
    # 检查是否设置了本地模型路径
    local_model_path = os.environ.get("LOCAL_MODEL_PATH")
    use_local_model = local_model_path is not None
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        logger.info("未检测到GPU，使用CPU")
        device = "cpu"
    
    try:
        # 初始化Hivemind后端
        backend = init_hivemind(cfg)
        
        # 初始化Hivemind Rendezvous
        logger.info("初始化Hivemind Rendezvous...")
        HivemindRendezvous.init(
            dht_rendezvous=cfg.rendezvous.dht_rendezvous,
            backend=backend
        )
        logger.info("Hivemind Rendezvous初始化成功")
        
        # 加载模型
        model_name = os.environ.get("MODEL_NAME", cfg.model.name)
        
        # 如果指定了本地模型路径，优先使用
        if use_local_model:
            logger.info(f"使用本地模型路径: {local_model_path}")
            model_name = local_model_path
            local_only = True
        else:
            local_only = False
        
        # 加载模型和分词器
        model, tokenizer = load_model_with_retry(
            model_name,
            cache_dir=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            local_only=local_only,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None
        )
        
        # 将模型移至适当的设备
        model = model.to(device)
        logger.info(f"模型已加载到设备: {device}")
        
        # 初始化游戏管理器
        logger.info("初始化游戏管理器...")
        from rgym_exp.src.game_manager import GameManager
        
        game_manager = None
        retries = 0
        max_game_manager_retries = 3
        
        while retries < max_game_manager_retries:
            try:
                game_manager = GameManager(
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    backend=backend,
                    device=device
                )
                logger.info("游戏管理器初始化成功")
                break
            except Exception as e:
                retries += 1
                logger.error(f"游戏管理器初始化失败 (尝试 {retries}/{max_game_manager_retries}): {e}")
                if retries >= max_game_manager_retries:
                    raise
                logger.info("5秒后重试...")
                time.sleep(5)
        
        # 运行游戏管理器
        logger.info("启动游戏管理器...")
        game_manager.run()
        
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭...")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("清理资源...")
        # 这里可以添加任何必要的清理代码
        logger.info("关闭完成")

if __name__ == "__main__":
    # 设置超时环境变量
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    main()