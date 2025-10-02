import requests
import time
import socket
from typing import Optional, Dict, Any, List
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/judge_client.log"),
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

class JudgeClient:
    """Judge API客户端，提供与Judge服务交互的方法，支持网络重试"""
    
    def __init__(self, base_url: str):
        """
        初始化JudgeClient
        
        Args:
            base_url: Judge服务的基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.logger = get_logger()
        self.logger.info(f"初始化JudgeClient，基础URL: {self.base_url}")
    
    @with_retry
    def request_question(self, game_id: str, player_id: str) -> Dict[str, Any]:
        """
        请求一个问题
        
        Args:
            game_id: 游戏ID
            player_id: 玩家ID
        
        Returns:
            包含问题信息的字典
        """
        url = f"{self.base_url}/question"
        payload = {"game_id": game_id, "player_id": player_id}
        
        self.logger.info(f"请求问题 - 游戏ID: {game_id}, 玩家ID: {player_id}")
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"成功获取问题: {result.get('question_id')}")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求问题失败: {e}")
            raise
    
    @with_retry
    def get_current_clue(self, game_id: str, player_id: str) -> Dict[str, Any]:
        """
        获取当前线索
        
        Args:
            game_id: 游戏ID
            player_id: 玩家ID
        
        Returns:
            包含线索信息的字典
        """
        url = f"{self.base_url}/clue"
        payload = {"game_id": game_id, "player_id": player_id}
        
        self.logger.info(f"获取当前线索 - 游戏ID: {game_id}, 玩家ID: {player_id}")
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"成功获取线索: {result.get('clue_id')}")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"获取线索失败: {e}")
            raise
    
    @with_retry
    def submit_answer(self, game_id: str, player_id: str, question_id: str, answer: str) -> Dict[str, Any]:
        """
        提交答案
        
        Args:
            game_id: 游戏ID
            player_id: 玩家ID
            question_id: 问题ID
            answer: 答案内容
        
        Returns:
            包含答案结果的字典
        """
        url = f"{self.base_url}/answer"
        payload = {
            "game_id": game_id,
            "player_id": player_id,
            "question_id": question_id,
            "answer": answer
        }
        
        self.logger.info(f"提交答案 - 游戏ID: {game_id}, 问题ID: {question_id}")
        
        try:
            response = requests.post(url, json=payload, timeout=60)  # 提交答案可能需要更长时间
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"答案提交成功，得分: {result.get('score')}")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"提交答案失败: {e}")
            raise
    
    @with_retry
    def submit_prompt(self, prompt: str, max_tokens: int = 100) -> Dict[str, Any]:
        """
        提交提示词并获取生成结果
        
        Args:
            prompt: 提示词内容
            max_tokens: 最大生成token数
        
        Returns:
            包含生成结果的字典
        """
        url = f"{self.base_url}/prompt"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens
        }
        
        self.logger.info(f"提交提示词，长度: {len(prompt)} 字符")
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"提示词处理成功，生成长度: {len(result.get('generated_text', ''))}")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"提交提示词失败: {e}")
            raise