"""
配置管理模块
提供配置的保存和加载功能，支持前端调用
"""

from . import commands
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from pytauri.webview import WebviewWindow

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class _BaseModel(BaseModel):
    """Base model that accepts camelCase from JS and snake_case from Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class SaveConfigRequest(_BaseModel):
    """Save config request model."""
    config_name: str
    config: Dict[str, Any]


class LoadConfigRequest(_BaseModel):
    """Load config request model."""
    config_name: Optional[str] = None


class GetConfigValueRequest(_BaseModel):
    """Get config value request model."""
    config_name: str
    key: str
    default: Optional[Any] = None


class SetConfigValueRequest(_BaseModel):
    """Set config value request model."""
    config_name: str
    key: str
    value: Any


class DeleteConfigRequest(_BaseModel):
    """Delete config request model."""
    config_name: str


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，默认为项目根目录的 config.json
        """
        if config_file is None:
            # 默认配置文件路径：项目根目录/config.json
            self.config_file = Path(__file__).parent.parent.parent.absolute() / "config.json"
        else:
            self.config_file = config_file
    
    def _load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有配置，键为配置名称，值为配置内容
        """
        try:
            if not self.config_file.exists():
                return {}
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                all_configs = json.load(f)
            
            # 确保返回的是字典格式
            if isinstance(all_configs, dict):
                return all_configs
            else:
                return {}
        except Exception as e:
            print(f"加载配置失败: {e}")
            return {}
    
    def _save_all_configs(self, all_configs: Dict[str, Dict[str, Any]]) -> bool:
        """
        保存所有配置
        
        Args:
            all_configs: 所有配置，键为配置名称，值为配置内容
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保配置文件所在目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存配置到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(all_configs, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def save_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """
        保存配置（按名称）
        
        Args:
            config_name: 配置名称
            config: 配置字典
            
        Returns:
            bool: 保存是否成功
        """
        all_configs = self._load_all_configs()
        all_configs[config_name] = config
        return self._save_all_configs(all_configs)
    
    def load_config(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置（按名称）
        
        Args:
            config_name: 配置名称，若为 None 则返回所有配置
            
        Returns:
            Dict[str, Any]: 配置字典，如果配置不存在则返回空字典
        """
        all_configs = self._load_all_configs()
        if config_name is None:
            return all_configs
        return all_configs.get(config_name, {})
    
    def get_config_value(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        获取配置值（按名称和键）
        
        Args:
            config_name: 配置名称
            key: 配置键
            default: 默认值，如果键不存在则返回默认值
            
        Returns:
            Any: 配置值或默认值
        """
        config = self.load_config(config_name)
        return config.get(key, default)
    
    def set_config_value(self, config_name: str, key: str, value: Any) -> bool:
        """
        设置配置值（按名称和键）
        
        Args:
            config_name: 配置名称
            key: 配置键
            value: 配置值
            
        Returns:
            bool: 设置是否成功
        """
        all_configs = self._load_all_configs()
        if config_name not in all_configs:
            all_configs[config_name] = {}
        all_configs[config_name][key] = value
        return self._save_all_configs(all_configs)
    
    def list_configs(self) -> List[str]:
        """
        列出所有配置名称
        
        Returns:
            List[str]: 配置名称列表
        """
        all_configs = self._load_all_configs()
        return list(all_configs.keys())
    
    def delete_config(self, config_name: str) -> bool:
        """
        删除配置（按名称）
        
        Args:
            config_name: 配置名称
            
        Returns:
            bool: 删除是否成功
        """
        all_configs = self._load_all_configs()
        if config_name in all_configs:
            del all_configs[config_name]
            return self._save_all_configs(all_configs)
        return True


# 创建全局配置管理器实例
config_manager = ConfigManager()


# 前端可调用的方法
@commands.command()
async def save_config(body: SaveConfigRequest, webview_window: WebviewWindow) -> Dict[str, Any]:
    """
    保存配置（前端调用）
    
    Args:
        body: 请求体，包含 config_name 和 config
        webview_window: Webview 窗口
        
    Returns:
        Dict[str, Any]: 结果字典，包含 success 字段
    """
    success = config_manager.save_config(body.config_name, body.config)
    return {"success": success}


@commands.command()
async def load_config(body: LoadConfigRequest, webview_window: WebviewWindow) -> Dict[str, Any]:
    """
    加载配置（前端调用）
    
    Args:
        body: 请求体，包含可选的 config_name
        webview_window: Webview 窗口
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    return config_manager.load_config(body.config_name)


@commands.command()
async def get_config_value(body: GetConfigValueRequest, webview_window: WebviewWindow) -> Dict[str, Any]:
    """
    获取配置值（前端调用）
    
    Args:
        body: 请求体，包含 config_name、key 和可选的 default
        webview_window: Webview 窗口
        
    Returns:
        Dict[str, Any]: 结果字典，包含 value 字段
    """
    value = config_manager.get_config_value(body.config_name, body.key, body.default)
    return {"value": value}


@commands.command()
async def set_config_value(body: SetConfigValueRequest, webview_window: WebviewWindow) -> Dict[str, Any]:
    """
    设置配置值（前端调用）
    
    Args:
        body: 请求体，包含 config_name、key 和 value
        webview_window: Webview 窗口
        
    Returns:
        Dict[str, Any]: 结果字典，包含 success 字段
    """
    success = config_manager.set_config_value(body.config_name, body.key, body.value)
    return {"success": success}


@commands.command()
async def list_configs() -> Dict[str, List[str]]:
    """
    列出所有配置名称（前端调用）
    
    Args:
        body: 无请求体
        webview_window: Webview 窗口
        
    Returns:
        Dict[str, List[str]]: 结果字典，包含 configs 字段
    """
    configs = config_manager.list_configs()
    return {"configs": configs}


@commands.command()
async def delete_config(body: DeleteConfigRequest, webview_window: WebviewWindow) -> Dict[str, Any]:
    """
    删除配置（前端调用）
    
    Args:
        body: 请求体，包含 config_name
        webview_window: Webview 窗口
        
    Returns:
        Dict[str, Any]: 结果字典，包含 success 字段
    """
    success = config_manager.delete_config(body.config_name)
    return {"success": success}


# 导出供 commands.generate_handler 使用的函数
__all__ = [
    "save_config",
    "load_config",
    "get_config_value",
    "set_config_value",
    "list_configs",
    "delete_config"
]


