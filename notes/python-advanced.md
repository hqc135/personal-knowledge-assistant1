# Python 进阶笔记

## 并发编程

### GIL (Global Interpreter Lock)
- CPython 的 GIL 限制同一时刻只有一个线程执行 Python 字节码
- CPU 密集型任务用 `multiprocessing`，IO 密集型用 `threading` 或 `asyncio`
- Python 3.13 引入 free-threaded mode (实验性去除 GIL)

### asyncio
- `async def` 定义协程，`await` 等待异步操作
- `asyncio.gather()` 并发执行多个协程
- 适用场景：网络请求、文件 IO、数据库查询

```python
import asyncio

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## 类型系统

### Type Hints 最佳实践
- 使用 `from __future__ import annotations` 延迟类型求值
- 复杂类型用 `TypeAlias` 或 `TypeVar`
- 用 `mypy --strict` 检查类型安全

### 常用类型
```python
from typing import Optional, Union, TypeVar, Protocol

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None
```

## 设计模式

### 常用模式
1. **单例模式**: 用 `__new__` 或模块级变量实现
2. **工厂模式**: 根据参数创建不同类的实例
3. **观察者模式**: 发布-订阅，解耦事件触发和处理
4. **装饰器模式**: Python 的 `@decorator` 语法天然支持
5. **策略模式**: 将算法封装为可互换的策略对象
