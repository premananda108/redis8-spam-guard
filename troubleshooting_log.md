# Troubleshooting Log

## Issue: `TypeError: duplicate base class TimeoutError` on startup

**Date:** 2025-07-25

### Symptoms

When starting the application with `uvicorn main:app --reload`, the application fails with the following traceback:

```
Traceback (most recent call last):
  File "D:\redis8-spam-guard\main.py", line 13, in <module>
    import aioredis
  File "D:\redis8-spam-guard\.venv\Lib\site-packages\aioredis\__init__.py", line 1, in <module>
    from aioredis.client import Redis, StrictRedis
  File "D:\redis8-spam-guard\.venv\Lib\site-packages\aioredis\client.py", line 32, in <module>
    from aioredis.connection import (
        ...
    )
  File "D:\redis8-spam-guard\.venv\Lib\site-packages\aioredis\connection.py", line 33, in <module>
    from .exceptions import (
        ...
    )
  File "D:\redis8-spam-guard\.venv\Lib\site-packages\aioredis\exceptions.py", line 14, in <module>
    class TimeoutError(asyncio.TimeoutError, builtins.TimeoutError, RedisError):
        pass
TypeError: duplicate base class TimeoutError
```

### Cause

The error is caused by an incompatibility between the `aioredis` library and newer versions of Python. In recent Python versions, `asyncio.TimeoutError` and `builtins.TimeoutError` are the same class. The `aioredis` code attempts to inherit from both, causing a `TypeError`.

### Resolution

The issue was resolved by modifying the source code of the installed `aioredis` library within the virtual environment.

1.  **File Modified**: `D:\redis8-spam-guard\.venv\Lib\site-packages\aioredis\exceptions.py`
2.  **Change**: The `TimeoutError` class definition was changed to remove the redundant base class.

    **Original line:**
    ```python
    class TimeoutError(asyncio.TimeoutError, builtins.TimeoutError, RedisError):
        pass
    ```

    **Modified line:**
    ```python
    class TimeoutError(asyncio.TimeoutError, RedisError):
        pass
    ```

This change eliminates the error and allows the application to start successfully.

```