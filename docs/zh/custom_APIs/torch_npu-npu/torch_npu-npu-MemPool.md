# torch_npu.npu.MemPool

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

`MemPool`表示NPU缓存分配器中的一个内存池。该对象封装了由`NPUCachingAllocator`维护的内存池ID，并管理内存池的生命周期。

通过为`MemPool`指定自定义内存分配器，可以在不替换整个进程当前内存分配器的情况下，将指定代码区域中的NPU内存申请路由到自定义内存分配器，同时保留NPU缓存分配器的缓存能力。内存池需要与`torch_npu.npu.use_mem_pool`配合使用。

该接口对应PyTorch原生接口[torch.cuda.memory.MemPool](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.MemPool.html)。`torch_npu.npu.MemPool`、`torch.npu.MemPool`和`torch.npu.memory.MemPool`功能一致。

## 定义文件

torch_npu/npu/memory.py

## 函数原型

```python
torch_npu.npu.MemPool(allocator=None, use_on_oom=False, no_split=False)
```

也可以使用以下形式调用：

```python
torch.npu.memory.MemPool(allocator=None, use_on_oom=False, no_split=False)
torch.npu.MemPool(allocator=None, use_on_oom=False, no_split=False)
```

## 参数说明

- **allocator**（`torch_npu._C._npu_NPUAllocator`，可选）：用于定义内存池内存申请方式的NPU内存分配器。默认值为`None`，表示按照`NPUCachingAllocator`当前的配置申请内存。通过`torch_npu.npu.NPUPluggableAllocator`创建自定义内存分配器时，可传入其底层分配器`_allocator`。
- **use_on_oom**（`bool`，可选）：当内存池外的内存申请发生内存不足（Out of Memory，OOM）时，是否允许缓存分配器将该内存池作为最后的内存分配来源。默认值为`False`。
- **no_split**（`bool`，可选）：是否禁止拆分该内存池中的内存段。默认值为`False`。设置为`True`时，每个内存段仅包含一个内存块。

## 返回值说明

返回创建好的`torch_npu.npu.MemPool`对象。

## 属性说明

### id

```python
pool.id
```

返回内存池的唯一标识，类型为`Tuple[int, int]`。

## 方法说明

### use_count

```python
pool.use_count()
```

返回底层内存池当前的引用计数，类型为`int`。

`MemPool`对象本身持有一个内存池引用，因此创建内存池后引用计数为1。进入`torch_npu.npu.use_mem_pool`上下文后，该上下文持有另一个引用，引用计数变为2；退出上下文后释放该引用，引用计数恢复为1。该方法主要用于调试内存池的生命周期。

### snapshot

```python
pool.snapshot()
```

返回该内存池在所有NPU设备上的内存分配器状态快照，类型为`list`。快照已根据当前内存池的`id`进行过滤，列表中的每个元素表示该内存池中的一个内存段。理解快照内容需要熟悉NPU缓存分配器的内部实现。

与PyTorch原生`torch.cuda.memory.MemPool.snapshot(include_traces=True)`接口不同，当前NPU接口不支持`include_traces`参数。

## 约束说明

- `MemPool`对象需要与`torch_npu.npu.use_mem_pool`配合使用，才能将内存申请路由到指定内存池。
- `torch_npu.npu.use_mem_pool`仅将当前线程中的内存申请路由到指定内存池。在上下文中创建的新线程不会自动使用该内存池。
- 设置`use_on_oom=True`后，该内存池中的空闲内存可以在内存池外的内存申请发生OOM时被使用。
- 设置`no_split=True`后，内存池不会拆分内存段，每个内存段仅包含一个内存块，可能增加内存占用。
- 回收内存池前，需要确保已退出`torch_npu.npu.use_mem_pool`上下文，并且使用该内存池的Tensor均已释放。当仅剩`MemPool`对象本身持有内存池引用时，删除`MemPool`对象会释放内存池中的空闲缓存并将内存归还系统。
- 使用自定义内存分配器时，需要保证内存申请和释放函数正确实现，否则可能出现内存申请失败、内存泄漏等问题。

## 调用示例

**使用默认NPU缓存分配器**：

```python
import torch
import torch_npu

torch.npu.set_device(0)

pool = torch.npu.memory.MemPool()
print(pool.id)
assert pool.use_count() == 1

with torch.npu.use_mem_pool(pool):
    tensor = torch.empty((1024, 1024), device="npu")
    assert pool.use_count() == 2

assert pool.use_count() == 1

snapshot = pool.snapshot()
print(len(snapshot))

# 释放Tensor后再删除内存池，回收内存池中的空闲缓存。
del tensor
del pool
```

**为指定代码区域使用自定义内存分配器**：

```python
import torch
import torch_npu

torch.npu.set_device(0)

allocator = torch_npu.npu.NPUPluggableAllocator(
    "./pluggable_allocator_extensions.so",
    "my_malloc",
    "my_free"
)
pool = torch.npu.memory.MemPool(allocator._allocator)

with torch.npu.use_mem_pool(pool):
    # tensor的内存通过allocator指定的自定义内存分配器申请。
    tensor = torch.empty((1024, 1024), device="npu")
```
