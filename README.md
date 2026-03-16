# 华为昇腾 AscendC 自定义算子开发

## 项目介绍

本项目基于华为昇腾 AI 计算平台，使用 **AscendC** 框架实现了一个自定义算子 **Softplus**。

项目完整实现了昇腾算子开发的标准流程：

```markdown 
算子实现 → 算子编译 → 算子部署 → ACLNN调用 → 数值验证 → 性能测试
```


算子在 **Atlas 200I DK A2（Ascend 310B SoC）** 开发套件上完成编译、部署与运行验证。

该项目主要用于学习和实践：

- 昇腾 AI 处理器算子开发流程
- AscendC 编程模型
- AI Core 并行执行机制
- Tiling 数据切分策略
- Pipeline 流水线执行
- Double Buffer 性能优化机制

***

# Softplus 算子

Softplus 是一种常见的神经网络激活函数，是 **ReLU 的平滑近似函数**。

其计算公式为：

$$
Softplus(x)=\frac{1}{\beta}\log(1+e^{\beta x})
$$

算子对输入张量 `x` 逐元素执行该计算，并输出结果。

***

# 项目架构

项目整体由两个主要部分组成：

```markdown 
+----------------------+
|   AclNNInvocation    |
|  算子调用与测试程序     |
+----------+-----------+
           |
           v
+----------------------+
|    SoftplusCustom    |
|  AscendC算子开发工程   |
+----------+-----------+
           |
           v
+----------------------+
|       AI Core        |
|   Kernel并行执行计算   |
+----------------------+
```


模块说明：

| 模块               | 作用                               |
| ---------------- | -------------------------------- |
| SoftplusCustom   | AscendC算子开发工程，包含Host侧与Kernel侧实现  |
| AclNNInvocation  | 算子调用与测试程序，通过ACLNN API调用算子        |
| AI Core          | 昇腾AI处理器核心，执行Kernel计算             |

***

# 项目结构

```c++ 
Softplus
├── AclNNInvocation
│   ├── inc
│   ├── input
│   ├── output
│   ├── run.sh
│   ├── scripts
│   │   ├── gen_data.py
│   │   └── verify_result.py
│   └── src
│       ├── main.cpp
│       └── op_runner.cpp
│
├── SoftplusCustom
│   ├── CMakeLists.txt
│   ├── Softplus.json
│   ├── build.sh
│   ├── framework
│   ├── op_host
│   │   ├── softplus.cpp
│   │   └── softplus_tiling.h
│   └── op_kernel
│       └── softplus.cpp
│
├── build_and_run.sh
└── doc
```


***

# 系统设计

## AclNN 算子调用模块

该模块用于在昇腾设备上调用自定义算子并进行测试。

主要组成：

### src

核心代码：

```c++ 
main.cpp
op_runner.cpp
```


主要功能：

- 初始化 ACL 运行环境
- 创建输入输出 Tensor
- 调用 ACLNN API 执行算子
- 获取计算结果

***

### scripts

包含两个 Python 脚本：

```python 
gen_data.py
verify_result.py
```


作用：

- 生成测试输入数据
- 对比 NPU 计算结果与 CPU 计算结果

***

### input / output

```markdown 
input  : 输入数据
output : NPU计算结果
```


***

# 算子实现设计

AscendC 算子开发分为两个部分：

```text 
Host侧
Kernel侧
```


***

# Host侧实现（Tiling）

Host 侧代码位于：

```c++ 
SoftplusCustom/op_host/softplus.cpp
```


Host侧主要负责：

- 计算 tiling 参数
- 将输入数据划分到不同 AI Core
- 将 tiling 参数传递给 kernel

***

## UB大小获取

在昇腾 AI Core 中，数据需要先从 **Global Memory** 搬运到 **Local Memory（Unified Buffer, UB）** 才能参与计算。

由于 Global Memory 访问延迟较高，如果频繁进行小规模数据搬运，会导致算子性能下降。因此在设计 Tiling 策略时，需要尽可能 **减少 Global Memory → UB 的搬运次数**。

一种常见优化策略是：

> 每次搬运尽可能大的数据块，从而减少数据搬运次数。

但单次搬运的数据规模受到 **UB 容量限制**，因此需要首先获取 AI Core 的 UB 大小，再据此计算单个 Tile 可处理的数据量。

UB 大小可以通过 AscendC 提供的接口获取：

```c++ 
ascendcPlatform.GetCoreMemSize(
    platform_ascendc::CoreMemType::UB,
    ub_size
);
```


***

## BLOCK\_SIZE 对齐

Ascend AI Core 的 Vector 执行单元最小执行单位为 **一个 Block**。

```text 
BLOCK_SIZE = 32 Byte
```


因此在进行数据计算时，需要将输入数据字节数 **按 32 字节进行向上对齐**，以满足向量计算单元的对齐要求。

***

## AI Core 数据划分

为了最大化 AI Core 的计算效率，需要让多个 AI Core **并行处理输入数据**。

因此需要将输入数据划分到多个 AI Core 上执行。

在实现中采用 **大小核切分机制**：

- 正常情况下每个核处理 `tilingDataNum`
- 部分核需要多处理一部分数据

因此需要处理尾块数据：

```javascript 
bigCoreTailDataNum
```


从而保证所有数据都能被正确处理。

***

# Kernel实现

Kernel 代码位于：

```c++ 
SoftplusCustom/op_kernel/softplus.cpp
```


Kernel 在 AI Core 上执行 Softplus 的逐元素计算。

***

# Pipeline 执行流程

算子的执行流程采用 **三阶段流水线结构**：

```markdown 
Global Memory → Local Memory → Compute → Global Memory
```


具体执行步骤：

```markdown 
CopyIn   : 从 Global Memory 读取数据到 UB
Compute  : 在 AI Core 上执行计算
CopyOut  : 将结果写回 Global Memory
```


这种 Pipeline 结构可以提高 AI Core 的执行效率。

***

# Double Buffer 机制

为了进一步提高算子的执行效率，在实现中采用 **Double Buffer（双缓冲）机制**。

通过维护两个 Buffer：

```text 
Buffer A
Buffer B
```


实现：

```text 
Buffer A : 执行计算
Buffer B : 预取下一块数据
```


这样可以在计算当前 Tile 的同时加载下一块数据，从而实现：

```markdown 
数据搬运 与 计算 并行执行
```


该机制可以有效 **隐藏 Global Memory 访问延迟**，提高算子整体吞吐率。

***

# 使用方法

## 克隆项目

```bash 
git clone https://github.com/ForestFrame/Softplus.git
cd Softplus
```


***

## 生成测试数据

修改：

```python 
AclNNInvocation/scripts/gen_data.py
```


例如：

```python 
'case7': {
    'x': (torch.rand(6000, 6000) * 20 - 10).to(dtype),
    'beta': 1.0,
    'threshold': 20.0
}
```


***

## 编译并运行算子

运行一键脚本：

```bash 
./build_and_run.sh
```


该脚本会自动完成：

```markdown 
算子编译
算子安装
ACLNN调用
算子测试
```


***

# 项目亮点

### 1 完整实现 AscendC 自定义算子开发流程

实现了从 **算子开发 → 编译 → 部署 → 调用 → 测试** 的完整流程。

***

### 2 AI Core 并行计算

通过 Tiling 将输入数据划分到多个 AI Core 上并行执行，提高计算效率。

***

### 3 Tiling 数据切分策略

根据 UB 大小动态计算单次 Tile 可处理的数据规模，实现高效的数据搬运策略。

***

### 4 Pipeline 流水线优化

实现 CopyIn → Compute → CopyOut 三阶段流水线，提高 AI Core 利用率。

***

### 5 Double Buffer 优化

通过双缓冲实现数据搬运与计算的重叠执行，从而提高算子整体吞吐率。

***

# 学习收获

通过本项目实践，主要收获包括：

- 理解昇腾 AI 计算平台的软件架构
- 掌握 AscendC 自定义算子开发流程
- 熟悉 AI Core 的并行计算模型
- 理解 Tiling 与 Pipeline 的性能优化方法
- 熟悉 CANN 工具链与算子开发环境
