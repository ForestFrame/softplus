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

---

# Softplus 算子

Softplus 是一种常见的神经网络激活函数，是 **ReLU 的平滑近似函数**。

其原始计算公式为：

$$
Softplus(x)=\frac{1}{\beta}\log(1+e^{\beta x})
$$

算子对输入张量 `x` 逐元素执行该计算，并输出结果。

---

# 项目架构

项目整体由两个主要部分组成：

```markdown
+----------------------+
|   AclNNInvocation    |
|    AcendCL算子调用    |
+----------+-----------+
           |
           v
+----------------------+
|    SoftplusCustom    |
|  AscendC算子开发工程  |
+----------+-----------+
           |
           v
+----------------------+
|       AI Core        |
|   Kernel并行执行计算  |
+----------------------+
```

模块说明：


| 模块            | 作用                                          |
| --------------- | --------------------------------------------- |
| SoftplusCustom  | AscendC算子开发工程，包含Host侧与Kernel侧实现 |
| AclNNInvocation | 算子调用与测试程序，通过ACLNN API调用算子     |
| AI Core         | 昇腾AI处理器核心，执行Kernel计算              |

---

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

---

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

---

### scripts

包含两个 Python 脚本：

```python
gen_data.py
verify_result.py
```

作用：

- 生成测试输入数据
- 对比 NPU 计算结果与 CPU 计算结果

---

### input / output

```markdown
input  : 输入数据
output : NPU计算结果
```

---

# 算子实现设计

AscendC 算子开发分为两个部分：

```text
Host侧
Kernel侧
```

---

# Host侧实现（Tiling）

Host 侧代码位于：

```c++
SoftplusCustom/op_host/softplus.cpp
```

Host侧主要负责：

- 计算 tiling 参数
- 将输入数据划分到不同 AI Core
- 将 tiling 参数传递给 kernel

---

## UB大小获取

在昇腾 AI Core 中，数据需要先从 **Global Memory** 搬运到 **Local Memory（Unified Buffer, UB）** 才能参与计算。

由于 Global Memory 访问延迟较高，如果频繁进行小规模数据搬运，会导致算子性能下降。因此在设计 Tiling 策略时，需要尽可能 **减少 Global Memory → UB 的搬运次数**。

一种常见优化策略是：

> 每次搬运尽可能大的数据块，从而减少数据搬运次数。

但单次搬运的数据规模受到 **UB 容量限制**，因此需要首先获取 AI Core 的 UB 大小，再据此计算单个 Tile 可处理的数据量`tilingDataNum`。

UB 大小可以通过 AscendC 提供的接口获取：

```c++
ascendcPlatform.GetCoreMemSize(
    platform_ascendc::CoreMemType::UB,
    ub_size
);

...

alignNum = BLOCK_SIZE / sizeofdatatype;
tilingBlockNum = ((ub_size) / BLOCK_SIZE / BUFFER_NUM) / ubPartNum;
tilingDataNum = tilingBlockNum * alignNum;
```

---

## BLOCK\_SIZE 对齐

Ascend AI Core 的 Vector 执行单元最小执行单位为 **一个 Block**。

```text
BLOCK_SIZE = 32 Byte
```

因此在进行数据计算时，需要将输入数据字节数 **按 32 字节进行向上对齐**，以满足向量计算单元的对齐要求。

---

## AI Core 数据划分

为了最大化 AI Core 的计算效率，需要让多个 AI Core **并行处理输入数据**。

因此，需要先将输入数据划分到多个 AI Core 上执行。在本工程中，采用的是**大小核切分 + 核内尾块处理**的方式：

1. 首先将总 block 数 `totalBlockNum` 按 AI Core 数量进行划分，得到两类核：

   - 大核：每个大核处理 `bigCoreBlockNum`
   - 小核：每个小核处理 `smallCoreBlockNum`

   其中，大核比小核多处理一个 block，因此：

   - `bigCoreBlockNum` 表示**每一个大核**的总工作量
   - `smallCoreBlockNum` 表示**每一个小核**的总工作量
2. 在完成上述划分后，每个核再根据单轮可处理的 block 数 `tilingBlockNum` 进行循环处理：

   - 完整轮数分别为 `bigCoreLoopNum` 和 `smallCoreLoopNum`
   - 不足一整轮的剩余部分就是尾块，对应 `bigCoreTailBlockNum` 和 `smallCoreTailBlockNum`

### 尾块划分的理解

需要注意的是，尾块不是针对全局总数据量来计算的，而是针对**每个核自己的总工作量**来计算的。

也就是说，在步骤 1 中，总 block 数已经被分配给各个核；步骤 2 中的尾块，表示的是某个核在处理自己那一段数据时，按 `tilingBlockNum` 切分后最后剩余的一部分。

因此，尾块不是“全局最后剩下的一小块数据交给某一个核处理”，而是“每个核在核内分轮处理时可能产生的最后一轮残余数据”。

进一步说：

- 如果 `bigCoreBlockNum % tilingBlockNum != 0`，那么所有大核都会有尾块
- 如果 `smallCoreBlockNum % tilingBlockNum != 0`，那么所有小核都会有尾块

所以，尾块可能只出现在大核中，只出现在小核中，也可能两类核同时存在尾块，而不是只出现在某一个核上。

### 举例说明

假设 Host 端最终计算得到如下参数：

```text
coreNum = 5
bigCoreNum = 2
smallCoreNum = 3
bigCoreBlockNum = 10
smallCoreBlockNum = 8
tilingBlockNum = 4
```

其含义为：

- 一共使用 5 个 AI Core
- 前 2 个为大核，后 3 个为小核
- 每个大核处理 10 个 block
- 每个小核处理 8 个 block
- 每轮最多处理 4 个 block

对于大核：

```text
bigCoreLoopNum = 10 / 4 = 2
bigCoreTailBlockNum = 10 % 4 = 2
```

即每个大核先执行 2 轮完整计算，每轮处理 4 个 block，最后再处理 1 个尾块轮次，共 2 个 block。

对于小核：

```text
smallCoreLoopNum = 8 / 4 = 2
smallCoreTailBlockNum = 8 % 4 = 0
```

即每个小核执行 2 轮完整计算，每轮处理 4 个 block，不存在尾块。

AI Core 划分示意如下：

```text
core0 (大核) : [4] [4] [2]
core1 (大核) : [4] [4] [2]
core2 (小核) : [4] [4]
core3 (小核) : [4] [4]
core4 (小核) : [4] [4]
```

由此可以看出，在这个例子中，所有大核都有尾块，而所有小核都没有尾块。尾块是核内切分产生的，不是全局只剩一块交给某一个核处理。

### 具体实现

```c++
// 大小核个数
bigCoreNum = totalBlockNum % coreNum;  // 总block数/核数，取余数
smallCoreNum = coreNum - bigCoreNum;

// 每个大/小核处理总Block数
smallCoreBlockNum = totalBlockNum / coreNum;  // 总blcok/核数，平均每个核处理的总block数，向下取整即小核处理的block数
bigCoreBlockNum = smallCoreBlockNum + 1;

// 每个大/小核处理总数据个数
bigCoreDataNum = bigCoreBlockNum * alignNum;  // 每个大核处理总block数*block对齐数据个数，即每个大核处理的总数据个数
smallCoreDataNum = smallCoreBlockNum * alignNum;

// 每个大/小核最后一次处理的Block数
bigCoreTailBlockNum = bigCoreBlockNum % tilingBlockNum;  // 大核处理总block数%单核单次tiling可处理的block数，即大核处理处理的尾块数
smallCoreTailBlockNum = smallCoreBlockNum % tilingBlockNum;

// 每个大/小核最后一次处理的数据个数
bigCoreTailDataNum = bigCoreTailBlockNum * alignNum;
smallCoreTailDataNum = smallCoreTailBlockNum * alignNum;

// 每个大/小核常规批次搬运次数，最后一次的搬运另算
bigCoreLoopNum = bigCoreBlockNum / tilingBlockNum;  // 大核处理总block数/单核单次tiling可处理的block数，向下取整即标准搬运次数，不包括尾块那一次
smallCoreLoopNum = smallCoreBlockNum / tilingBlockNum;
```

# Kernel实现

Kernel 代码位于：

```c++
SoftplusCustom/op_kernel/softplus.cpp
```

## Kernel侧地址定位

在完成大小核工作量划分之后，还需要进一步确定**每个核从全局输入张量的哪个位置开始处理数据**。这一部分由 `globalBufferIndex`、`xGm` 以及 `progress * tilingDataNum` 共同决定。

### 1. 每个核的基础地址偏移

`globalBufferIndex` 表示**当前核在全局数据中的起始偏移量**，也就是当前核负责数据段的起点。

对于大核：

- 由于大核排在前面，且每个大核处理的数据量相同，因此它们的起始地址可以直接写成：

```c++
globalBufferIndex = bigCoreDataNum * coreIndex;
```

对于小核：

- 小核排在大核之后
- 如果仍然直接使用 `bigCoreDataNum * coreIndex`，就等价于“假设前面的所有核都是大核”
- 但实际上，小核前面可能已经出现了一些同样是小核的核，而这些小核处理的数据量比大核更少
- 因此需要把“多算出来的那一部分地址”减掉

修正公式为：

```c++
globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIndex - bigCoreNum);
```

其中：

- `bigCoreDataNum - smallCoreDataNum` 表示每个小核相比大核少处理的数据量
- `coreIndex - bigCoreNum` 表示当前小核之前已经出现了多少个小核

因此，这个修正量表示的是：**前面这些小核一共比“大核假设”少处理了多少数据**。

### 2. 每个核的基地址

在得到 `globalBufferIndex` 之后，就可以基于原始输入地址 `x` 得到当前核自己的基地址：

```c++
xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex, this->coreDataNum);
```

这一步可以理解为：

- `x` 是整块输入数据的起始地址
- `x + globalBufferIndex` 是当前核负责数据段的起始地址
- `xGm` 则表示“当前核自己的全局内存视图”

同理，输出张量 `yGm` 也是按同样方式设置的。

### 3. 每轮数据的地址偏移

在 `CopyIn` 流程中，使用 `AscendC::DataCopy` 在 Global Memory 和 UB 之间搬运数据：

```c++
AscendC::DataCopy(xLocal, xGm[progress * this->tilingDataNum], dataNum);
```

这里的含义是：

- `xGm` 已经指向当前核负责数据段的起点
- `progress * tilingDataNum` 表示当前核内部第 `progress` 轮处理的数据偏移
- `dataNum` 表示本轮实际搬运的数据量

因此，当前轮次访问的实际地址可以理解为：

```text
实际地址 = x + globalBufferIndex + progress * tilingDataNum
```

需要注意的是：

- `tilingDataNum` 表示正常轮次中每次搬运的数据量
- `dataNum` 表示当前轮次真实搬运的数据量
- 对于普通轮次，通常有 `dataNum = tilingDataNum`
- 对于尾块轮次，`dataNum = tailDataNum`

因此，`globalBufferIndex` 决定的是**当前核的整体起点**，而 `progress * tilingDataNum` 决定的是**当前核内部第几轮处理的数据位置**。两者结合起来，才能确定每一轮实际访问的全局内存地址。

---

## Pipeline 执行流程

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

---

## Double Buffer 机制

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

---

## 算子的三种实现方式

### 原始公式

$$
Softplus(x)=\frac{1}{\beta}\log(1+e^{\beta x})
$$

这是 Softplus 最直接的数学表达式，实现思路也非常清晰：

1. 计算 $\beta x$
2. 计算指数 $e^{\beta x}$
3. 计算 $1 + e^{\beta x}$
4. 取对数 $\log(1 + e^{\beta x})$
5. 乘以 $\frac{1}{\beta}$ 得到最终结果

这种实现方式的优点在于：

- 与理论公式完全一致，直观易懂
- 代码实现简单，适合作为算子开发的初版进行功能验证

#### 潜在问题

尽管实现简单，但原始公式存在明显的数值风险：

- 当 $x$ 较大时，$e^{\beta x}$ 增长迅速，容易导致溢出
- 对于输入范围宽、数据规模大的情况，精度和稳定性可能受影响

#### 比赛经历与验证

在比赛中，我最初提交的版本仍采用了原始公式。提交后官方反馈如下：

> 作品泛化性不足，未对 threshold 做任何处理，较为明显地针对用例答题，因此成绩被判为无效。

实际上，我在比赛中完成了三种算子的实现，但在提交阶段发现原始公式在当时样例下能够通过验证，因此误以为该实现方式可接受。

官方测试脚本 `AclNNInvocation/scripts/test_op.py` 的校验逻辑并非严格逐元素匹配，而是采用带误差容忍的方式：

- `float32` 类型使用 `rtol = 1e-4`、`atol = 1e-4`
- 低精度类型使用 `rtol = 1e-3`、`atol = 1e-3`
- 除了允许单点误差外，还允许一定比例的数据点不满足误差条件

核心判断逻辑如下：

```python
if real_result.numel() * rtol < err_num:
    print(f"[ERROR] result error")
    return False
```

这意味着即使原始公式在数值上存在一定不稳定性，也可能在部分样例上通过验证。

#### 总结

- 原始公式适合功能验证和初期实现，但缺乏泛化能力
- 没有对 `threshold` 分段处理的版本，在输入范围扩大或边界条件变化时容易出错
- 比赛中未获奖的主要原因正是泛化性不足，而不仅仅是“样例通过与否”

### 数值稳定公式

$$
Softplus(x) = \frac{1}{\beta} \left\{ \max(\beta x, 0) + \log\big(1 + e^{-|\beta x|}\big) \right\}
$$

### 分段函数形式

$$
\text{Softplus}(x) =
\begin{cases} 
x, & x > \text{threshold} \\
\frac{1}{\beta} \log(1 + e^{\beta x}), & -\text{threshold} \le x \le \text{threshold} \\
\frac{1}{\beta} e^{\beta x}, & x < -\text{threshold}
\end{cases}
$$

#### `compare + select` 机制实现分段函数

在 kernel 端实现分段函数时，逐元素使用 `if` 判断效率非常低。为此，我们采用 `compare + select` 的机制，将分段逻辑并行化：

1. `Compare`
   使用 `AscendC::CompareScalar` 对输入向量与阈值 `threshold` 并行比较，生成布尔掩码 `mask`：

   ```c++
   AscendC::CompareScalar(mask, temp2, threshold, AscendC::CMPMODE::GT, dataNum);
   ```

   其中 `mask[i]` 为 `true` 表示该元素大于阈值，应选取对应分段的输出。
2. 计算各分段函数值
   对所有元素并行计算 Softplus 的公式部分：

   ```c++
   AscendC::Exp(temp2, temp2, dataNum);                 // e^(beta*x)
   AscendC::Adds(temp2, temp2, scalar, dataNum);        // 1 + e^(beta*x)
   AscendC::Ln(temp2, temp2, dataNum);                  // log(1 + e^(beta*x))
   AscendC::Muls(temp2, temp2, 1.0f / beta, dataNum);   // (1/beta) * log(1 + e^(beta*x))
   ```
3. Select
   根据掩码 `mask` 选择对应分段输出，完成分段逻辑：

   ```c++
   AscendC::Select(temp1, mask, temp1, temp2, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, dataNum);
   ```

   - `temp1` 中存放大于阈值的原始值
   - `temp2` 中存放公式计算的 Softplus 值
   - `mask` 指示哪些元素选 `temp1`，哪些选 `temp2`

---

# 使用方法

## 克隆项目

```bash
git clone https://github.com/ForestFrame/Softplus.git
cd Softplus
```

---

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

---

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

---

# 项目亮点

### 1 完整实现 AscendC 自定义算子开发流程

实现了从 **算子开发 → 编译 → 部署 → 调用 → 测试** 的完整流程。

---

### 2 AI Core 并行计算

通过 Tiling 将输入数据划分到多个 AI Core 上并行执行，提高计算效率。

---

### 3 Tiling 数据切分策略

根据 UB 大小动态计算单次 Tile 可处理的数据规模，实现高效的数据搬运策略。

---

### 4 Pipeline 流水线优化

实现 CopyIn → Compute → CopyOut 三阶段流水线，提高 AI Core 利用率。

---

### 5 Double Buffer 优化

通过双缓冲实现数据搬运与计算的重叠执行，从而提高算子整体吞吐率。

---

# 学习收获

通过本项目实践，主要收获包括：

- 理解昇腾 AI 计算平台的软件架构
- 掌握 AscendC 自定义算子开发流程
- 熟悉 AI Core 的并行计算模型
- 理解 Tiling 与 Pipeline 的性能优化方法
- 熟悉 CANN 工具链与算子开发环境
