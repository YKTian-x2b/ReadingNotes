# OpenMP核心技术指南

## 1.并行化循环

- 块状分配对于缓存局部性和内存预取更为有效，即使在编译器向量化的情况下。
  - 与之相对的是周期分配
- 共享工作构造结束时有一个隐式栅栏，可以通过nowait子句来禁用它。

- 应该尝试使用不同数量的线程和循环调度来优化程序。然后思考是否能使用nowait子句。
- 尝试思考循环主体的数据结构如何映射到多处理器系统的缓存上，以提高性能。

## 2.OpenMP数据环境

- 一个变量不能既是私有的，又是文件作用域范围的。
- 应该主要使用private，只有需要一个初始化的私有变量时才用firstprivate。
  - 因为如果平日*private标记的变量是大型数据结构，firstprivate意味着大量的线程内存操作和数据复制。

- OpenMP的设计目标：在并行化时不破坏或改变串行程序。
  - 可以借助数据环境子句来实现该目标。
- 理论上，default(none)是编写OpenMP的标准做法。

## 3.OpenMP任务

- single构造，由线程组任一线程工作，其他线程在结尾隐式栅栏处等待。
- 执行task的线程，将在最近一个隐式栅栏或taskwait处完成之前创建的所有任务。

## 4.OpenMP内存模型

- 冲刷：强制线程变量的临时视图和内存中的变量值保持一致。正在读取的变量被标记为无效，下次访问它们时将从内存加载。被线程写入缓存/寄存器文件/其他写缓存区的变量也会被写入内存。
  - 就是内存以上的存储层次结构失效后遗症。
  - 内存读+写直达
- 冲刷集：线程间共享的所有变量的集合。
- 在程序顺序/编译器顺序/执行顺序存在的情况下，内存操作只需遵守程序顺序。
- 安全的混合读写共享变量的操作，**依次**进行：
  - 写入
  - 刷回
  - 刷出
  - 读取

- 冲刷是为了内存一致，同步是为了冲刷和读写符合上述顺序。同步需要栅栏和临界区来完成。
- OpenMP在需要它们的地方暗含了冲刷：
  - 当一个新的线程组被parallel构造分叉时。
  - 线程进入临界区
  - 线程退出临界区
  - 进入任务区域
  - 退出任务区域
  - 退出任务等待
  - 退出显式或隐式栅栏

- 即：临界区前后，数据环境构造和析构时。

## 5.超越通用核心的多线程

### 5.1通用核心的附加子句

#### 5.1.1 并行构造

- if子句：条件判断并行线程组的大小。

#### 5.1.2 共享工作循环构造

- lastprivate子句标识的变量在共享工作循环结构的线程组中共享，变量的原始变量将被赋值循环顺序执行的最后一次迭代的值。
- 一个变量出现在一个以上的数据环境子句中是非法的。
  - firstprivate和lastprivate的组合除外
- 调度：
  - 静态调度
  - 动态调度
  - 启发式调度：动态调度的一种，chunk_size会在每个新的分块迭代中减小，直到最小值
  - 自动调度：可能不同于上述调度的一种由编译器和运行时安排的调度方式
  - 运行时调度：为了不编译，用环境变量调整调度方法

```C++
// 运行时调度
#pragma omp for schedule(runtime)
// export OMP_SCHEDULE="dynamic,7"

// void omp_set_schedule(omp_sched_t kind, int chunk_size);
// void omp_get_schedule(omp_sched_t * kind, int * chunk_size);
omp_set_schedule(omp_sched_dynamic, 7);
```

- collapse(n)子句：规定跟在共享工作循环构造后的n个循环将被合并成一个隐式大循环。任何额外子句都会应用到该循环中。

#### 5.1.3 任务构造

- if：判定为假则任务会被遇到任务的线程立即执行。
- untied：openmp默认任务和线程是绑定的。untied显式解绑。
  - 绑定：任务挂起并恢复后，执行该任务的是原线程。
- 任务调度点：允许任务切换。
  - 任务创建/完成/等待
  - 栅栏
- priority：任务优先级。并不会被强制执行，所以有depend。
- depend：依赖类型+变量列表。用来构造按序执行的DAG。依赖类型：
  - in：等待带相同变量的out任务结束。
  - out
  - inout
- 其他任务构造子句：taskloop/taskgroup/final/mergeable。

### 5.2通用核心中缺失的多线程功能

- threadprivate：线程私有内存，在线程内部是全局的。声明性指令，声明的变量得是文件作用域/命名空间作用域/静态块作用域的。依托于宿主语言的规则来初始化。也可以在并行构造上用copyin(list)子句动态初始化。

- master构造定义了一个由线程组主线程执行的工作块。末尾无隐式栅栏。也就是说，主线程跑master区域的代码，其他线程继续执行master构造后的语句。
- aotmic构造：原子读取或写入或更新变量。多个线程同时执行相同atomic构造会串行执行。

```C++
#pragma omp atomic
	full_sum += partial_sum;
```

- OMP_STACKSIZE：线程栈大小。当要求系统提供的栈内存大小大于可以提供的量时，系统如何相应是未定义的。栈溢出也是未定义的。

## 6.同步和OpenMP内存模型

- 集体同步操作：临界区和栅栏
- 成对同步：顺序一致的原子操作
- 数据同步和线程同步：栅栏将两者结合了起来。互斥结构（临界区和锁）最好只用于数据同步。线程同步则依赖顺序一致的原子操作。
- 编译器不被允许围绕冲刷来重新排序指令。
  - 这里的“围绕“可能是指冲刷前后的顺序点。

- sequenced-before：单线程 执行的事件之间的偏序。如果A的评估在B的执行之前完成，则说A被排序在B之前。
- 顺序点：程序执行中的点，在此处涉及的有关语句（及其附带结果）均已完成。C的常见顺序点
  - 一个完整表达式结束（；）
  - 逻辑运算符/条件运算符/逗号
  - 函数调用前，尤其是参数评估之后，调用之前
  - 函数返回前

- 顺序点排序的三种情况：
  - sequenced-before：顺序点间的关系是一个顺序点接着另一个顺序点。有明确顺序。
    - 单行多初始化/for(;;)

  - indeterminately sequenced：顺序点间的关系是以某种顺序执行，但顺序未定义。
    - 函数调用顺序

  - unsequenced：顺序点发生冲突并导致未定义的结果
    - 函数参数评估


- 单线程的happens-before就是sequenced-before。
- synchronized-with：多个线程围绕一个事件协调执行，以定义其执行顺序约束时，该关系成立。
  - 只有栅栏和原子操作能建立synchronized-with关系
- 锁的设置和解除都意味着一次冲刷。所以，锁默认进行了数据互斥和同步。
- C++常用的内存顺序：
  - seq_cst（顺序一致）：执行顺序==程序顺序。对所有线程来说，内存的加载和存储将以相同的顺序发生。
    - 编译器不能将共享变量的更新围绕原子操作移动。会是sequenced-before的。
  - release
  - acquire
  - acquire_release

- 原子操作没有上述任一子句时，被称为松弛原子，这**不**意味着内存load/store有顺序。

## 7.超越OpenMP通用核心的硬件

- NUMA情况下，需要控制内存数据和线程的“距离”来优化程序。
  - 控制线程亲和力
  - 管理数据局部性
- SIMD：用向量寄存器完成运算
- 设备构造：OpenMP针对device端的编译器指令













## 100.电脑参数

```bash
# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数

# 查看物理CPU个数		1个CPU
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

# 查看每个物理CPU中core的个数(即核数)	8核
cat /proc/cpuinfo| grep "cpu cores"| uniq		

# 查看逻辑CPU的个数		12线程/12个逻辑CPU
cat /proc/cpuinfo| grep "processor"| wc -l


# 每个核都有自己的L1/L2/L3缓存
root@JiguangPro:# cat /sys/devices/system/cpu/cpu11/cache/index0/level
1
root@JiguangPro:# cat /sys/devices/system/cpu/cpu11/cache/index0/type
Data
root@JiguangPro:# cat /sys/devices/system/cpu/cpu11/cache/index0/size
32K
root@JiguangPro:# cat /sys/devices/system/cpu/cpu11/cache/index0/coherency_line_size
64
# cpu* 是逻辑CPU；index0/1/2/3分别指 D-L1cache/I-L1cache/L2cache/L3cache；


```

