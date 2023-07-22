# CUDA_C编程权威指南学习

## 1.基于CUDA的异构并行计算

- GPU容量的重要特征：CUDA核心数量和内存大小。
- GPU性能的评估指标：峰值计算性能（GFLOPS or TFLOPS）和内存带宽（GB/s）。



## 2.CUDA编程模型

- CPU Mem和 GPU Mem 通过PCI-e总线连接。Unified Memory编程模型连接了两个内存空间，可以用单个指针访问，避免了拷贝。

~~~C++
// 在GPU内存里分配空间
cudaError_t cudaMalloc (void **devPtr, size_t size);
// 同步执行 阻塞 
// kind: cudaMemcpyHostToHost HostToDevice DeviceToHost DeviceToDevice
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
//
cudaError_t cudaFree(void *devPtr);
// 错误码转错误消息
char* cudaGetErrorString(cudaError_t error);
~~~

- 除了kernel启动之外的CUDA调用都会返回错误枚举类型 cudaError_t。
- GPU的共享内存类似于CPU的缓存，可以由kernel直接控制。
- CUDA编程范式：
  - CPU、GPU分别申请内存空间
  - 把数据从CPU Mem 拷贝到 GPU Mem
  - 调用核函数，让GPU对GPU Mem中的数据进行计算
  - 把计算结果从GPU Mem 拷贝到 CPU Mem
  - CPU、GPU分别释放内存空间

~~~c++
const int N = 1024;

__global__ void sumArraysOnDevice(float *A, float *B, float *C){
    int idx = threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

void initialData(float *ip, int size){
    srand((unsigned int)time(NULL));
    for(int i = 0 ; i < size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv){
    size_t nBytes = N * sizeof(float);
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    // Allocate GPU Mem and CPU Mem
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);
    
    initialData(h_A, N);
    initialData(h_B, N);
    
    // CPU Mem to GPU Mem
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // call Kernel func && GPU Compute
    sumArraysOnDevice<<<1, N>>>(d_A, d_B, d_C);

    // GPU Mem to CPU Mem
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    // Free GPU Mem And CPU Mem
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
~~~

- 一个Kernel启动产生的所有线程统称一个grid。一个grid由多个Thread Block组成。Thread Block内包含一组Thread。
- blockIdx、threadIdx、gridDim、blockDim。

- 手动定义的dim3类型的网格、块变量仅主机端可见。uint3类型的内置预初始化的网格、块变量仅设备端可见，在kernel里能被访问到。
- 经验性地：**grid.x** 应该是 **(nElem+block.x -1) / block.x**

- __global__修饰的函数可以在两个端调用，在设备端执行。__device__修饰的函数，设备端调用，设备端执行。__host__修饰的函数，主机端调用，主机端执行。

- linux <sys/time.h>

- 并行线程的索引：
  - 线程索引、块索引
  - 矩阵中给定点的坐标
  - 全局线性内存中的偏移量


~~~c++
ix = threadIdx.x + blockIdx.x * blockDim.x;
iy = threadIdx.y + blockIdx.y * blockDim.y;
// 矩阵坐标(ix, iy) disp = idx
idx = iy * nx + ix;
~~~

- 设备信息获取：CUDA runtime API 和 NVIDIA-smi 命令行实用程序

~~~c++
cudaError_t cudaSetDevice(int dev);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int dev);
cudaError_t cudaGetDeviceCount(int *devCount);
cudaError_t cudaDeviceReset();
~~~



## 3.CUDA执行模型

### 3.1 概述

- CUDA采用SIMT(single instruction mutli thread)架构来管理和执行线程。每32个线程一组，称为一个线程束。
- 线程束中的所有线程同时执行相同的指令。线程有自己的PC和Reg，利用自身的数据执行当前指令。
- SIMT确保了线程束内线程的独立执行，可以编写独立的线程级并行代码、标量线程以及用于协调线程的数据并行代码。
  - SIMT的线程有自己的指令地址计数器、寄存器状态和独立的执行路径。这是和SIMD的区别。
- 线程块内线程束间的切换没有开销。因为SM已经把硬件资源分配到了所有常驻线程和块中。活跃的线程束数量受SM资源的限制。

#### 3.1.2 Fermi架构的硬件详情

- CUDA核心(core)：包括一个全流水线的整数算术逻辑单元ALU和一个浮点运算单元FPU，每个时钟周期执行一个整数或浮点数指令。
- GigaThread引擎：一个全局调度器，用来分配线程块到SM线程束调度器上。
- 二级缓存：被16个SM共享
- SM包括：执行单元core、调度线程束的调度器和调度单元、共享内存、寄存器文件和一级缓存。
  - 每个SM有16个load/store单元，允许每个时钟周期有16个线程计算源地址和目的地址。
  - SFU特殊功能单元，执行固有指令，如正弦、余弦、平方根和插值。每个SFU在每个时钟周期内的每个线程上执行一个固有指令。
  - 每个SM有两个线程束调度器和两个指令调度单元。
- 线程块被调度到SM时，块内线程被分为线程束。俩线程束调度器选择俩线程束，再把一个指令从线程束中发送到一个组上。
  - 组内有16个core、16个ld/st单元或4个sfu

#### 3.1.3 Kepler架构

- 在Fermi的基础上，增强了SM，如：加入了双精度单元。
- Kepler允许GPU动态启动新的网格。任一内核都能启动其他的内核，并管理核间需要的依赖关系，以正确执行附加工作。
  - 更容易创建和优化递归及数据相关的执行模式。

- Hyper-Q：多硬件任务队列，防止某个任务阻塞单个任务队列。

#### 3.1.4 配置文件驱动优化

- 性能分析工具：nvvp、nvprof
- CUDA性能分析中，事件 是可计算的活动，对应一个在内核执行期间被收集的硬件计数器。
- 指标 是内核的特征，由一个或多个事件计算得到。
  - 大多计数器基于SM，而不是GPU。
  - 性能分析可能需要获得所有相关的计数器，而单一的运行只能获得几个计数器。有些计数器的获得是互斥的。
  - GPU的执行是变化的，多次执行得到的计数器值可能不一样
- 内核性能的限制因素：
  - 存储带宽
  - 计算资源
  - 指令和内存延迟

### 3.2 线程束执行的本质

#### 3.2.1 线程束

- 线程束（warp~[wɔː(r)p]~)是SM的基本执行单元。
- 硬件角度来看，所有线程都被组织成一维的。
- threadIdx.x连续的线程被分配到线程束中。x是最内层的维度，z是最外层的维度。
- 线程块内的一维化：threadIdx.z * blockDim.y*blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
- 线程块会被划分为整数个线程束，如果不是整数倍，最后一个线程束会分配一些 非活跃态的线程。这些线程也占用硬件资源，如寄存器。

#### 3.2.2 线程束分化

- GPU支持传统C风格的显式控制流，如 if else，for和while。
- 线程束分化：**同一线程束中的线程 遇上 控制流的不同条件**，执行了不同的路径，就叫线程束分化。
- **线程束分化发生时，线程束会连续执行每一条分支路径，禁用不执行这一路径的线程。会明显导致性能下降。**
- 所以应该 **避免线程束分化**。线程块中线程束的分配是确定的，所以可以也应该避免线程束分化。

#### 3.2.3 资源分配

- 线程束的本地执行上下文资源：程序计数器PC、寄存器Register、共享内存。
- 由SM处理的每个线程束的上下文在整个线程束生命周期中是保存在芯片内的，所以上下文切换没有损失。
- **一个SM中同时存在的线程块和线程束数量，取决于SM可用的且内核需要的寄存器、共享内存数量。**
- **每个线程块的最大线程数：1024**
- 活跃的线程块：已分配计算资源的线程块
- 活跃的线程束：活跃线程块的线程束，包括：
  - 选定的线程束 执行
  - 阻塞的线程束 没有做好执行准备
  - 符合条件的线程束 做好执行准备但尚未执行
- 符合执行条件：32个core可用 且 当前指令的所有参数就绪

#### 3.2.4 延迟隐藏

- 指令延迟：指令发出和完成之间的时钟周期。
- 当每个时钟周期中所有 **线程束调度器 **都有一个符合条件的线程束时，可以保证，通过在其他常驻线程束中发布其他指令，以隐藏指令延迟。
- 指令：算术指令和内存指令。
  - 算术指令延迟：一个算术操作从开始到产生输出间的时间。10~20个周期
  - 内存指令延迟：发送出的加载或存储操作和数据到达目的地间的延迟。400~800个周期

- 延迟隐藏是针对线程束调度器来讲的，合理的活跃线程束数量：
  - Little's Law 延迟 * 吞吐量 = 所需线程束数量
- 提高并行的方法：
  - 指令级并行：一个线程中有很多独立的指令
  - 线程级并行：很多并发地符合条件的线程
- 计算所需并行的简单计算：SM核心数 * 算术指令延迟 = 每个SM至少所需的线程数。
  - 一个SM有128个core，算术指令延迟平均16个周期，那么一个SM需要线程数为2048？
  - 一个SM最多容纳48个线程束（1536个线程），Block最好是32*16=512？

- **延迟隐藏就是让线程束调度器别停。**

#### 3.2.5 占用率

- 核心占用率 = 活跃线程束数量 / 最大线程束数量
  - 唯一注重的是：SM中并发线程束的数量
- **CUDA工具包含了CUDA占用率计算器。**
- 内核使用的寄存器数量会显著影响常驻线程束数量，**通过nvcc手动设置 -maxrregcount 可以调整至推荐寄存器数**，以改善程序性能。
- 线程块大小的影响：
  - 小线程块：在达到每个SM的**线程束数**限制时，**硬件资源**未被充分利用。
  - 大线程块：每个SM中每个线程可用的硬件资源较少。

> 对于Kepler，从启动到完成在任何时候都必须小于或等于64个并发线程束的架构限度。 在任何周期中，选定的线程束数量都小于或等于4。

- 网格和线程块大小的准则：
  - 块线程数是线程束大小的倍数
  - 每个块至少128/256个线程
  - 根据内核资源需求调整块大小
  - **块的数量要远多于SM的数量，以达到足够的并行**
  - **通过实验得到最佳执行配置和资源使用情况**

#### 3.2.6 同步

- CUDA同步的两个级别：
  - 系统级：等待host和device完成所有工作
  - 块级：device执行过程中等待一个线程块中所有线程到达同一点

~~~C++
// 阻塞主机应用程序，直到所有的CUDA操作完成
cudaError_t cudaDeviceSynchronize(void);
// 块局部栅栏 在内核中标记同步点 同一线程块的所有线程等待所有其他线程都到达该同步点
// 同时会保证该点前影响的全局内存和共享内存的可见性 可以用于协调通信
__device__ void __syncthreads(void);
~~~

- GPU要保证 **以任意顺序执行块**，保证大规模并行GPU的可扩展性。所有不允许块间线程同步。

#### 3.2.7 可扩展性

- 可扩展性：加硬件就能保证性能提升。

### 3.3 并行性的表现

- 一个块的最内层维数应该是线程束大小的倍数。

- 需要在几个相关指标间寻找一个恰当的平衡来达到最佳的总体性能。网格/块启发式算法为性能调节提供了一个很好的起点。

### 3.4 避免分支分化

- 归约运算：在向量中执行满足交换律和结合律的运算。
- 向量组的相邻归约会造成1/2,3/4 ...的性能浪费。
- 通过重新组织每个线程操作的数组索引来强制 ID相邻（用一线程束）的线程执行求和操作，线程束分化就能被归约了。
  - 就是让工作的线程集中在相同线程束中。避免部分线程工作造成的线程束高度分化。

~~~c++
// 邻接
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
   
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        // 线程束全程都分化
        if(tid % (2*stride) == 0){
            idata[tid] += idata[tid + stride];
        }
        __synchthreads();
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}
// 邻接 + 调整计算线程
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n)    return;
   
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        // 线程处理的数据位置发生改变
        // 索引靠后的线程束什么都不干（不分化），靠前的线程束不分化
        // 直到后几轮 因为参与线程过少 导致线程束分化
        int index = 2 * stride * tid;
        if(index < blockDim.x)
            idata[index] += idata[index + stride];
        __synchthreads();
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}
// 交错
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
     unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx >= n)    return;
    // 同样要求头部线程执行归约操作 和上个算法的线程束分化程度一样
    // 这个的内存加载/存储模式更符合局部性 所以性能比上个算法更好
    for(int stride = blockDim.x / 2; stride > 0 ; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __synchthreads();
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}
~~~

### 3.5 循环展开

- 循环展开：一种优化循环的技术，通过减少分支出现频率和循环维护指令来优化。
- 任何的封闭循环都可以被减少循环次数，甚至完成删除。

- 循环展开因子：循环体的复制数量。
- 循环展开因子 * 迭代次数 = 原始循环迭代次数。
- 对于CUDA来说，循环展开的意义在于：减少指令消耗和增加更多的独立调度指令。以提高并行度。

~~~C++
// 交错 + 同时操作两个相邻数据块
// 线程块数减半 并行性down 内存局部性up
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    // 一般的线程块 每个线程块处理两个数据块，先归约成一个数据块，再做块内归约
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if(idx + blockDim.x < n)    g_idata[x] += g_idata[idx + blockDim.x];
    __syncthreads(); 


    for(int stride = blockDim.x / 2; stride > 0 ; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}
// 进一步展开 reduceUnrolling4、8 会有更好的性能
~~~

- 一个线程内有更多的独立内存加载/存储操作会产生更好的性能，因为内存延迟可以更好的隐藏起来了。

~~~c++
// 再优化最后几个循环 因为计算线程太少 造成的线程束分化问题
// 线程块数为最开始的1/8
__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    // 每个线程块处理八个数据块，先归约成一个数据块，再做块内归约
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if(idx + 7*blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[ix] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads(); 


    for(int stride = blockDim.x / 2; stride > 32 ; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}
~~~

~~~c++
// 完全展开
__global__ void reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    // 一般的线程块 每个线程块处理两个数据块，先归约成一个数据块，再做块内归约
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if(idx + 7*blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[ix] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads(); 
    // 完全展开
    if(blockDim.x >= 1024 && tid < 512)	idata[tid] += idata[tid+512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256)	idata[tid] += idata[tid+256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128)	idata[tid] += idata[tid+128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64)	idata[tid] += idata[tid+64];
    __syncthreads();
    
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}
~~~

- 模板函数

~~~C++
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if(idx + 7*blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2*blockDim.x];
        int a4 = g_idata[idx + 3*blockDim.x];
        int a5 = g_idata[idx + 4*blockDim.x];
        int a6 = g_idata[idx + 5*blockDim.x];
        int a7 = g_idata[idx + 6*blockDim.x];
        int a8 = g_idata[idx + 7*blockDim.x];
        g_idata[ix] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads(); 
    // 完全展开
    if(iBlockSize >= 1024 && tid < 512)	idata[tid] += idata[tid+512];
    __syncthreads();
    if(iBlockSize >= 512 && tid < 256)	idata[tid] += idata[tid+256];
    __syncthreads();
    if(iBlockSize >= 256 && tid < 128)	idata[tid] += idata[tid+128];
    __syncthreads();
    if(iBlockSize >= 128 && tid < 64)	idata[tid] += idata[tid+64];
    __syncthreads();
    
    if(tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if(tid == 0)
        g_odata[bid] = idata[tid];
}

// int main
switch(blockSize){
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x/8, block>>>(d_idata, d_odata, size);	break;
    case 512:
        reduceCompleteUnroll<512><<<grid.x/8, block>>>(d_idata, d_odata, size);	break;
    case 256:
        reduceCompleteUnroll<256><<<grid.x/8, block>>>(d_idata, d_odata, size);	break;
    case 128:
        reduceCompleteUnroll<128><<<grid.x/8, block>>>(d_idata, d_odata, size);	break;
    case 64:
        reduceCompleteUnroll<64><<<grid.x/8, block>>>(d_idata, d_odata, size);	break;
}
~~~

### 3.6 动态并行

- 动态并行 允许在GPU端直接创建和同步新的GPU内核。
- 动态并行中，内核执行分为父母和孩子。父网格、父线程块、父线程启动子网格、子线程块、子线程。子需要在父之前完成。
- 主机配置和启动父网格，父网格配置和启动子网格。

- 设备线程中的网格启动，在线程块间是可见的。这意味着线程可以和 （自己创建的）或（同线程块其他线程创建的）子网格同步。
- 线程块中，只有当所有线程创建的所有子网格完成后，线程块的执行才完成。
- 如果在所有子网格完成前，父线程块的所有线程都退出了，子网格上的隐式同步就会被触发。
- 父母启动一个子网格，父线程块与孩子显式同步后，孩子才能开始执行。
- 父网格和子网格共享相同的全局和常量内存存储，但有不同的局部内存和共享内存。
  - 前半句话：父线程的全局内存操作要对子网格可见。子网格完成时进行同步操作后，子网格的所有内存操作应该对父母可见。
  - 后半句话：共享内存对线程块私有，局部内存对线程私有。启动子网格时，向局部内存传递指针是无效的。


- 嵌套归约

~~~C++
// 交错递归
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int *idata = g_idata + bid * blockDim.x;
    int *odata = g_odata + bid;
    

    if(isize == 2 && tid == 0){
        odata[bid] = idata[tid] + idata[tid + stride];
        return;
    }

    int stride = isize>>1;
    if(stride > 1 && tid < stride){
        idata[tid] += idata[tid+stride];
    }
    __syncthreads();
    
    if(tid == 0){
        gpuRecursiveReduce<<<1, stride>>>(idata, odata, stride);
        // 是为了同步该块的所有子网格 但这里只有一个子网格 所以不同步也行
        cudaDeviceSynchronize();
    }
    // 然后 同步该块的所有线程
    __syncthreads(); 
}

// 交错递归 + 少网格 + 半数线程量
__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int iStride, const int iDim){
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int *idata = g_idata + bid * iDim;
    
    if(iStride == 1 && tid == 0){
        g_odata[bid] = idata[tid] + idata[tid + iStride];
        return;
    }
    // 只开数据量一半的线程数，这样就不会浪费了
    idata[tid] += idata[tid+iStride];
    
    if(tid == 0 && bid == 0){
        // 减少开启子网格的浪费 开一个合适大小的
        gpuRecursiveReduce2<<<gridDim.x, iStride/2>>>(g_idata, g_odata, iStride/2, iDim);
    }
}
~~~



## 4.全局内存

### 4.1 概述

- 一个核函数中的线程都有自己私有的本地内存、寄存器。
- 一个线程块有自己的共享内存，块内线程可见，内容持续线程块的整个生命周期。
- 全局内存、常量内存和纹理内存是所有线程可见的，对一个application来说，生命周期相同。
- 常量内存空间和纹理内存空间是只读的。
- 纹理内存为各种数据布局提供了不同的寻址模式和滤波模式。

#### 4.1.1 CUDA内存模型

<img src="https://raw.githubusercontent.com/JiXuanYu0823/ReadingNotes/main/assets/Streaming%20Multiprocessors%E6%9E%B6%E6%9E%84.png" alt="Streaming Multiprocessors架构" style="zoom: 80%;" />

##### 4.1.1.1 寄存器

- 核函数声明的未被修饰符修饰的自变量，通常存在寄存器里。数组索引如果是常量且编译时可确定，则数组也在寄存器里。
- 寄存器变量和核函数生命周期相同，私有。
  - Kepler每个线程最多有255个寄存器。
- 超额度的寄存器会由本地内存替代。寄存器溢出会影响性能。nvcc编译器采用启发式策略来最小化寄存器的使用，来避免溢出。

##### 4.1.1.2 本地内存

- 可能存在本地内存中的变量：
  - 编译时使用未知索引的本地数组
  - 可能会占用大量寄存器的较大本地结构体或数组
  - 任何不满足核函数寄存器限定条件的变量

- **本地内存和全局内存用的是同一块存储区域**，高延迟低带宽。本地内存数据可以由每个SM的一级缓存和每个设备的二级缓存进行缓存。

##### 4.1.1.3 共享内存

- __shared__修饰的变量存放在共享内存中。
- 共享内存是片上内存，比全局（本地）内存带宽高延迟低。类似于CPU一级缓存，但它是可编程的。
- 过度使用共享内存也会影响活跃线程束的数量。
- 共享内存数据持续线程块的生命周期。__syncthreads()调用是同步使用共享内存所必须的。

- **SM的一级缓存和共享内存都使用64KB的片上内存。**

##### 4.1.1.4 常量内存

- **常量内存驻留在设备内存中。**并在每个SM专用的常量缓存中缓存。
- 常量变量用 __constant__ 来修饰。必须在全局空间内和所有核函数之外声明。

- 核函数只能读常量内存，因此，常量内存只能在主机端初始化：

~~~c++
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);	// 这个函数绝大部分情况下是同步的
~~~

- 每从常量内存中读取一次数据，都会广播给线程束里的所有线程。这种情况下，常量内存表现最好。

##### 4.1.1.5 纹理内存

- **纹理内存驻留在设备内存中。**并在每个SM的只读缓存中缓存。
- 纹理内存是一种通过指定的只读缓存访问的全局内存。纹理内存是对**二维空间局部性的优化**，所以线程束里用纹理内存访问二维数据性能最好。只读缓存包括**硬件滤波**功能，可以将浮点插入作为读过程的一部分来执行。

##### 4.1.1.6 全局内存

- 全局内存贯穿整个应用程序的生命周期。
- __device__修饰符可以静态的声明一个变量。cudaMalloc和cudaFree都是在操作全局内存。
- 多个块间线程并发访问全局内存可能会出问题。

- 可以通过32/64/128字节的内存事务进行全局内存访问。
  - 内存事务必须自然对齐，即首地址必须是这些字节的倍数。

##### 4.1.1.7 GPU缓存

- GPU的四种缓存：一级缓存、二级缓存、只读常量缓存、只读纹理缓存。
- 每个SM都有一个一级缓存，所有SM共享一个二级缓存。一级缓存和二级缓存都用来存储本地内存和全局内存中的数据。
- GPU中只有内存加载操作能被缓存，内存存储操作不能。
- 每个SM都有一个只读常量缓存和只读纹理缓存。
  - **换句话说，所有的（SM）片外内存，都在SM里有缓存。**

##### 4.1.1.8 静态全局内存

~~~c++
__device__ float devData;   // gpu的全局内存变量 devData是在主机端的设备变量，是个标识符，不是变量地址
__global__ void checkGlobalVariable(){
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;	// device直接访问
}
// host不能直接访问gpu变量 cudaMemcpyToSymbol是CUDA runtime的API，主机端是借用gpu硬件访问的
int main(void){
    float value = 3.14f;
    // 复制到Device的全局或常量内存中
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    checkGlobalVariable<<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
}
// 这里cudaMemcpy是行不通的 因为devData不是实际GPU内存地址
cudaMemcpy(&devData, &value, sizeof(float), cudaMemcpyHostToDevice);
// 通过devData标识，获取实际物理地址
float *dptr = NULL;
cudaGetSymbolAddress((void**)&dptr, devData);
cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice);
~~~

### 4.2 内存管理

- CUDA提供了在主机端准备设备内存并显式传输数据的函数。就是cudaMemcpy、cudaMemset、cudaFree这些。
  - 这里，再啰嗦一遍，这章说的都是全局内存。
- 设备内存的分配和释放成本较高，应该重复利用设备内存。
- CUDA编程基本原则：**尽量减少主机和设备间的传输。**

- Host虚拟内存机制导致GPU无法安全地访问可分页内存空间。所以，实际的数据传输过程是：CUDA驱动程序首先分配临时页面固定的主机内存，将host source data复制到 **固定内存** 中，然后从固定内存传输到device Mem。

~~~C++
cudaError_t cudaMallocHost(void **devPtr, size_t count);	// 直接分配固定主机内存
cudaError_t cudaFreeHost(void *ptr);
~~~

- 固定内存的分配和释放成本很高，且会影响到虚拟内存的容量。所以，当传输数据量大时，才建议使用。

- **零拷贝内存**：固定内存，该内存会映射到设备地址空间中。
  - 主机和设备都可以访问。
  - 使用零拷贝内存时，必须同步主机和设备间的内存访问。
  - 零拷贝同样要走PCIe复制，只不过不是显式的赋值。

~~~C++
// 分配count字节的主机内存，该内存是页面锁定且设备可访问的。
// 用cudaFreeHost释放
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
// flags用来配置该内存的特殊属性
// 行为同cudaMallocHost
cudaHostAllocDefault
// 返回的固定内存能被所有CUDA上下文使用，而不是仅执行内存分配的那个
cudaHostAllocPortable
// 该配置 对于（mapped pined memory 或 hostToDevice transfer方式）主机写设备读 的buffers来说是个好选择
cudaHostAllocWriteCombined
// 零拷贝内存的标志 分配的主机内存被映射到设备地址空间 
cudaHostAllocMapped
    
// 通过hostPointer 获取 mapped pined memory 的设备指针 devicePointer
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
~~~

- **统一虚拟寻址(Unified Virtual Addressing)** ：通过UVA，cudaHostAlloc分配的固定主机内存具有相同的主机和设备指针。免去了cudaHostGetDevicePointer这个函数的转换。
  - UVA为系统的所有处理器提供了一个单一的虚拟内存地址空间。

- **统一内存寻址**：统一内存中创建了一个**托管内存池**，CPU和GPU可以用相同的指针在池中分配的空间上进行访问。底层系统自动在主机和设备间进行数据传输。
  - 所有在设备上有效的CUDA操作，均适用于托管内存。
  - 托管内存可以被静态或动态分配
  - 在文件范围或全局范围内，由 __managed__ 注释指明该设备变量为托管变量。托管变量可以被主机或设备代码直接引用。

~~~c++
// 动态分配托管内存
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
// cuda6.0中 设备端不能调用该函数，只能主机端动态声明或全局静态声明
~~~

### 4.3 内存访问模式

- 多数设备端数据访问都是从全局内存开始的，容易受内存带宽影响。

- 核函数的内存请求通常是在DRAM设备和片上内存间以128/32字节内存事务来实现的。
  - 所有对global mem的访问都会通过二级缓存，也有许多会通过一级缓存。
  - 两个缓存都用到就是128字节内存事务
  - 只用二级缓存就是32字节内存事务
  - 缓存读的情况下，SM执行的物理加载操作必须在128个字节的界限上对齐。
  - 非缓存读，则32个字节对齐。
  
- **对齐内存访问**：内存事务的首地址是事务服务**缓存粒度的偶数倍**时，就叫对齐内存访问。
  - 非对齐加载会造成带宽浪费
- **合并内存访问**：当一个线程束的32个线程访问一个连续的内存块时，就会出现合并内存访问。

- 对齐合并访问就是理想状态。对齐合并只需要1个128字节内存事务时，非对齐未合并的内存访问可能需要3个128字节。

#### 4.3.1 全局内存读取

- SM中的数据通过以下3种缓存路径传输：
  - 一级和二级缓存（默认）
  - 常量缓存（需程序显示指明）
  - 只读缓存（需程序显示指明）

~~~shell
-Xptxas -dlcm=cg	# 禁用一级缓存
-Xptxas -dlcm=ca	# 启用一级缓存
~~~

- 缓存加载：经过一级缓存，在粒度为128字节的一级缓存行上有设备内存事务进行传输。
  - 分为对齐、非对齐和合并、非合并
- **GPU一级缓存专为空间局部性设计。频繁访问一个一级缓存的内存位置不会增加数据留在缓存中的概率。**

- 没有缓存的加载：不经过一级缓存，在内存段粒度（32字节）上而非缓存池粒度（128字节）上执行。
  - 因为粒度小，所以可以很好的改善非对齐和非合并的情况
- 只读缓存最初是预留给纹理内存加载的，也可以替代一级缓存，支持全局内存的加载。加载粒度为32字节。
- 指导只读缓存读取：
  - 使用__ldg函数
  - 在间接引用的指针上使用修饰符


~~~C++
__global__ void cpyKernel(int *out, int *in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = __ldg(&in[idx]);
}
// __restrict__帮助编译器识别无别名指针，通过只读缓存指导该指针的加载。
__global__ void cpyKernel(int * __restrict__ out, const int * __restrict__ in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}
~~~

#### 4.3.2 全局内存写入

- 一级缓存不能在Fermi和Kepler架构上进行存储操作，发送到设备内存前，只经过二级缓存。执行粒度为32字节。
- 内存事务可以同时被分为一段、两段和四段。

#### 4.3.3 Struct of Array 和 Array of Struct

- GPU更偏向于使用Struct of Array，因为更好的合并内存访问性。

#### 4.3.4 性能调整

- 优化设备内存带宽利用率的两个目标：
  - 对齐及合并内存访问，以减少带宽的浪费
  - 足够的并发内存操作，以隐藏内存延迟
- 对于IO密集型核函数，内存访问并行有很高的优先级。

### 4.4 核函数可达到的带宽

- 有效带宽(GB/s) = （读字节数+写字节数）* 10^-9^ / 运行时间。
- 网格启动时，线程块会按bid的顺序分配给SM。但由于线程块完成的速度和顺序不确定，所以，随着内核线程的执行，bid会变得不太连续。
  - bid = blockIdx.y * gridDim.x + blockIdx.x。
- 但是可以主动调整并行bid的内存访问位置，以避免分区冲突。
  - 这是block level的角度进行的优化

- **分区冲突：多个内存访问的地址在同一分区，那么这些访问操作将被串行执行**。
  - 发给全局内存的请求由DRAM分区完成。
  - **设备内存中连续256字节区域被分配到连续的分区**。

### 4.5 使用统一内存的矩阵加法

- 托管内存的使用可能会花费更多时间。因为它会在主机和设备间来回迁移数据。
  - 比如内存空间最开始在GPU上分配，然后迁移到CPU上初始化。

- GPU页面错误机制：如果在GPU上运行的内核访问一个不在其内存中的页面，那么它会出错，允许该页面按需自动迁移到GPU内存。
- 显存超量分配：具有计算能力6.x的设备扩展了寻址模式以支持49位虚拟寻址。它的大小足以覆盖现代CPU的48位虚拟地址空间，以及GPU显存。巨大的虚拟地址空间和页面错误功能使应用程序能够访问整个系统虚拟内存，而不受任何一个处理器的物理内存大小的限制。只要有足够的系统内存可供分配，cudaMallocManaged就不会耗尽内存。
- 具有计算能力6.x设备的系统上的托管分配对所有GPU都可见，可以按需迁移到任何处理器。

~~~shell
export CUDA_VISIBLE_DEVICES=0	# 限制设备0对应用程序可见 这样托管内存便可以只分配在一个设备上。
~~~

- 当CPU需要访问当前驻留在GPU中的托管内存时，统一内存使用CPU页面故障来触发设备到主机的数据传输。



## 5.共享内存和常量内存

### 5.1 CUDA共享内存概述

- 全局内存的所有加载/存储请求都要经过二级缓存。
  
- 共享内存（SMEM）是片上内存，相较于板载的全局内存，它拥有较低的延迟（20~30倍），较高的带宽（10倍）。是可编程的。用途有：
  - 块内线程通信的通道
  - 全局内存数据的可编程管理缓存
  - 高速暂存存储器，用于转换数据以优化全局内存访问模式

- 线程块开始执行时，会分配给它一定数量的共享内存。该地址空间被块内线程共享，与线程块生命周期相同。
- 最好情况下，线程束对共享内存的访问在一个事务中完成。最坏情况下，在32个事务中完成。
- 多个线程访问共享内存同一个字时，由一个线程读取该字后，多播给其他线程。
- 共享内存被SM中的所有常驻线程划分，所以，一个核函数使用的共享内存越多，并发活跃的线程块越少。
- 共享内存既可以被声明在核函数的局部，也可以被声明在CUDA源文件的全局。
  - 核函数内声明，就是局部的。文件的任何核函数外声明就是全局的。
- __shared__ 修饰符可以声明共享内存变量。extern修饰符可以声明一个编译时未知大小的共享内存一维数组。

~~~C++
extern __shared__ int tile[];	// 动态共享内存必须被声明为一个未定大小的一维数组
// 这样的话，每个核函数被调用时，都需要动态分配共享内存
kernel<<<grid, block, isize*sizeof(int)>>>(...)
~~~

- **共享内存被分为32个同样大小的内存模式，被称为存储体，可以被同时访问。**
- 共享内存是一个一维地址空间。
- 如果线程束发布共享内存加载或存储操作，且每个存储体上只访问不多于一个的内存地址，那么该操作由一个内存事务来完成。否则，多个。
- 存储体冲突：多个相异地址请求落在相同的内存存储体上。硬件会将存储体冲突的请求分割到尽可能多的独立无冲突事务中，这会导致有效带宽降低，诱因因素的数量等同于所需独立内存事务的数量。
  - 这会导致请求被重复执行。

- 共享内存请求的三种访问模式：
  - 并行访问：多个地址访问多个存储体
  - 串行访问：多个地址访问同一存储体
  - 广播访问：单一地址读取单一存储体
- 存储体索引：对于Fermi设备，存储体宽度是32位（4字节）且有32个存储体，每个存储体每两个时钟周期都有32位带宽。这样的话，四个相邻对齐字节地址对应的存储体是一个。每隔32个这样的四字节地址就又回到同一存储体。
  - 存储体索引 = （字节地址 / 4 ）% 32；4是存储体字节宽度，32是存储体个数。
  - 为了最大化并行访问，邻近的字被分到不同存储体。

- **对于同一线程束，并行读同一地址，被读取的字会被广播到请求线程。并行写的字，只会被一个线程写入，该线程不是确定的**。
- 对于Kepler设备，共享内存有32个存储体，64/32位模式，每个存储体每个时钟周期都有64位带宽。
  - 如果是64位模式，线程束中两个线程访问同一64位字的任何子字，发出的共享内存请求都不会造成存储体冲突，因为读取的粒度64bits。
  - 如果是32位模式，同一存储体中访问两个32位字并不意味着重操作。

- 内存填充：如果某个存储体的不同地址的并发访问比较严重，那么可以在每N个（存储体数量）元素后填充一个字，以改善该情况。

~~~C++
cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig *pConfig);
// 查看是 cudaSharedMemBankSizeFourByte还是cudaSharedMemBankSizeEightByte
cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config);
~~~

- 共享内存和一级缓存共享64KB的片上内存。可以按照设备进行配置，也可以按照核函数进行配置。
- 按设备配置：
  - 对于一级缓存处理寄存器溢出的架构，应该prefer一级缓存。
  - 对于本地内存处理寄存器溢出的架构，本地内存也可能用一级缓存作为缓存，也可以prefer一级缓存。
- 按核函数配置：

~~~C++
// 以与最新prefer不同的prefer启动内核，可能导致隐式设备同步
// 每个核函数只需要调用一次该函数，片上内存配置不需要在每个核函数启动时重置。
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCacheca cheConfig);
~~~

- GPU的缓存数据删除行为使用了不同的启发式算法，更加频繁和不可预知。使用GPU共享内存不仅可以显式管理数据，还能保证SM的局部性。

- 同步的两种机制：Barriers和Memory fences。
  - Barrier保证所有线程等待其他线程到达barrier point。
  - Memory fences保证所有线程等待全部内存修改对其余调用线程可见时，才继续执行。

- 弱排序内存模式：内存访问顺序不一定和它们在程序中出现的顺序一致。
  - 为了确保程序按设想的方式运行，需要在代码中插入Memory fences和Barriers。
- 用__syncthreads()的时候，要求条件要对所有线程进行同等评估，否则会很糟糕。
  - 就是要确保 会到达同一个barrier
- 连续的内核启动，要求等待之前的内核启动完成。在同步点，通过将一个核函数划分为多个内核启动，可以达到跨线程块全局同步的效果。

- Memory fence前的写，对Memory fence后的线程可见。

~~~C++
// 创建线程块级memory fence
void __threadfence_block();
// 创建网格级memory fence
void __threadfence();
// 创建系统级memory fence	通过挂起调用线程，保证所有内存设备所有写操作的HostAndDevice线程可见
void __threadfence_system();
// volatile修饰符也可以保证写可见
~~~

### 5.2 共享内存的数据布局

- 线程束是threadIdx.x连续的几个线程，要想并行访问存储体，可以int tile[threadIdx.y] [threadIdx.x]，这样会访问不同的存储体。
- 方块共享内存：
  - 动态声明共享内存会增加少量消耗
  - 记得有内存填充这个选项
- 矩形共享内存：
  - 实际编码过程中，需要考虑线程块内的线程索引 对应 共享内存和全局内存数据的一维和二维索引 的关系。
  - 数据位置变换带来的索引转换。


- 共享内存的意义：
  - 缓存片上数据，减少全局内存访问量
  - 改变数据访问方式，避免非合并全局内存访问

### 5.3 常量内存

- 常量内存用于只读数据和统一访问线程束中线程的数据。
  - 常量内存对于内核代码而言是只读的，对主机而言既可读又可写。
- 常量内存位于设备的DRAM上，且有一个专用的片上缓存。每个SM**常量缓存**大小的限制是64KB。
- 常量内存的最优访问模式是：线程束中所有线程访问该内存的相同位置。如果访问不同位置，则串行执行。
- __constant__ 在全局作用域下必须用它来声明常量变量。
- 常量内存的生命周期和应用程序的生存期相同，对网格内所有线程可见，且通过运行时函数对主机可访问。
- 因为设备只能读取常量内存，所以值必须用以下运行时函数进行初始化：

~~~C++
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, cudaMemcpyKind kind);
// Kind默认是cudaMemcpyHostToDevice
// 将src数据复制到symbol指定的常量内存中
~~~

- GPU纹理流水线作为独立的**只读缓存**，来存储global mem的数据。它带有从标准全局内存读取的独立内存带宽。只有48KB。
- 只读缓存更适合于分散读取。粒度为32B。
- 通过只读缓存访问全局内存时，要向编译器指出在内核的持续时间里，数据是只读的。
  - 内部函数__ldg：用于替代标准指针解引用，并强制通过只读缓存加载数据。
    - 在只读缓存需要更多显式控制或编译器不太好判断只读缓存的使用是否安全时，__ldg更好。

  - const __restritct__ ：表明通过只读缓存访问


- **洗牌指令(shuffle instruction)**：只要两个线程在相同的线程束中，就允许一个线程直接读取另一个线程的寄存器。

- 束内线程(lane)：一个束内线程指线程束内的单一线程，每个束内线程都有唯一标识(lane index)[0, 31]。
  - laneID = threadIdx.x % 32
  - warpID = threadIdx.x / 32
- 两组、每组四个洗牌指令，分别用于整型和浮点型变量。

~~~C++
// 使线程束中每个线程都可以从一个特定线程中获得某个值，每个线程都会有4个字节的数据移动
// var是操作的值，width是[2, 32]间的2的幂值， srcLane确定了束（段）内线程索引
// width是默认的warpSize即32时，洗牌指令跨整个线程束来执行，srcLane指定源线程的束内线程索引
// width是其他时，每个段独立洗牌，洗牌操作的ID和束内线程ID不一定一样，计算：shuffleID = threadIdx.x % width
int __shfl(int var, int srcLane, int width=warpSize);
// 0~32里 0~15从线程3接收x，16~31从线程19接收x
int y = __shfl(x, 3, 16);
// 广播
int y = __shfl(x, 2);

// laneID线程 将数据复制到 laneID+delta线程中 没有首尾相联且只考虑本线程束（段）
int __shfl_up(int var, unsigned int delta, int width=warpSize);
// laneID线程 将数据复制到 laneID-delta线程中 没有首尾相联且只考虑本线程束（段）
int __shfl_down(int var, unsigned int delta, int width=warpSize);
// laneID线程 将数据复制到 laneID xor laneMask线程中 没有首尾相联且只考虑本线程束（段）
int __shfl_xor(int var, int laneMask, int width=warpSize);

// 浮点洗牌函数用的是浮点型的var参数
~~~

## 6.流和并发

### 6.1 流和事件

- **CUDA流是一系列异步的CUDA操作**，这些操作按主机代码确定的顺序在设备上执行。流可以封装操作、保持操作的顺序。流中的操作执行对于主机来说是异步的。CUDA运行时来决定何时可以在设备上执行操作。程序员的任务就是保证异步操作在结果被使用前可以完成。
  - **涉及流的操作 都是 主机异步的。**

- 在同一个流中的操作有严格的执行顺序，在不同CUDA流中的操作则没有执行顺序的限制。使用多个流同时启动多个内核，可以实现网格级并发。

- 将计算和IO调度到不同的流上，可以重叠操作，隐藏延迟。
- 流在CUDA的API调度粒度上可实现流水线或双缓冲技术。
- 对于CUDA的API，一般分同步和异步。同步函数会阻塞主机端线程，直至完成。异步函数在调用后，会立即将控制权还给主机。
- 流分为：显式声明的流（非空流）和隐式声明的流（空流）。
  - 没有显式指定一个流时，内核启动和数据传输默认使用空流。

- 基于流的异步内核启动和数据传输支持以下粗粒度并发：
  - 重叠主机计算和设备计算
  - 重叠主机计算和DH数据传输
  - 重叠DH数据传输和设备计算
  - 并发设备计算

- 异步数据传输
  - 必须使用固定主机内存：cudaMallocHost和cudaHostAlloc。（现在好像不是这样了）

~~~C++
// 异步数据传输 默认情况下使用默认流。
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0);
// 非空流创建
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
// 非空流释放 释放操作是异步的，只有当工作完成时，才会进行实际的释放操作
cudaError_t cudaStreamDestroy(cudaStream_t stream);
// 异步API可能会从先前启动的异步操作中返回错误代码，因此返回错误的API不一定是产生错误的调用

// 非默认流启动内核
kernel_name<<<grid, block, sharedMemSize, stream>>>(arg list);

// 流操作完成检测
// 阻塞主机直至流操作完成
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
// 检测但不阻塞 cudaSuccess or cudaErrorNotReady
cudaError_t cudaStreamQuery(cudaStream_t stream);

// 优先级分配 高优先级的网格队列可以优先占有低优先级流已经执行的工作，优先级只会对内核计算产生影响，不影响数据传输。
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
// 优先级范围查询
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPri, int greatestPri);
~~~

- 并发内核的最大数量依赖设备。Fermi支持16路，Kepler支持32路。Kepler的Hyper-Q技术，使用了32个硬件工作队列，只有流超过32个时，才会共享硬件队列。

- CUDA事件本质上是CUDA流的标记，与流内特定点关联。事件的功能：
  - 同步流的执行
  - 监控设备的进展
- 事件：

~~~C++
cudaEvent_t event;
cudaError_t cudaEventCreate(cudaEvent_t* event);
// 异步 实际事件标记完成时自动释放与该事件有关的资源
cduaError_t cudaEvenetDestory(cudaEvent_t event);

// 事件排队进入CUDA流
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0);
// 已经排队进入流的事件可用于等待或测试在指定流中先前操作的完成情况。
// 等待一个事件会阻塞主机线程的调用 和cudaStreamSynchronize区别在于，该函数允许主机等待流执行的中间点
cudaError_t cudaEventSynchronize(cudaEvent_t event);
//
cudaError_t cudaEventQuery(cudaEvent_t event);
// 两事件间CUDA运行时间 事件启动和停止不必在同一个流中 cudaEventRecord也是异步的，所以可能标记时间会延后
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop);
~~~

~~~c++
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);

float time;
cudaEventElapsedTime(&time, start, stop);

cudaEventDestory(start);
cudaEventDestory(stop);
~~~

- 对于主机来说：默认流/空流是同步流，大部分空流操作会导致主机阻塞（除内核启动）。非空流是异步流，所有操作都不阻塞主机执行。
- **非空流也可以进一步分为：阻塞流和非阻塞流。非空流的操作可以被空流中的操作所阻塞**，尽管非空流是主机非阻塞的。空流可以阻塞非空阻塞流的操作，非空非阻塞流不会阻塞空流的操作。
- cudaStreamCreate函数创建的流是阻塞流，需要等到空流先前的操作执行结束。
  - 操作被发布到空流前，CUDA上下文会等待所有先前的操作发布到所有阻塞流中。
  - 任何发布到阻塞流中的操作，会被挂起等待，直到空流先前的操作执行结束。

~~~C++
// 定义关于空流的非空流行为 创建非空阻塞和非空非阻塞流
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
// cudaStreamDefault: blocking;	cudaStreamNonBlocking: non-blocking;
~~~

- 隐式同步：如cudaMemcpy，可能会带来意外的性能下降和不必要的阻塞。
  - 锁页主机内存分配 cudaMallocHost和cudaHostAlloc
  - 设备内存分配 cudamalloc
  - 设备内存初始化 cudamemset
  - 同设备两地址间的内存复制 cudaMemcpy D2D
  - 一级缓存、共享内存配置修改

~~~C++
// 使指定流等待指定事件 用来搞定流间依赖 跨流同步
// 调用cudaStreamWaitEvent后，在执行流中排队的任何操作之前，cudaStreamWaitEvent会导致指定的流等待指定的事件。
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event);
~~~

- 可配置事件

~~~C++
// 定制事件性能和行为
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
// cudaEventDefault

/**
cudaEventBlockingSync 说明该event会阻塞host。cudaEventSynchronize默认行为是使用CPU时钟来固定的查询event状态。使用cudaEventBlockingSync，调用线程会进入休眠，将控制权交给其他线程或者进程，直到event完成为止。这样会导致少量的CPU时钟浪费，但也会增加event完成和唤醒线程的之间的时间消耗。
**/

// cudaEventDisableTiming 不需要记录时序数据，创建的事件仅用来同步
// cudaEventInterprocess 事件可能用于进程间事件
~~~

### 6.2 并发内核执行

- 32个硬件工作队列+同流依赖，限制了内核并发量。
- 深度优先调度、广度优先调度、openmp编译选项。
- cudaStreamWaitEvent可以用作创建流间依赖关系

### 6.3 重叠内核执行和数据传输

- GPU有俩复制引擎队列，一个用来I，一个用来O。所以，最多重叠两个数据传输。
- 重叠IO和计算的时候，有两种情况：
  - 内核使用数据A，则A的传输必须在内核启动前，且位于相同的流中。
  - 内核不使用数据A，则内核执行和数据传输可以位于不同流中。

### 6.4 流回调

- 一旦流回调之前的所有流操作全部完成，流回调指定的主机函数将被CUDA运行时所调用。主机函数由应用程序提供，允许任意主机端逻辑插入到CUDA流中。流回调是另一种CPU和GPU同步的方法。
  - 回调函数会阻塞地在主机端执行，执行结束继续流操作。

~~~C++
// 注册流回调函数 flags没用的时候必须置0
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, 
                                  void *userData, unsigned int flags);
~~~

- 回调函数的限制：
  - 从回调函数中不能调用CUDA的API函数
  - 在回调函数中不可以执行同步
- 回调函数指定的数据，需要分配到固定内存里，防止回调发生时，页面切换带来错误数据。



## 7.调整指令级原语

### 7.1 CUDA指令概述

- 显著影响CUDA内核生成指令的3大因素：**浮点运算**、**内置和标准函数**、**原子操作**。
- 浮点运算：单双精度区别、浮点数粒度问题等。
- 标准函数：用于支持 **访问并标准化主机和设备** 的操作。
  - 包括math.h和单指令运算
- 内置函数：只能对设备代码进行访问。
  - 内置函数的编译指令优化更强大和专业。
  - 比标准函数更快，但精度低。
- 原子操作：一条原子操作指令用来执行一个数学运算，独立不可间断。原子操作指令在两个竞争线程共享的内存空间操作时，会有定义好的行为。
  - 多数原子函数是二进制函数，能够在两个操作数上进行操作。
  - 原子运算函数：算术运算、按位运算和替换函数。
    - 算术运算：在目标内存位置上 加、减、min、max、incr、decr。
    - 按位运算：在目标内存位置上 and、or、xor。
    - 替换函数：新值替换目标内存位置上的旧值，并返回旧值。
  - 原子操作的使用可能会严重降低性能

~~~C++
int atomicAdd(int *M, int V);
// 无条件替换
int atomicExch(int *M, int V); 	
// 有条件替换 Compare And Swap 比较时内存值*address 旧内存值compare 欲替换成val 如果*address==compare，替换。否则，不替换。
int atomicCAS (int *address, int compare, int val);	
~~~

### 7.2 程序优化指令

- 双精度存储空间是单精度的两倍，所以在数据传输过程中，不同精度的传输时间也是两倍的区别，设备端计算时间可能有优化。
- 单精度浮点数的声明必须谨慎，因为nvcc编译器也默认自动转double。
  - 所以，类似 pi=3.1415926f; 这种定义，忽略f就容易被编译器转double。
- GPU可能存在的精度损失是CUDA编译器强制浮点数值优化导致的。

- nvcc --ptx 能让编译器在PTX(Parallel Thread eXecution)和ISA中生成程序的中间表达式。
  - 可视的内置函数的指令条数远少于标准函数，所以性能差距还是蛮大的。
- 可以通过**编译器标志、内部和标准函数的调用**来控制编译器优化。

~~~shell
nvcc --ftz=true --prec-div=false --prec-sqrt=false --fmad=true --use_fast_math ...
# 将所有单精度非正规浮点数置零、禁止提高单精度除和倒数的精度、禁止执行高精度平方根函数、允许乘加融合、用等价的内部函数替换所有标准函数
~~~

- nvcc --fmad 用于启动和关闭FMAD指令（用于混合乘加、优化性能）。
- 单双精度乘法内部函数__fmul、dmul 可以阻止编译器将乘法优化为乘加优化的一部分。
  - MAD是乘加运算。
  - 理论上，可以有选择地应用__dmul这俩函数，并开启--fmad，来兼顾性能和精度。

~~~C++
__global__ void foo(float *ptr){
    // __f可能是内部函数 _rn是向偶舍入 默认的
    ptr = __fmul_rn(*ptr, *ptr) + *ptr;
}
/* IEEE 754 round-to-nearest-even. 二进制的向偶舍入,舍入的值保证最靠近原浮点数值，如果舍入为中间值，即舍还是入距离相等，那么按其最末尾一位是奇数，则入，如果为偶数，则舍. 最末尾y为需要保留的最后一位。*/
~~~

- 用atomicCAS实现所有线程都成功add的原子加

~~~C++
__device__ int myAtomicAdd(int *address, int incr){
    int guess = *address;
    int oldV = atomicCAS(address, guess, guess+incr);
    while(oldV != guess){
        guess = oldV;
        oldV = atomicCAS(address, guess, guess+incr);
    }
    return oldV;
}
~~~

- 原子操作成本高的可能原因：
  - 全局内存和共享内存的原子操作需要保证，对所有线程立即可见。可能需要越过缓存，直接读写内存。
  - 共享内存存储体冲突会导致不断地重试。
  - 线程束对同一内存地址的访问，会导致串行线程执行
- 原子函数不支持双精度浮点数（6.x之前）。可以通过位运算来搞定。

~~~C++
__device__ float myAtomicAdd(float *address, float incr){
    float *currentVal = *address;
    unsigned int *typedAddress = (unsigned int*)address;
    
    unsigned int guess = __float2uint_rn(currentVal);
    unsigned int newV = __float2uint_rn(currentVal + incr);
    int oldV = atomic(typedAddress, guess, newV);
    while(oldV != guess){
        guess = oldV;
        oldV = atomic(typedAddress, guess, __float2uint_rn(__uint2float(currentVal) + incr));
    }
    return __uint2float_rn(oldValue);                    
}
~~~



## 8.GPU加速库和OpenACC

- GPU加速库：CUFFT、CUBLAS、CUSPARSE、Librn(math.h)、CURAND、NPP、Thrust。
- OpenACC：用编译指令 注释 H和D端用于减荷的代码和数据区域。编译器会优化它们。

### 8.1 CUDA库概述

| 库名              | 作用域             |
| ----------------- | ------------------ |
| cuFFT             | 快速傅里叶变换     |
| cuBLAS            | 线性代数（BLAS库） |
| CULA Tools        | 线性代数           |
| MAGMA             | 新一代线性代数     |
| cuSPARSE          | 稀疏线性代数       |
| cuRAND            | 随机数生成         |
| NPP               | 图像和信号处理     |
| CUDA Math Library | 数学运算           |
| Thrust            | 并行算法和数据结构 |
| Paralution        | 稀疏迭代方法       |

- 通用CUDA库工作流：
  - 为库操作创建特定库句柄来管理上下文信息
  - 为库函数的输入输出分配设备内存
  - 将输入格式转换为函数库支持的格式
  - 输入填入内存
  - 配置要执行的库函数
  - 执行带有库函数的计算
  - 取回设备内存的计算结果
  - 将计算结果转换为原始格式
  - 释放CUDA资源

- 创建库句柄：
  - 分配句柄内存、初始化。
  - 句柄是存在主机上的、包含库函数可能访问信息的、对程序员不透明（可见）的对象。

- 输入数据传输到设备内存：
  - 多数情况下用的是库函数

- 配置函数库：
  - 传参、配置库句柄、管理数据对象
- 释放CUDA资源：
  - 最好能重用设备内存、库句柄和CUDA流等资源

### 8.2 cuSPARSE

- COO(Coordinate list)：坐标稀疏矩阵格式，存储非零元素的行坐标、列坐标和元素值。
- CSR(Compress sparse row)：压缩稀疏行，存储非零元素的列坐标、每行首位非零值在压缩存储中的偏移量和元素值。
  - 适用于行稠密的稀疏矩阵
  - 第i+1行偏移量 减去 第i行偏移量 就是 i行的非零长度。线性存储中，偏移量+长度可以确定一组元素的索引，再根据该索引获得列坐标和元素值。
- CSC：压缩稀疏列，与CSR不同的是，存储是列主序的，压缩列坐标。

- cuSPARSE的数据转换开销很大，应该尽量避免。

### 8.3 cuBLAS

- 不支持稀疏格式，善于稠密向量和稠密矩阵的优化。
- 列主序、数组索引从1开始。
  - ld refers to the leading dimension of the matrix, which in the case of column-major storage is the number of rows of the allocated matrix (even if only a submatrix of it is being used).
- cudaMalloc分配设备空间，但用cuBLAS函数传输数据。
  - cublas_Set/Get_Vector/Matrix

~~~C++
// A矩阵的维度、元素大小、主存地址 B矩阵的设备地址 lda和ldb是A矩阵和B矩阵的主维度 列主序就是行数
cublasSetMatrix(int rows, int cols, int eleSize, const void* A, int lda, void *B, int ldb);
cublasSetMatrix(M, N, sizeof(float), A, M, dA, M);
// incx incy 是各种内存地址的地址间隔
cublasSetVector(int n, int eleSize, const void *x, int incx, void *y, int incy);
// 传第一列给dV
cublasSetVector(M, sizeof(float), A, 1, dV, 1);
// 传第i行给dV
cublasSetVector(N, sizeof(float), A + i, M, dV, 1);
~~~

~~~C++
// 宏实现 以0为基准的行优先索引到列优先索引的转换
#define R2C(r, c, nrows) ((c)*(nrows) + (r))
for(int c = 0; c < ncols; c++){
    for(int r = 0; r < nrows; r++){
        A[R2C(r, c, nrow)] = ...;
    }
}
~~~

### 8.4 cuFFT

- 快速傅里叶变换

### 8.5 cuRAND

- 基于CUDA库生成伪随机数和拟随机数。
- 伪随机数生成器（PRNG）使用RNG算法生成随机数序列。每次采样是独立随机事件，有取值范围。
- 拟随机数生成器（QRNG）不是独立随机采样，会尽量均匀填充输出范围。
- 库函数既可以被主机端调用，也可以被设备端调用。
  - 主机和设备cuRAND API的配置项有4个：RNG算法、返回值遵守的分布、初始种子数值和采样偏移量。初始值可以默认初始化
  - 主机API的句柄就是随机数生成器。只需一个生成器来访问主机API。
  - 设备API的句柄是cuRAND的状态。状态对象用来维护GPU上单线程cuRAND上下文的配置和状态。通常需要分配很多设备状态对象来对应不同的GPU线程。

~~~C++
// 主机API
curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type);

curandGenerator_t rand_state;
// RNG算法
curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT);
// 正态 均匀 对数正态 泊松
curandGenerateUniform(rand_state, d_rand, d_rand_length);
// 种子
curandSetPseudoRandomGeneratorSeed(rand_state, 9872349ULL);
// 偏移量
curandSetGeneratorOffset(rand_state, 0ULL);

// QRNG
// 唯一被主机和设备端API支持的QRNG是基于Sobol拟随机序列的。
// 主机端API 只有用于QRNG的维数可以使用curandSetQuasiRandomGeneratorDimensions来设置
curandCreateGenerator(&rand_state, CURAND_RNG_QUASI_SOBOL32);
curandSetQuasiRandomGeneratorDimensions(rand_state, 2);
~~~

~~~C++
// 设备API通过初始化函数配置 RNG特定的状态对象，该对象被当成一个cuRAND生成器
// RNG算法
__device__ void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset,
                           	curandStateORWOW_t *state);
__device__ void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, 
                           	curandStateMRG32k3a_t *state);
// 正态 均匀 对数正态 泊松
float f = curand_uniform(&rand_state);
// 种子 每个线程的PRNG都需要种子
curand_init(unsigned long long seed, ...);
// 偏移量
curand_init(..., unsigned long long offset, ...);

// QRNG 
// 设备端API允许指定种子的方向向量
curandDirectionVectors32_t *h_direction_vector;
curandGetDirectionVectors32(&h_direction_vector, CURAND_DIRECTION_VECTORS_32_JOEKUO6);
cudaMemcpy(d_direction_vector, h_direction_vector, ...);

__global__ void kernel(curandDirectionVectors32_t *h_direction_vector, ...){
    curand_init(*d_direction_vector, 0, &rand_state);
}
~~~

### 8.6 Drop-in

- 该库可以使某些GPU加速库无缝替换已有的CPU库。甚至不需要重新编译。
  - 需要GPU加速库和CPU库使用相同的API。

- NVBLAS替换BLAS库，cyFFTW替换FFTW库。

~~~shell
# drop-in.c是C + BLAS程序
# CPU计算
gcc drop-in.c -lblas -lm -o drop-in
# 如果不想重新编译
env LD_PRELOAD=libnvlas.so ./drop-in

# 重新编译运行
gcc drop-in.c -lnvblas -o drop-in
~~~

### 8.7 多GPU库

- 多GPU库（又称XT库接口）能使单一的函数库调用在多个GPU上自动执行。

### 8.8 CUDA函数库的性能研究

- 数学核心库(MKL)是稀疏线代性能的黄金准则。MKL使用向量指令在多核CPU上执行密集和稀疏线代，并手动优化。
  - 用cuSPARSE和MKL做性能比较。
  - 用cuBLAS和MKL的BLAS例程做性能比较。
  - 用cuFFT和MKL的FFT例程做性能比较。
- cuFFT和FFTW库也能做性能比较。

### 8.9 OpenACC

- OpenACC是CUDA的一个补充编程模型。用基于编译器指令的API。目标是建立一个具有单线程的主机程序平台。
- OpenACC线程模型：gang（类似于线程块）、worker（类似于线程束）、vector（类似于线程）。
  - gang包含一个或多个线程，一个或多个worker。
  - worker里有一个向量宽度，由一个或多个同时执行相同指令的向量元素组成
- OpenACC主机程序中，将内核交付给多处理单元(PU)。每个PU一次只能运行一个gang，可以同时执行多个独立并发的线程（worker）。gang并行使用多个PU。GPU跑OpenACC时，一个PU就类似于一个SM。
- gang冗余模式：每个gang只有一个worker的一个vector在跑。每个gang都执行相同运算。
  - 开始执行并行区域时，gang以gang冗余模式执行，有利于gang状态初始化。
- gang分裂模式：每个gang只有一个worker的一个vector在跑。但每个活跃的vector元素执行不同的并行区域。
- worker分裂模式：每个gang里的所有worker都有一个vector在跑。
- vector分裂模式：每个gang里的所有worker，每个worker的所有vector都在跑。

- 编译器指令是一行C/C++代码，以#pragma开头。OpenACC指令要用acc关键字作唯一标识。
  - 即#pragma acc

#### 8.9.1 计算指令

- 内核指令

~~~C++
#pragma acc kernels
for(i = 0; i < N; i++){
    C[i] = A[i] + B[i];
}

#pragma acc kernels if(cond)
#pragma acc kernels async(id)	// OpenACC计算 内核指令结束时有一个默认的等待命令，async则不会被阻塞 
// 可选参数id 唯一标识该内核块 用于测试、等待什么的
#pragma acc wait(3)
// 同
acc_async_wait(3);
// 
acc_async_wait_all();
acc_async_test(3);
acc_async_test_all();
~~~

- 并行指令

~~~C++
#pragma acc parallel	// 比kernels更加详尽的并行控制 而不是依赖编译器
// 支持 num_gangs(int) num_workers(int) vector_length(int)

#pragma acc parallel reduction(op:var1, var2, ...)
// 每个gang都有每个变量的副本 并默认初始化 内核执行结束时，执行op
#pragma acc parallel reduction(+:result)
// + * max min & | ^ && ||

int a;
#pragma acc parallel private(a){
	a = ...;	// a对于其他gang和主机程序不可见
}

int a = 5;
#pragma acc parallel firstprivate(a){
	...			// a统一初始化 5，然后是private的
}
~~~

- 循环指令

~~~C++
#pragma acc parallel
{
#pragma acc loop
    for(int i = 0; i < N; i++){
        C[i] = A[i] + B[i];
    }
}

#pragma acc parallel
{
    int b = a + c;
#pragma acc loop gang	// 指明了要达到的并行度 gang冗余到gang分裂
    for(int i = 0; i < N; i++){
        ...
    }
}

#pragma acc kernels loop
for(int i = 0; i < N; i++){
        ...;
}
~~~

#### 8.9.2 数据指令

~~~C++
#pragma acc data	// 显式用于主机应用程序和加速器间传输数据

// A、B要拷到设备端 C、D要拷回来
// 这样可以省去一半的数据传输量
#pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N], D[0:N])
{
#pragma acc loop
    {
#pragma acc loop
        for(int i = 0; i < N; i++){
            C[i] = A[i] + B[i];
        }
        for(int i = 0; i < N; i++){
            D[i] = A[i] + B[i];
        }
    }
}
~~~

~~~C++
#pragma acc enter data copyin(B[0:N]) async(0)
host_work();
#pragma acc kernels async(1) wait(0)
{
    for(i = 0; i < N; i++){
        A[i] = device_work(B[i]);
    }
}
#pragma acc exit data copyout(A[0:N]) async(2) wait(1)
host_workk();
#pragma wait(2)
~~~

#### 8.9.3 运行时API

- runtimeAPI分为4个方面：设备管理、异步控制、运行时初始化和内存管理。
- 设备管理：显式控制使用哪个加速器或加速器类型来执行计算。get和set device
  - acc_device_none/acc_device_default/acc_device_host/acc_device_not_host/acc_device_nvidia/radeon/xeonphi/pgi_opencl
- 异步控制函数：检查或等待异步操作的执行状态。
  - acc_async_test/acc_async_wait
- 运行时初始化：初始化或管理OpenACC的内部状态。不显式调用时会自动调用。
  - acc_init/acc_shutdown
- 内存管理：管理加速器内存分配和DH数据传输。
  - acc_malloc/acc_free/acc_is_present/acc_copyin/acc_create/acc_delete



## 9.多GPU编程

- 待阅读。。。



## 10.程序实现的注意事项

### 10.1 CUDA C开发过程

- CUDA开发迭代模型：
  - 评估
  - 并行化
  - 优化
  - 部署
- 评估：确定性能瓶颈和高强度计算的临界区，评估GPU加速的可能性、策略。
  - 循环结构、性能分析工具发掘的热点区、已经被并行化的区域 都是评估的重点。
- 并行化：用CUDA库、手写内核、采用并行化及向量化编译器。
- 优化：网格级优化和内核级优化
  - 网格级优化：优化GPU利用率、效率。同时运行多个内核、使用流、重叠计算和IO。
  - 内核级优化：优化GPU内存带宽和计算资源，减少或隐藏指令和内存延迟。
- 部署：适应硬件环境。

#### 10.1.1 优化因素

- 重要性递减：展现足够的并行性、优化内存访问、优化指令执行。
- 并行性：保证一个SM有更多活跃的并发线程束、为每个线程束分配更多独立的工作。
  - 活跃线程束数要和SM资源占用率做一个平衡。
  - BlockDim和GridDim要做一个优化。
- 优化访存：内存访问模式、充足的并发内存访问
  - 带宽利用率要上去，对齐合并访问，一路冲突 这种。
  - 缓存的使用要考虑
  - 共享内存和全局内存配合好
- 优化指令执行：保证足够多的活跃线程束来隐藏延迟、给线程分配更多的独立工作来隐藏延迟、避免线程束分化
  - 同步指令要小心使用，怕用错，也怕影响性能

#### 10.1.2 CUDA代码编译

- 一般会有.cu和.c文件。编译分两部分：nvcc编译设备函数、c/c++ compiler编译主机函数。然后编译完的设备对象被嵌入主机目标文件中，再链接CUDA运行时库。
- CUDA的两种编译方法：整体程序编译和独立编译
  - 整体程序编译（默认）要求核函数的定义和它调用的所有设备函数必须在同一文件范围内。
  - 独立编译：一个文件定义的Device Code可以访问另一个文件的Device Code。

- 独立编译的三步：设备编译器将可重定向设备代码嵌入主机目标文件中。设备链接器链接设备对象。主机链接器将设备和主机对象组合成最终可执行程序。

- .c要想调用.cu的内核函数，需要在.cu中创建内核封装函数。

#### 10.1.3 错误处理

- 三个函数：cudaGetLastError、cudaPeekLastError、cudaGetErrorString。
  - cudaGetLastError：返回最后一个错误，并将CUDA内部状态清理为cudaSuccess。
  - cudaPeekLastError：返回最后一个错误。

### 10.2 配置文件驱动优化

- 性能抑制因素：内存带宽、指令吞吐量、延迟。

#### 10.2.1 nvprof寻找优化因素

> 虽然nvprof已经过时了，但有部分对应指标还存在，且过时指标提供了寻找优化的思路

- 全局内存的指标和事件
  - 全局内存加载和存储效率、每个全局内存加载和存储请求所造成的事务的平均数、全局内存加载和存储吞吐量。
- 共享内存的指标和事件
  - 每个共享内存加载和存储请求所造成的事务的平均数、存储体冲突事件、共享内存加载和存储指令的数量、共享内存效率。

- 寄存器溢出
  - l1_local_load/store_hit/miss、手动计算hit/miss_ratio、一级缓存命中率(l1_cache_local_hit_rate)

- 指令吞吐量
  - 分支、分化分支、分支效率、指令发出量、指令执行量

#### 10.2.2 NVIDIA工具扩展

- NVTX：追踪CPU事件和代码范围、OS和CUDA资源命名。

~~~C++
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>

nvtxEventAttributes_t eventAttrib = {0};
eventAttrib.version = NVTX_VERSION;
eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
eventAttrib.colorType = NVTX_COLOR_ARGB;
eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;


eventAttrib.color = RED;
eventAttrib.message.ascii = "HostMalloc";
nvtxRangeId_t hostMalloc = nvtxRangeStartEx(&eventAttrib);

// malloc host memory
float *h_A, *h_B, *hostRef, *gpuRef;
h_A = (float *)malloc(nBytes);
h_B = (float *)malloc(nBytes);
hostRef = (float *)malloc(nBytes);
gpuRef = (float *)malloc(nBytes);

nvtxRangeEnd(hostMalloc);
~~~

### 10.3 CUDA调试

- 内核调试：cuda-gdb、assert、printf

~~~shell
nvcc -g -G foo.cu -o foo	# 关闭了多数优化，确保执行状态真实
cuda-gdb foo

run
cuda thread lane warp block sm grid device kernel
cuda thread(128)
help cuda
~~~

- 内存调试：memcheck工具和racecheck工具



## 99.API

~~~shell
nvprof --metrics achieved_occupancy ./sumMatrix 256 1	# 占用率 = 每周期活跃线程束平均数量 / SM支持的最大线程束数量
nvprof --metrics gld_throughtput ./sumMatrix 256 1		# 全局内存加载吞吐量 GB/s
nvprof --metrics gld_efficiency ./sumMatrix 256 1		# 全局内存加载效率 = 被请求的全局加载吞吐 / 所需的全局加载吞吐
nvprof --metrics gld_transactions ./sumMatrix 256 1		# 全局加载事务

nvprof --unified-memory-profiling per-process-device ./managed 8	# 统一内存通信量
nvprof --metrics shared_load_transactions_per_request ./smemSquare	# 共享内存每个请求所需加载事务量
nvprof --metrics shared_store_transactions_per_request ./smemSquare	# 共享内存每个请求所需存储事务量
~~~

~~~C++
// 强制主机程序等待所有核函数执行结束
cudaError_t cudaDeviceSynchronize(void);
// 动态配置SM的一级缓存和共享内存
// cudaFuncCachePerferNone 无参考	cudaFuncCachePerferShared 建议48KB共享内存16KB一级缓存
// cudaFuncCachePerferL1 建议48KB一级缓存16KB共享内存		cudaFuncCachePerferEqual 建议尺寸相同
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
// CPUMem到GPUMemSymbol
cudaError_t cudaMemcpyToSymbol(void* , void* , size_t);
cudaError_t cudaMemcpyFromSymbol(void* , void* , size_t);
// GPU全局内存symbol到实际硬件地址的映射
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
//
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
// 分配统一内存
cudaMallocManaged(void**, size_t);
~~~



## 100.其他

### 1.CUDA开发环境

- NVIDIA Nsight集成开发环境
- CUDA-GDB命令行调试器
- CUDA-MEMCHECK内存分析器
- GPU设备管理工具
- 用于性能分析的可视化和命令行分析器



### 2.优化启发

- 块的数量要远多于SM数量
- 内核使用的寄存器数量会显著影响常驻线程束数量
- 考虑到指令延迟和内存延迟的隐藏，一个线程应该有 **更多独立** 的内存加载/存储操作和计算操作。
- 对齐合并内存访问

- 避免内存访问的分区冲突，这样也能尽可能碰瓷缓存的作用。
  - 在并行性较高的时候，对于某个线程束无用的数据缓存，可能会对其他并行的线程束有用，利用这个可能性。
  - 避免并行的block访问相同的分区。这是block level的优化。
- 考虑使用常量缓存和只读缓存。
- 优化层面包括：
  - 网格、线程块、线程束。
  - 全局内存（一级缓存和二级缓存）、常量内存（常量缓存）、纹理内存（只读缓存）、共享内存、寄存器。



### 3.我的3060

- CUDA cores数量：3840
- SM数量：30（128 cores per SM）
- 一个SM最多可以容纳：1536个线程（48个线程束）
- 一个block最多可以容纳：1024个线程、48KB共享内存、65536个寄存器

- 常量内存：64KB

- DRAM：6GB

- CUDA driver / runtime driver：12.0 / 11.8

- 算力：8.6

- L2 cache：3MB



- 全局内存总量：5938MB
- 每个Block的共享内存上限：48KB
- 每个Block的寄存器上限：65536
- 每个SM的共享内存总量：100KB
- 每个SM的寄存器总量：65536