# CUDA编程指南

## 1.简介

- Difference between GPU and CPU：GPU通过计算隐藏内存访问延迟，而不是依靠大型数据缓存和复杂的流控制来避免长内存访问延迟。所以更多的晶体管用来计算而不是缓存和流控制。

- CUDA：通用并行计算平台和编程模型。
  - 三个核心抽象：层次化线程组、共享存储和障碍同步。
  
  

## 2.编程模型

- CUDA中，定义的C++函数称为核函数，核函数会在N个CUDA线程中并行运行。

### 2.1 Kernel

~~~C++
// 核函数定义
// 必须返回void 异步的
__global__ void Vecadd(float *A, float *B, float *C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}
int main() {
	float *A, *B, *C;
	A = (float *)malloc(N * sizeof(float));
	B = (float *)malloc(N * sizeof(float));
	C = (float *)malloc(N * sizeof(float));
	for (int i = 0; i < N; i++) {
		A[i] = i + 1.0;
		B[i] = i + 2.0;
	}
	// N个线程的Kernel调用
	// <<< Dg, Db, Ns, S >>> dim of grid, dim of block, 可选动态空间, stream
	Vecadd <<< 1, N>>>(A, B, C);
	for (int i = 0; i < N; i++) {
		cout << C[i] << " ";
	}
	cout << endl;
	return 0;
}
~~~

### 2.2 Thread Hierarchy

~~~C++
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
	// threadIdx是三维向量，thread ID就是三维数字线性化后的idx。
	int i = threadIdx.x;
	int j = threadIdx.y;
	C[i][j] = A[i][j] + B[i][j];
}
int main() {
	float A[N][N], B[N][N], C[N][N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = i + j + 1.0;
			B[i][j] = i + j + 2.0;
		}
	}
	int numBlocks = 1;
	dim3 threadsPerBlock(N, N);
	MatAdd <<< numBlocks, threadsPerBlock>>>(A, B, C);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << C[i][j] << " ";
		}
		cout << endl;
	}
}
// 每个block的线程数上限是1024
~~~

~~~C++
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < N && j < N)
		C[i][j] = A[i][j] + B[i][j];
}

dim3 threadsPerBlock(16, 16);
dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
~~~

- Thread Blocks的执行被要求是独立。无论有多少个core，而且可以以任意顺序调度。

- Block内Thread可以通过 __syncthreads() 调用来指定同步点，类似于barrier。
- Thread Block Clusters：一个可选的层次，GPU Processing Cluster(GPC)内的thread blocks的协同方式类似于streaming multiprocessor内的thread的协同方式。
- Thread Block Clusters可用的两种方法：一个是编译时核属性 __cluster_dims__(X,Y,Z)，一个是cudaLaunchKernelEx API。
  - 编译时，不可改变cluster size
  - 运行时，cudaLaunchKernelEx API 可以设置cluster size

~~~C++
// 编译时
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float *out) {

}
cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);	// grid dims 必须是 cluster size的倍数

// 运行时
__global__ void cluster_kernel(float *input, float *out) {

}
int main() {
	float *input, *output;
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	cluster_kernel<<<numBlocks, threadsPerBlock>>>();
    {
        cudaLaunchConfig_t config = {0};

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;	// 也就是说 这里可以是运行时确定的变量
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim =  numBlocks;
        config.blockDim = threadsPerBlock;
        
        cudaLaunchKernelEx(&config, cluster_kernel, input, ouput);
    }
}
~~~

- 一个cluster的线程块可以访问分布式共享存储，原子读、写、操作地址空间。

### 2.3 Memory Hierarchy

- 线程有自己的寄存器和local memory
- 线程块有shared memory
- 线程块集群有 由各Block的shared memory 组成的分布式共享存储
- GPU有一个global memory 供所有kernel使用

### 2.4 异构编程

- CUDA假设C++程序运行在独立的物理device上，作为host的协处理器。运行时互不影响。
- CUDA也假设host和device有各自独立的memory空间。

### 2.5 异步SIMT编程模型

- 异步模型 用来加速内存操作。
  - 定义了 Asynchronous Barrier
  - 定义了 计算同时执行的 访问global mem 操作 memcpy async
- 异步操作 被定义为由一个CUDA线程发起 并 由另一个线程异步执行。
- 异步线程 会和发起该异步操作的线程关联。
- 异步线程 会用一个同步对象来同步操作的完成，该对象可以由用户或库管理。cuda::memcpy_async 或 cooperative_groups::memcpy_async
- 同步对象 可以是一个 cuda::barrier 或 一个 cuda::pipeline。
- 同步对象可以作用的scope（scope内的异步操作可能用该对象同步）如下：
  - cuda::thread_scope::thread_scope_thread	仅初始化异步操作的CUDA线程 进行同步
  - cuda::thread_scope::thread_scope_block    和初始化线程相同的block的 ALL or ANY CUDA线程进行同步
  - cuda::thread_scope::thread_scope_device    和初始化线程相同的device的 ALL or ANY CUDA线程进行同步
  - cuda::thread_scope::thread_scope_system    和初始化线程相同系统的 ALL or ANY CUDA or CPU线程进行同步

### 2.6 计算能力

- CUDA X.Y. 版本 对应了 某个GPU架构。如：CUDA 9.0 for NVIDIA Hopper GPU。



## 3.编程接口

- CUDA C++ 由 最小C++语言扩展集 和 运行时库 构成。
- CUDA Driver API通过扩展low level的概念来提供额外控制
  - CUDA context：设备主机进程的模拟
  - CUDA module：设备动态加载库的模拟

- 包含扩展集的源文件需要 nvcc 的编译。

### 3.1 NVCC

- PTX：CUDA指令集架构。
- nvcc驱动器把各编译阶段需要的工具串起来，将PTX code或者C++ code 编译成二进制可执行代码，放在设备端执行。
- 二进制形式叫cubin object，汇编形式叫PTX code。
- -code指出了二进制对应的目标架构。
  - -code=sm_35，表明算力是3.5。
- -arch指出了C++编译成PTX对应的算力。
  - -arch=compute_30，表明算力应该是3.0及以上
- Just-in-Time Compilation是把PTX或NVVM IR进一步编译成二进制码的编译方式。这种编译方式无疑会增加启动时间，但它通过缓存二进制代码的方式弥补了这一点。Just-in-Time是应用程序在编译应用程序时不存在的设备上运行的唯一方法。
  - 设备驱动升级时，缓存二进制代码自动不可用，以享受新架构的特性。
  - 应该是对比，直接从C++编译成二进制，来获知好处。如果直接从Cpp到二进制的话，不存在适应新架构的可能性，但是可以降低启动时间。

- 为某些特定计算能力生成的PTX代码始终可以编译为具有更大或相等计算能力的二进制代码。但是新特性可能用不上。
  - 可能是因为PTX code并没有包含新特性的能力。

- 设备代码只支持部分C++特性。

### 3.2 CUDA 运行时

- 运行时库 包含了 运行在host上的C++函数，功能包括：device内存分配、释放，host和device内存间的数据传输，多device系统管理等。

#### 3.2.1 设备内存初始化

- 运行时在第一次调用运行时函数时初始化。CUDA context在第一次调用需要活跃上下文的运行时函数时被初始化，在cudaDeviceReset()调用时被摧毁。

#### 3.2.2 设备内存

- 设备内存分配：线性内存或CUDA数组。
  - CUDA 数组是不透明的内存布局，针对纹理获取进行了优化。

- cudaMallocPitch() and cudaMalloc3D() 可以保证分配的内存空间满足全局内存对齐要求。通过返回的pitch确定内存数组最低维的字节size。

- 数据传输：

~~~C++
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

// 获取分配空间的大小	
cudaGetSymbolSize();		
// is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space
cudaGetSymbolAddress();	 
~~~

#### 3.2.3 设备内存L2访问管理

- 从CUDA11.0，算力8.0开始，可以通过设置L2来适应 重复全局内存访问（persisting）和单次全局内存访问（streaming）。
- L2 cache的一部分会预留给 持久化数据访问。只有当持久化数据访问没在用的时候才把这部分让给常规或流式的数据访问。
- 调整预留空间大小

~~~C++
cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2
cache for persisting accesses or the max allowed */
~~~

- CUDA流设置数据访问持久化区域

~~~C++
/* 当内核随后在CUDA流中执行时，在全局内存范围[ptr..ptr+num_bytes]内的内存访问比对其他全局内存位置的访问更有可能持久化在L2缓存中 */
cudaStreamAttrValue stream_attribute; // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes; // Number of bytes for persistence access.
// (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio = 0.6; // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // Type of access property on cache miss.
// Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
// 该流在全局内存范围[ptr..ptr+num_bytes]内的数据访问 有随机的60%的数据会持久化，40%会流式访问。
~~~

- L2访问属性：
  - cudaAccessPropertyStreaming：优先被清理
  - cudaAccessPropertyPersisting：优先留在预留区内，更可能被持久化。
  - cudaAccessPropertyNormal：重置状态，删除持久属性。
- cudaCtxResetPersistingL2Cache() 重置所有L2 cache lines 到 normal状态。
- L2预留缓存区是所有内核共享的，所以要考虑，总的预留区，可能并行的内核一共用的预留区。
- cudaDeviceget/setLimit

#### 3.2.4 共享内存

- 共享内存：软件管理缓存，减少全局内存访问。

~~~C++
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col){
	return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value){
	A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col){
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C){
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}
~~~

#### 3.2.5 分布式共享内存

- 算力9.0提供的线程块集群才有分布式共享内存。

#### 3.2.6 锁页主机内存

- cudaHostAlloc() and cudaFreeHost() allocate and free page-locked host memory；
- cudaHostRegister() page-locks a range of memory allocated by malloc()；

- 优势：
  - 在一些设备里，锁页内存的copy能和内核运行并行。
  - 在一些设备里，锁页内存可以map到设备空间，避免copy。
  - 有前端总线的系统，锁页主机的数据传输带宽很高，如果是写合并内存，带宽更高。
- 劣势：
  - 会降低主机的性能
  - 属于稀缺资源
- Portable Memory允许多个设备享受锁页内存的优势。
- Write-Combining Memory就是不用一级和二级缓存，它很慢。
- Mapped Memory将锁页内存映射到设备指定空间。

#### 3.2.7 异步并行执行

##### 3.2.7.1 Concurrent and Overlap

- 以下任务是可以独立并行执行的：

  - Computation on the host;

  - Computation on the device;

  - Memory transfers from the host to the device;

  - Memory transfers from the device to the host;

  - Memory transfers within the memory of a given device;

  - Memory transfers among devices

- 不使用主机锁定内存页的异步传输是同步的。

- 显式同步：
  - cudaDeviceSynchronize() 同步所有主机线程所有流之前的命令
  - cudaStreamSynchronize() 同步参数指定的流之前的命令，不会阻塞其他流。
  - cudaStreamWaitEvent() 参数指定事件和流，事件完成后，流之后的命令开始执行。
  - cudaStreamQuery() 查询之前的命令有没有完成。

- 隐式同步：如果主机线程在不同的流中执行了以下操作，则来自不同流的两个命令不能并发运行
  - a page-locked host memory allocation,
  - a device memory allocation,
  - a device memory set,
  - a memory copy between two addresses to the same device memory,
  - any CUDA command to the NULL stream,
  - a switch between the L1/shared memory configurations.

- 应用程序并行优化指导建议：
  - 所有独立操作应该在非独立操作前发起
  - 任何类型的同步都应该被尽量延迟

- cudaLaunchHostFunc()来提供主机回调。


##### 3.2.7.2 CUDA Graph

- CUDA 图：一系列连接起来的操作，与执行分离。可以一次定义，多次执行。
  - 图比流的启动时间更短，因为更多的设置被提前完成了。图的工作流比流的分段提交机制更适合优化。

- 图的工作提交分为三个阶段：定义、实例化、执行。
  - 在定义阶段，程序在图中创建操作的描述以及它们之间的依赖关系。
  - 实例化获取图形模板的快照，验证它，并执行大量的设置和初始化工作，目的是最小化启动时需要做的事情。结果实例称为可执行图。
  - 一个可执行图可以被启动到一个流中，类似于任何其他CUDA工作。它可以在不重复实例化的情况下被启动任意次数。

- 图：操作是节点，操作间的依赖关系是边。一个操作的依赖操作都执行完成后，该操作可能在任何时间启动，这取决于CUDA系统。
- 节点可以是：kernel、CPU function call、memory copy、memset、empty node、waiting on an event、recording an event、signalling an external semaphore、waiting on an external semaphore、child graph: To execute a separate nested graph.
- 创建图的操作有两种：显式API和流捕获

~~~C++
/// API创建图
// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);
// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified
// at node creation.
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);
// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, 1); // A->B
cudaGraphAddDependencies(graph, &a, &c, 1); // A->C
cudaGraphAddDependencies(graph, &b, &d, 1); // B->D
cudaGraphAddDependencies(graph, &c, &d, 1); // C->D

/// 捕获图
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);
cudaStreamEndCapture(stream, &graph);
~~~

- cudaStreamLegacy (the “NULL stream”)不能捕获。
- 一个被捕获的event在捕获图中表现为a set of nodes。
  - 当被捕获事件在等待另一个未完成的流时，另一个流也会被捕获到图中。

- 全图更新

~~~C++
cudaGraphExec_t graphExec = NULL;
for (int i = 0; i < 10; i++) {
    cudaGraph_t graph;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;
    // In this example we use stream capture to create the graph.
    // You can also use the Graph API to produce a graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // Call a user-defined, stream based workload, for example
    do_cuda_work(stream);
    cudaStreamEndCapture(stream, &graph);
    // If we've already instantiated the graph, try to update it directly
    // and avoid the instantiation overhead
    if (graphExec != NULL) {
        // If the graph fails to update, errorNode will be set to the
        // node causing the failure and updateResult will be set to a
        // reason code.
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    }
    // Instantiate during the first iteration or whenever the update
        // fails for any reason
    if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {
    // If a previous update failed, destroy the cudaGraphExec_t
        // before re-instantiating it
        if (graphExec != NULL) {
        	cudaGraphExecDestroy(graphExec);
        }
        // Instantiate graphExec from graph. The error node and
        // error message parameters are unused here.
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }
    cudaGraphDestroy(graph);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}
~~~

- 单节点更新

~~~C++
cudaGraphExecKernelNodeSetParams();
cudaGraphExecMemcpyNodeSetParams();
cudaGraphExecMemsetNodeSetParams();
cudaGraphExecHostNodeSetParams();
cudaGraphExecChildGraphNodeSetParams();
cudaGraphExecEventRecordNodeSetEvent();
cudaGraphExecEventWaitNodeSetEvent();
cudaGraphExecExternalSemaphoresSignalNodeSetParams();
cudaGraphExecExternalSemaphoresWaitNodeSetParams();
~~~

- 图执行在流中完成，以便与其他异步工作排序。然而，该流仅用于排序;它不限制图的内部并行性，也不影响图节点的执行位置

##### 3.2.7.3 事件

##### 3.2.7.4 异步调用

- cudaSetDeviceFlags()可以设置主机线程在等待同步函数时，会自旋、礼让还是阻塞。

#### 3.2.8 多设备系统

##### 3.2.8.1 流和事件行为

- 如果内核启动被发送到与当前设备不关联的流，则内核启动将失败。
- 当操作关联的流或输入事件和当前设备并不关联时：
  - memory copy会成功
  - cudaEventRecord() will fail
  - cudaEventElapsedTime() will fail
  - cudaEventSynchronize() and cudaEventQuery() will succeed
  - cudaStreamWaitEvent() will succeed
- 每个设备都有自己的默认流，所以 发送到设备的默认流的命令可以无序执行，也可以与发送到任何其他设备的默认流的命令并行执行。

##### 3.2.8.2 点对点内存访问

- PCIe和/或NVLINK拓扑结构决定了，设备能够寻址彼此的内存(即在一个设备上执行的内核可以解引用指向另一个设备内存的指针)。如果cudaDeviceCanAccessPeer()为这两个设备返回true，则支持这两个设备之间的点对点内存访问特性。

~~~C++
cudaSetDevice(0); // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size); // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
cudaSetDevice(1); // Set device 1 as current
cudaDeviceEnablePeerAccess(0, 0); // Enable peer-to-peer access
// with device 0
// Launch kernel on device 1
// This kernel launch can access memory on device 0 at address p0
MyKernel<<<1000, 128>>>(p0);
~~~

##### 3.2.8.3 点对点内存复制

- 对等内存复制不用经过主机，速度快。

~~~C++
size_t size = 1024 * sizeof(float);

cudaSetDevice(0); // Set device 0 as current
float* p0;
cudaMalloc(&p0, size); // Allocate memory on device 0

cudaSetDevice(1); // Set device 1 as current
float* p1;
cudaMalloc(&p1, size); // Allocate memory on device 1

cudaSetDevice(0); // Set device 0 as current
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0

cudaSetDevice(1); // Set device 1 as current
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
~~~

#### 3.2.9 统一虚拟内存空间

- 64位进程的应用，主机和多个设备（2.0以上算力）的地址空间就是统一的。通过CUDA API分配的主机内存空间和设备内存空间在一个统一虚拟内存空间里。
- cudaPointerGetAttributes() 获得指针内存位置。
- the cudaMemcpyKind parameter of cudaMemcpy*() can be set to cudaMemcpyDefault，自动推断指针内存位置。
  - 前提是指针指向统一虚拟内存空间。
- cudaHostAlloc() are automatically portable，任何设备可用。

#### 3.2.10 进程间通信

- 由主机线程创建的任何设备内存指针或事件句柄都可以被同一进程中的任何其他线程直接引用。但是，它在该进程之外无效。
- 跨进程通信要用Inter Process Communication API。
  - 一个进程通过cudaIpcGetMemHandle()拿到特定指针的IPC handle
  - 然后通过标准IPC机制传递给另一个进程
  - 另一个进程再通过cudaIpcOpenMemHandle()拿到该指针

#### 3.2.11 错误检测

- 在某个异步函数调用之后检查异步错误的唯一方法是在调用之后通过调用cudaDeviceSynchronize()(或通过使用异步并发执行中描述的任何其他同步机制)进行同步，并检查cudaDeviceSynchronize()返回的错误代码。
- cudaError_t指出了运行时函数错误。
- cudaPeekAtLastError() returns this variable. cudaGetLastError() returns this variable and resets it to cudaSuccess.
- **内核启动不会返回任何错误代码，因此必须在内核启动之后调用cudaPeekAtLastError()或cudaGetLastError()来检索任何启动前的错误。**
- **为了确保cudaPeekAtLastError()或cudaGetLastError()返回的任何错误不是源于内核启动之前的调用，必须确保运行时错误变量在内核启动之前被设置为cudaSuccess，例如，在内核启动之前调用cudaGetLastError()。**
- 内核启动是异步的，因此为了检查异步错误，应用程序必须在内核启动和调用cudaPeekAtLastError()或cudaGetLastError()之间进行同步。

#### 3.2.12 调用栈

- the size of the call stack can be queried using cudaDeviceGetLimit() and set using cudaDeviceSetLimit().

#### 3.2.13 纹理和表面存储

- texture reference API和texture object API能访问到纹理和表面内存。

##### 3.2.13.1 纹理内存

- 通过纹理引用API读取纹理内存的过程叫 texture fetch。每次纹理获取都指定一个称为texture object的参数。
- 纹理：纹理内存中被取回的一块。在纹理对象创建时指定，通过API绑定到纹理引用。
- 纹理引用：编译时创建。
  - 多个纹理引用可以指向相同的纹理。
- 纹理对象：运行时创建。
- 根据纹理维度定位为一维数组、二维数组和三维数组，称作texels(texture elements)。
- texel的类型，仅限于基本整数和单精度浮点类型以及内置vector中定义的任何1-、2-和4分量向量（类型从基本整数和单精度浮点类型派生的类型）。
- Read Mode：cudaReadModeNormalizedFloat和cudaReadModeElementType
  - cudaReadModeNormalizedFloat会返回浮点数，the full range of the integer type is mapped to [0.0, 1.0] for unsigned integer type and [-1.0, 1.0] for signed integer type。
  - cudaReadModeElementType不会做映射
- 纹理坐标也可以被映射。
- 寻址模式：三维数组对应三维坐标。cudaAddressModeBorder, cudaAddressModeClamp, cudaAddressModeWrap, and cudaAddressModeMirror。
  - 默认(clamp)模式：将坐标固定到有效范围:对于非标准化坐标[0,N)，对于标准化坐标[0.0,1.0)。越界是合法的。
  - border模式：越界坐标返回0；
  - wrap模式：对于坐标x，转换为frac(x)=x-floor(x)；
  - mirror模式： each coordinate x is converted to frac(x) if floor(x) is even and 1-frac(x) if floor(x) is odd
- 过滤模式：cudaFilterModePoint 和 cudaFilterModeLinear。

###### 3.2.13.1.1. Texture Object API

- A texture object is created using cudaCreateTextureObject() from a resource description of type struct cudaResourceDesc, which specifies the texture, and from a texture description defined as such：

~~~C++
struct cudaTextureDesc{
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode filterMode;
    enum cudaTextureReadMode readMode;
    int sRGB;
    int normalizedCoords;
    unsigned int maxAnisotropy;
    enum cudaTextureFilterMode mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
};
~~~

~~~C++
// Allocate CUDA array in device memory
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaArray_t cuArray;
cudaMallocArray(&cuArray, &channelDesc, width, height);
// Set pitch of the source (the width in memory in bytes of the 2D array pointed
// to by src, including padding), we dont have any padding
const size_t spitch = width * sizeof(float);
// Copy data located at address h_data in host memory to device memory
cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float), height, cudaMemcpyHostToDevice);
// Specify texture
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
resDesc.res.array.array = cuArray;
// Specify texture object parameters
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));
texDesc.addressMode[0] = cudaAddressModeWrap;
texDesc.addressMode[1] = cudaAddressModeWrap;
texDesc.filterMode = cudaFilterModeLinear;
texDesc.readMode = cudaReadModeElementType;
texDesc.normalizedCoords = 1;
// Create texture object
cudaTextureObject_t texObj = 0;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
~~~

- Layered Textures
- Cubemap Textures
- Cubemap Layered  Textures
- Texture Gather

##### 3.2.13.2 表面内存

- CUDA array created with the **cudaArraySurfaceLoadStore** flag, can be read and written via a surface object or surface reference using the functions described in Surface Functions.

~~~C++
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
cudaArray_t cuInputArray;
cudaMallocArray(&cuInputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
cudaArray_t cuOutputArray;
cudaMallocArray(&cuOutputArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);
// Set pitch of the source (the width in memory in bytes of the 2D array
// pointed to by src, including padding), we dont have any padding
const size_t spitch = 4 * width * sizeof(unsigned char);
// Copy data located at address h_data in host memory to device memory
cudaMemcpy2DToArray(cuInputArray, 0, 0, h_data, spitch, 4 * width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
// Specify surface
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeArray;
// Create the surface objects
resDesc.res.array.array = cuInputArray;
cudaSurfaceObject_t inputSurfObj = 0;
cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
resDesc.res.array.array = cuOutputArray;
cudaSurfaceObject_t outputSurfObj = 0;
cudaCreateSurfaceObject(&outputSurfObj, &resDesc);
~~~

- Cubemap Surfaces
- Cubemap Layered Surfaces

##### 3.2.13.3 CUDA数组

- CUDA arrays are only accessible by kernels through texture fetching as described in Texture Memory or surface reading and writing as described in Surface Memory.
- CUDA数组是专为texture fetching优化的不透明内存布局。
- CUDA数组是1或2或3维的texel数组。

##### 3.2.13.4 读写一致性

- 纹理和表面内存会被缓存，相同内核调用产生的纹理或表面内存写后读并不安全。只有其他之前的内核调用产生的纹理或表面内存写才安全。

#### 3.2.14 Graphics Interoperability

- 图形化渲染相关的，未学习。

#### 3.2.15 External Resource Interoperability

- CUDA能用的外部API：操作系统原生的句柄、类似于NVIDIA Software Communication Interface的统一接口。
- 两种能导入的资源类型：内存对象和同步对象。
- Memory objects import：cudaImportExternalMemory(). An imported memory object can be accessed from within kernels using device pointers mapped onto the memory object via cudaExternalMemoryGetMappedBuffer() or CUDA mipmapped arrays mapped via
  cudaExternalMemoryGetMappedMipmappedArray(). Depending on the type of memory object, it may be possible for more than one mapping to be setup on a single memory object. The mappings must match the mappings setup in the exporting API. Any mismatched mappings result in undefined behavior. Imported memory objects must be freed using cudaDestroyExternalMemory(). Freeing a memory object does not free any mappings to that object. Therefore, any device pointers mapped onto that object must be explicitly freed using
  cudaFree() and any CUDA mipmapped arrays mapped onto that object must be explicitly freed using cudaFreeMipmappedArray(). It is illegal to access mappings to an object after it has been destroyed.
- Synchronization objects can be imported into CUDA using cudaImportExternalSemaphore(). An imported synchronization object can then
  be signaled using cudaSignalExternalSemaphoresAsync() and waited on using cudaWaitExternalSemaphoresAsync().  Depending on the type of the imported synchronization object, there may be additional constraints imposed on how they can be signaled and waited on,
  as described in subsequent sections. Imported semaphore objects must be freed using cudaDestroyExternalSemaphore().
- NvSciBuf and NvSciSync are interfaces developed for serving the following purposes:
  - NvSciBuf: Allows applications to allocate and exchange buffers in memory
  - NvSciSync: Allows applications to manage synchronization objects at operation boundaries

#### 3.2.16 CUDA用户对象

- A typical use case would be to immediately move the sole user-owned reference to a CUDA graph after the user object is created. CUDA will manage the graph operations automatically.

~~~C++
cudaGraph_t graph; // Preexisting graph
Object *object = new Object; // C++ object with possibly nontrivial destructor
cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object, // Here we use a CUDA-provided template wrapper for this API,
    // which supplies a callback to delete the C++ object pointer
    1, // Initial refcount
    cudaUserObjectNoDestructorSync // Acknowledge that the callback cannot be
    // waited on via CUDA
);
cudaGraphRetainUserObject(
    graph,
    cuObject,
    1, // Number of references
    cudaGraphUserObjectMove // Transfer a reference owned by the caller (do
    // not modify the total reference count)
);
// No more references owned by this thread; no need to call release API
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0); // Will retain a
// new reference
cudaGraphDestroy(graph); // graphExec still owns a reference
cudaGraphLaunch(graphExec, 0); // Async launch has access to the user objects
cudaGraphExecDestroy(graphExec); // Launch is not synchronized; the release
// will be deferred if needed
cudaStreamSynchronize(0); // After the launch is synchronized, the remaining
// reference is released and the destructor will
// execute. Note this happens asynchronously.
// If the destructor callback had signaled a synchronization object, it would
// be safe to wait on it at this point.
~~~

### 3.3 版本和算力

- 在开发CUDA应用程序时，开发人员应该关注两个版本号:描述计算设备的通用规范和功能的计算能力(参见计算能力)和描述驱动程序API和运行时支持的功能的CUDA驱动程序API的版本。
- 驱动API：CUDA_VERSION。它是向后兼容的。

### 3.4 计算模式

### 3.5 模式转换



## 4.硬件实现

- NVIDIA GPU架构是围绕一个可伸缩的多线程流阵列构建的多处理器(SMs)。当主机CPU上的CUDA程序调用内核网格时，网格的块被枚举并分布到具有可用执行能力的多处理器上。一个线程块的线程在一个多处理器上并发执行，多个线程块可以在一个多处理器上并发执行。当线程块终止时，新的块会在空出的多处理器上启动。SIMT体系结构中，指令是流水线化的，在单个线程中利用指令级并行性，以及通过同时硬件多线程实现的广泛的线程级并行性。
- NIVIDA GPU是小端字节序。

### 4.1 SIMT架构

- 多处理器以32个并行线程(称为warp)为一组来创建、管理、调度和执行线程。组成warp的各个线程从相同的程序地址开始，但是它们有自己的指令地址计数器和寄存器状态，因此可以自由地独立分支和执行。
- 当多处理器有一个或多个线程块要执行时，它将它们划分为warp，每个warp由warp调度器调度执行。
- 分支分化仅出现在一个warp里。
- 使用独立线程调度，GPU维护每个线程的执行状态，包括程序计数器和调用堆栈，并可以以每个线程的粒度生成执行，以更好地利用执行资源或允许一个线程等待另一个线程生成数据。调度优化器决定如何将来自相同warp的活动线程分组到SIMT单元中。这保留了之前NVIDIA gpu中SIMT执行的高吞吐量，但具有更大的灵活性:线程现在可以以sub-warp粒度发散和重新聚集。

- 参与当前指令的warp线程称为活动线程，而不在当前指令上的线程称为非活动线程(禁用)。
  - 线程提前退出、分支分化、不足一个warp等原因会导致非活动线程。
  - 如果由warp执行的原子指令为多个线程读取、修改和写入全局内存中的同一位置，则每次对该位置的读取/修改/写入都发生，并且都是序列化的，但它们发生的顺序是未定义的。

### 4.2 硬件多线程

- 多处理器处理的每个warp的执行上下文(程序计数器、寄存器等等)在warp的整个生命周期内都在芯片上维护。因此，从一个执行上下文切换到另一个执行上下文没有任何成本，并且在每条指令发出时，warp调度器都会选择一个具有线程准备执行下一条指令的warp，并向这些线发出指令。

- 每个多处理器都有一组32位寄存器，这些寄存器在warp之间进行分区，还有一个并行数据缓存或共享内存，这些并行数据缓存或共享内存在线程块之间进行分区。
- 对于一个给定的内核，多处理器上可以驻留和一起处理的块和warp的数量取决于：
  - 内核使用的寄存器和共享内存的数量
  - 多处理器上可用的寄存器和共享内存的数量
  - 每个多处理器也有最大驻留块数和最大驻留warp数

- 为一个块分配的寄存器总数和共享内存总量记录在CUDA Toolkit提供的CUDA Occupancy Calculator。



## 5.性能指南

### 5.1总体性能优化策略

- 最大化并行执行以实现最大的利用率
- 优化内存使用，以实现最大的内存吞吐量
- 优化指令使用，以实现最大的指令吞吐量

- 最小化内存抖动

### 5.2 最大化使用











## 100.其他

### 1. GP100 Streaming Multiprocessors架构：

- 核心 core/streaming Processor

- 共享内存/一级缓存

- 寄存器文件

- 加载/存储单元

- 特殊功能单元

- 线程束调度器

<img src="C:/Users/Lenovo/Desktop/%E9%B1%BC%E5%A7%AC%E7%8E%84%E7%9A%84%E4%B8%9C%E8%A5%BF/Typora%E5%9B%BE%E7%89%87/Streaming%20Multiprocessors%E6%9E%B6%E6%9E%84.png" style="zoom:80%;" />

### 2. CUDA Memory

- [CUDA ---- Memory Model - 苹果妖 - 博客园 (cnblogs.com)](https://www.cnblogs.com/1024incn/p/4564726.html)
- 在CUDA中，纹理内存和常量内存同全局内存一样，可以被所有线程访问。他俩都是只读内存。

![](C:/Users/Lenovo/Desktop/TyporaAll/Typora%E5%9B%BE%E7%89%87/CUDA%E5%8F%AF%E7%BC%96%E7%A8%8B%E5%86%85%E5%AD%98%E6%9E%B6%E6%9E%84.png)



# CUDA最佳实践指南

## 0.APOD

- Assess, Parallelize, Optimize, Deploy (APOD)
  - 评估、并行化、优化、部署

## 1.异构计算

### 1.1 主机和设备的区别

- 主要区别在于：线程模型和分离的物理内存。
  - CPU内核的设计目的是最小化同一时间的少数线程的延迟，而gpu的设计目的是处理大量并发的轻量级线程，以最大限度地提高吞吐量。这个考虑多线程数量和线程的量级等。
  - 物理内存是分开的，需要通信。

### 1.2 什么东西应该跑在GPU上？

- 供大量并发线程并行执行的大量数据。

- 关于数据传输成本的考虑：
  - 操作的复杂性应该证明在设备之间移动数据的成本是合理的。
  - 数据应该尽可能长时间地保存在设备上，避免无用传输。
- 设备上运行的相邻线程在内存访问方面应该具有一定的一致性。



## 2.应用分析

### 2.1 识别热点

- 识别热点，找到应用程序中运行耗时最长的部分。

### 2.2 Strong Scaling and Amdahl's Law

- 强伸缩性是对固定的总体问题大小，随着系统中添加更多处理器，解决问题的时间如何缩短的度量。具有线性强伸缩性的应用程序具有与所使用的处理器数量相等的加速。
- 最大加速比S：$S=\frac{1}{(1-P)+\frac{P}{N}}$
  - P是可并行化的代码部分所花费的总串行执行时间的百分比，N是并行处理器数量。
  - 假设原来在一个系统中执行一个程序需要时间$T_{old}$，其中某一个部分占的时间百分比为$\alpha$，然后，把这一部分的性能提升$k$倍。即这一部分原来需要的时间为$\alpha T_{old}$，现在需要的时间变为$\alpha T_{old}/k$。则整个系统执行此程序需要的时间变为：
  - $T_{new} = (1-\alpha)T_{old}+\alpha T_{old}/k = T_{old}(1-\alpha+\alpha/k)$

- 在P比较小的情况下，N再大，意义也不大。在P比较大的情况下，N大就很有用。

### 2.3 Weak Scaling and Gustafson's Law

- 弱伸缩性是对随着系统中添加更多处理器，总体问题大小也会随之增加。弱伸缩性是一种度量方法，用于度量在每个处理器具有固定问题大小的系统中添加更多处理器时，解决问题所需的时间如何变化。
- 加速比S：$S=N+(1-P)(1-N)$
  - P是可并行化的代码部分所花费的总串行执行时间的百分比，N是并行处理器数量。
  - a是串行执行时间，b是并行执行时间，n是处理器个数，P是并行比例（同上）。原执行时长a+b，加入新设备后执行时间a+nb。
  - $S=\frac{a+nb}{a+b}=\frac{a}{a+b}+\frac{nb}{a+b}=\frac{a}{a+b}+n(1-\frac{a}{a+b})=n+(1-n)(1-P)$



## 3.并行化应用

- 对于某些应用，可以简单到调用现有的gpu优化库(如cuBLAS、cuFFT或Thrust)，也可以简单到向并行编译器添加一些预处理器指令作为提示。
- 但另一些应用就得重构来暴露其固有的并行性。CUDA旨在使这种并行性的表达尽可能简单，同时在具有CUDA功能的gpu上实现最大并行吞吐量的操作。



## 4.开始

### 4.1并行库

- The CUDA Toolkit includes a number of such libraries that have been fine-tuned for NVIDIA CUDA GPUs, such as cuBLAS, cuFFT, and so on.
- 并行库的应用关键在于能符合应用需求。如BLAS改cuBLAS、FFTW改cuFFT。

- **Thrust**：并行C++模板库。Thrust提供了丰富的数据并行原语集合，如扫描、排序和减少，这些原语可以组合在一起，用简洁易读的源代码实现复杂的算法。Thrust可以用于CUDA应用程序的快速原型设计。

### 4.2 并行编译器

- pragma 这种notion来提示编译器哪里并行。
- OpenACC

### 4.3 编码以暴露并行性

- 确定热点后，可以用CUDA c++并行化该部分代码。
- 但如果程序的运行时长分布是扁平化的，那么可能需要进行一定程度的代码重构，以暴露应用程序中固有的并行性。这是值得的。



## 5.得到正确答案

- 异构并行计算的也需要得到正确的计算结果，可能遇到的困难包括：线程问题、浮点值计算方式导致的意外值，以及CPU和GPU处理器操作方式差异带来的挑战。

### 5.1 Verification

#### 5.1.1 对比参考答案

- 在修改已有代码时，通过对比已知的正确的（好的）的算法的结果，来校验正确性。
- 或许是按位对比，或许是很小的误差，就认为是正确的。

#### 5.1.2 单元测试

- 用多个短小的--device--取代集成的--global--，方便在集成前测试。
- --device--和--host--可以同时在CPU和GPU端测试。
- 同时，如果大部分工作分解到了--device--和--host--里，那么测试的时候就能少很多重复执行。

### 5.2 Debug

- CUDA-GDB
- 第三方：https://developer.nvidia.com/debugging-solutions

### 5.3 数值准确率和精确度

- 浮点计算出问题主要是存储和计算方式的锅。
- 浮点算法的主要特性如下讨论。其他特性在CUDA c++编程指南的特性和技术规范以及白皮书和附带的关于浮点精度和性能的网络研讨会中都有介绍。

#### 5.3.1 单双精度

- 单双精度的计算结果不一样，因为数值表示精度的不同和舍入问题。
- 不能期望确切的值，只能是在一定误差范围内比较。

#### 5.3.2 浮点运算不符合结合律

#### 5.3.3 IEEE 754

- 所有CUDA计算设备都遵循IEEE 754二进制浮点表示标准。
- 有一些小例外，在《Features and Technical Specifications of the CUDA C++ Programming Guide》里说了。
- fused multiply-add (FMA) instruction通常就和分开算不一样。

#### 5.3.4 x86 80-bit Computations

- X86处理器在执行浮点计算时可以使用80位双扩展精度数学。这些计算的结果经常不同于在CUDA设备上执行的纯64位操作。要在值之间获得更接近的匹配，请将x86主机处理器设置为使用常规双精度或单精度(分别为64位和32位)。这是通过FLDCW x86汇编指令或等效的操作系统API完成的



## 6.优化CUDA应用

- APOD是迭代的，在看到良好的加速结果之前，没必要记住优化策略，可以边学边记。

- 优化可以应用于不同的级别，从与计算重叠的数据传输一直到微调浮点操作序列。可用的分析工具对于指导这个过程是非常宝贵的，因为它们可以帮助为开发人员的优化工作提供次优行动方案，并为本指南的优化部分的相关部分提供参考。



## 7.性能标准

- 使用CPU定时器和CUDA事件正确测量性能。探讨带宽如何影响性能指标，以及如何减轻它带来的一些挑战。

### 7.1 CPU计时器

- CUDA的好多API都是异步的，即，执行结束前就会调用返回。所以，CPU计时前，需要调用cudaDeviceSynchronize()。

- 因为驱动程序可能会交错执行来自其他非默认流的CUDA调用，其他流中的调用可能会包含在计时中。
- 因为默认流stream 0在设备上表现出了串行化行为，所以这些函数可以在默认流中可靠地进行计时。
  - 默认流中的操作只能在任何流中的所有调用都完成之后才开始;在流结束之前，任何后续操作都不能开始

### 7.2 GPU计时器

~~~C++
cudaEvent_t start, stop;
float time;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start, 0 );
kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y, NUM_REPS);
cudaEventRecord( stop, 0 );
cudaEventSynchronize( stop );
cudaEventElapsedTime( &time, start, stop );
cudaEventDestroy( start );
cudaEventDestroy(stop);
~~~

### 7.3 带宽

- 带宽可能会受到存储数据的内存选择、数据的布局方式和访问顺序以及其他因素的显著影响。

#### 7.3.1 理论带宽计算

- 理论内存带宽峰值：HBM2 (double data rate) RAM with a memory clock rate of 877 MHz and a 4096-bit-wide memory interface.
  - $(0.877*10^9*(4096/8)*2)/10^9 = 898GB/s$

#### 7.3.2 有效带宽计算

- 有效内存带宽：$((B_{read} + B_{write})/10^9)/time$

#### 7.3.3 Visual Profiler的吞吐指标

- 如下：

  - Requested Global Load Throughput

  - Requested Global Store Throughput

  - Global Load Throughput

  - Global Store Throughput

  - DRAM Read Throughput

  - DRAM Write Throughput

- 请求的全局加载吞吐量(Requested Global Load Throughput)和请求的全局存储吞吐量(Requested Global Store Throughput)值表示**内核请求的全局内存吞吐量**，因此对应于**有效带宽计算**。
- 对于全局内存访问，**实际吞吐量**由全局加载报告吞吐量(Global Load Throughput)和全局存储吞吐量值(Global Store Throughput)。
- Global Load/Store Throughput反映了代码离硬件极限的距离。

- **将有效带宽或请求带宽与实际带宽进行比较，可以很好地估计由于内存访问的次优合并所浪费的带宽**

- 对于全局内存访问，请求内存带宽与实际内存带宽的比较由全局内存加载效率和全局内存存储效率指标来报告(Global Memory Load Efficiency and Global Memory Store Efficiency)。

## 8.内存优化

### 8.1 H&D数据传输

- 因为host和device间的PCIe数据传输远慢于device和device mem间的数据传输，所以要**尽可能减少host和device间的数据传输**。

- **中间数据结构应该在设备内存中创建，操作并销毁**。
- 由于每次传输都有开销，**将许多小传输批处理为一个较大的传输**比单独进行每次传输的性能要好得多。
  - 即使这样做需要将非连续的内存区域打包到一个连续的缓冲区中，然后在传输后解包
- 当使用**锁页主机内存**时，主机和设备之间可以获得更高的带宽

#### 8.1.1 Pinned Mem

- cudaHostAlloc()
- 对于已经预分配的系统内存区域，可以使用cudaHostRegister()动态固定内存
- **固定内存是稀缺资源且分配它是heavyweight操作**。

#### 8.1.2 和计算重叠的异步传输

- cudaMemcpyAsync()
  - 要求pinned mem和stream ID
- 不同流的操作可以交叉执行来隐藏H&D数据传输
- 异步数据传输的时候，可以执行CPU计算

~~~C++
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, 0);
kernel<<<grid, block>>>(a_d);
cpuFunction();
~~~

- 异步数据传输的时候，可以执行GPU计算，但这种重叠需要非默认流，因为使用默认流的内存复制、内存设置函数和内核调用仅在设备上(在任何流中)的所有前面调用完成之后才开始，并且设备上没有任何操作(在任何流中)开始直到结束。

~~~C++
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(otherData_d);
~~~

- 重叠行为的使用场景

~~~C++
// 串行
cudaMemcpy(a_d, a_h, N*sizeof(float), dir);
kernel<<<N/nThreads, nThreads>>>(a_d);
// 数据分批 并行
size=N*sizeof(float)/nStreams;
for (i=0; i<nStreams; i++) {
    offset = i*N/nStreams;
    cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
    kernel<<<N/(nThreads*nStreams), nThreads, 0,
    stream[i]>>>(a_d+offset);
}
~~~

#### 8.1.3 零拷贝

- 零拷贝要求mapped pinned memory。
- **在集成GPUs里，GPU和CPU内存是物理相同的**，可以始终避免冗余拷贝。在分离GPUs里，则只有部分案例下零拷贝是个增益。
- 因为数据不是缓存在GPU上的，映射固定内存应该只读或写一次，读写内存的全局加载和存储应该合并。

~~~C++
float *a_h, *a_map;
...
cudaGetDeviceProperties(&prop, 0);
if (!prop.canMapHostMemory)
	exit(0);
// 启用锁页内存映射
cudaSetDeviceFlags(cudaDeviceMapHost);
cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&a_map, a_h, 0);
kernel<<<gridSize, blockSize>>>(a_map);
~~~

#### 8.1.4 统一虚拟寻址

- 统一虚拟寻址下，主机内存和设备内存共享一个单虚拟地址空间。

- cudaPointerGetAttributes()用来检查统一虚拟寻址下的指针指向位置。
- cudaHostAlloc()获得的pinned主机内存的主机指针和设备指针都一样。
  - 这种情况下，cudaHostGetDevicePointer()就没用了。
- 然而，事后通过cudaHostRegister()固定的主机内存分配将继续拥有不同于其主机指针的设备指针
  - 这种情况下，cudaHostGetDevicePointer()仍然是必要的。
- NVA也是点对点数据传输的必要条件。

### 8.2 设备地址空间

<img src="C:/Users/Lenovo/Desktop/%E9%B1%BC%E5%A7%AC%E7%8E%84%E7%9A%84%E4%B8%9C%E8%A5%BF/Typora%E5%9B%BE%E7%89%87/GPU_Memory_Space.png" style="zoom:50%;" />

#### 8.2.1 合并访问全局内存

- 全局内存的加载和存储应该被设备合并成尽可能少的事务。
- 一个warp线程的并发访问将合并为一些事务，这些事务的数量等于为warp所有线程服务所必需的32字节事务的数量。
- 非对齐顺序全局内存访问会导致多一个事务，但是缓存会减轻吞吐浪费。
- 因此，确保每个缓存行上取回的数据尽可能被使用是很重要的。

~~~C++
__global__ void strideCopy(float *odata, float* idata, int stride)
{
    int xid = (blockIdx.x*blockDim.x + threadIdx.x)*stride;
    odata[xid] = idata[xid];
}
// stride越大内存访问效率越低
~~~

#### 8.2.2 L2 Cache

- 从CUDA 11.0开始，计算能力8.0及以上的设备能够影响L2缓存中数据的持久性。因为L2缓存是在芯片上的，它可能提供更高的带宽和更低的延迟访问全局内存。
- L2里，常访问的数据属于persisting，只访问一次的数据属于streaming。L2可以预留空间给persisting，预留空间没被用persistent访问的时候，streaming和normal数据可以使用。

~~~C++
/* Set aside max possible size of L2 cache for persisting accesses */
cudaGetDeviceProperties(&prop, device_id);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); 

cudaStreamAttrValue stream_attribute; //Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); //Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes; //Number of bytes for persisting accesses.
//(Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio = 1.0; //Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; //Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; //Type of access property on cache miss.
//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
~~~

- if the hitRatio value is 0.6, 60% of the memory accesses in the global memory region [ptr..ptr+num_bytes) have the persisting property and 40% of the memory accesses have the streaming property.

~~~C++
__global__ void kernel(int *data_persistent, int *data_streaming, int dataSize, int
freqSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /*Each CUDA thread accesses one element in the persistent data section
    and one element in the streaming data section.
    Because the size of the persistent memory region (freqSize * sizeof(int)
    bytes) is much smaller than the size of the streaming memory region (dataSize * sizeof(int)
    bytes), data in the persistent region is accessed more frequently*/
    data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize];
    data_streaming[tid % dataSize] = 2 * data_streaming[tid % dataSize];
}

stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(data_persistent);
// Number of bytes for persisting accesses in range 10-60 MB
stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int); 
// Hint for cache hit ratio. Fixed value 1.0
stream_attribute.accessPolicyWindow.hitRatio = 1.0; 
~~~

~~~C++
// 调优 始终保持num_bytes的数据是驻留的
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
stream_attribute.accessPolicyWindow.num_bytes = 20*1024*1024;                                  //20 MB
stream_attribute.accessPolicyWindow.hitRatio  = (20*1024*1024)/((float)freqSize*sizeof(int));  
// Such that up to 20MB of data is resident.
~~~

#### 8.2.3 共享内存（案例待学习）

- 在没有bank conflicts的情况下，片上的共享内存带宽更高，延迟更低。
- 共享内存被划分为n个banks。如果访问的地址横跨n个独立的bank，则可以并行访问。但是，单个bank的多地址访问是冲突的，会串行执行。
  - 硬件会把bank conflict的请求尽可能划分为多个独立的非冲突访问。
- 一个例外，一个warp的多个线程访问共享内存的同一位置会产生广播行为。多个bank的广播会形成一个多播，从内存请求位置们到线程们。
- 5.0及以上算力，一个bank在一个时钟周期里有32bit的带宽，且连续32bits的字会分配在连续的32个bank上。
- 共享内存配合全局内存：
  - 同一block的多个线程访问同一global mem location，可以用共享内存读来减少global mem access。
  - 共享内存还可以通过从全局内存中以合并模式加载和存储数据，然后在共享内存中重新排序来避免非合并内存访问。
- C=AB：

~~~C++
// 未优化
__global__ void simpleMultiply(float *a, float* b, float *c, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
    sum += a[row*TILE_DIM+i] * b[i*N+col];
    }
    c[row*N+col] = sum;
}
// 使用共享内存来提高矩阵乘法中的全局内存加载效率
__global__ void coalescedMultiply(float *a, float* b, float *c, int N){
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    __syncwarp();
    for (int i = 0; i < TILE_DIM; i++) {
    sum += aTile[threadIdx.y][i]* b[i*N+col];
    }
    c[row*N+col] = sum;
}
// 读取额外数据到共享内存来提高
__global__ void sharedABMultiply(float *a, float* b, float *c, int N){
    __shared__ float aTile[TILE_DIM][TILE_DIM];
    bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
    	sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}
~~~

- C=AA^T^

~~~C++
// 全局内存跨步访问的未优化处理
__global__ void simpleMultiply(float *a, float *c, int M){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
    	sum += a[row*TILE_DIM+i] * a[col*TILE_DIM+i];
    }
    c[row*M+col] = sum;
}
// 使用全局内存的合并读取来优化跨步访问的处理
__global__ void coalescedMultiply(float *a, float *c, int M){
    __shared__ float aTile[TILE_DIM][TILE_DIM],
    transposedTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
    a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
    threadIdx.x];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
    	sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}
// 
~~~

- Asynchronous Copy from Global Memory to Shared Memory

~~~C++
template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count)
{
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);
    uint64_t clock_start = clock64();
    for (size_t i = 0; i < copy_count; ++i) {
    	shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
    }
    uint64_t clock_end = clock64();
    atomicAdd(reinterpret_cast<unsigned long long *>(clock),
    clock_end - clock_start);
}
template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count)
{
    extern __shared__ char s[];
    T *shared = reinterpret_cast<T *>(s);
    uint64_t clock_start = clock64();
    //pipeline pipe;
    for (size_t i = 0; i < copy_count; ++i) {
    	__pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x], 
                                &global[blockDim.x * i + threadIdx.x], sizeof(T));
	}
    __pipeline_commit();
    __pipeline_wait_prior(0);
    uint64_t clock_end = clock64();
    atomicAdd(reinterpret_cast<unsigned long long *>(clock),
    clock_end - clock_start);
}
~~~

#### 8.2.4 本地内存

- 本地内存名字来源于其作用域是线程的局部。本地内存和全局内存共用一个块物理内存。
- 本地内存仅用于保存自动变量。这是由nvcc编译器在确定没有足够的寄存器空间来保存变量时完成的。可能被放置在本地内存中的自动变量是会消耗太多寄存器空间的大型结构或数组，以及编译器确定可以动态索引的数组。

#### 8.2.5 纹理内存

- 只读纹理内存会被缓存。所以，一个texture fetch在缓存失效时才会读设备内存，否则会读纹理缓存。
- 纹理缓存针对2D空间局部性进行了优化，因此读取邻近纹理地址的同一warp线程们将获得最佳性能。
- 纹理内存也被设计为具有恒定延迟的streaming fetch;也就是说，缓存命中会减少DRAM带宽需求，但不会减少获取延迟。

- 纹理内存写是顺序一致性不是线性一致性。

#### 8.2.6 常量内存

- 一个设备有64KB的常量内存空间。常量内存空间会被缓存。所以，缓存失效时才会读设备内存，否则会读常量缓存。
-  一个warp里多个threads对常量内存地址访问是串行的。所以，最好是一个warp里的threads都访问同一个地址。

#### 8.2.7 寄存器

- 一般来讲，每条指令访问寄存器消耗的额外时钟周期为零，但由于寄存器写后读依赖和寄存器内存库冲突，可能会导致延迟。
- 编译器和硬件线程调度器将尽可能优化地调度指令，以避免寄存器内存库冲突。应用程序不能直接控制这些银行冲突。
- Each multiprocessor contains thousands of 32-bit registers，但是也可能发生寄存器不足的情况。
  - nvcc -maxrregcount=N
  - --global-- void --launch_bounds--(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) kernelName();

### 8.3 内存分配

- Device memory allocation and de-allocation via cudaMalloc() and cudaFree() are expensive operations, so device memory should be reused and/or sub-allocated by the application.



## 9.运行配置优化

- 保持SM忙碌是高性能的关键。在设计应用程序时，以最大化硬件利用率的方式使用线程和块，并限制阻碍工作自由分发的实践，这一点非常重要。
- 占用率、并行内核执行、系统资源管理是实现上述实践的重要概念。

### 9.1 占用率

- 当一个warp暂停或停止时，执行其他warp是隐藏延迟和保持硬件繁忙的唯一方法。占用率反映了multiprocessor上活跃线程束的数量，对硬件繁忙效率有重要参考意义。
- 占用率：multiprocessor的**当前活跃线程束数量和最大可能活跃线程束数量之比**。
  - 另一个视角：硬件处理在使用的活跃线程束的能力的比例
- 更高的占用率并不总是等同于更高的性能——超过某个点，额外的占用率就不能提高性能。然而，低使用率总是会干扰隐藏内存延迟的能力，从而导致性能下降。
- 每个内核线程需要的资源数，可能会限制最大block size。为了保持对未来硬件和工具包的向前兼容性，并确保至少有一个线程块可以在SM上运行，开发者应该写上--launch_bounds--(maxThreadsPerBlock, minBlockPerMultiprocessor)

#### 9.1.1 计算占用率

- 寄存器是影响占用率的因素之一。寄存器集由驻留在多处理器上的所有线程共享，整个block一起分配。寄存器有限的情况下，一个block用的寄存器越多，可驻留的block就越少。
  - 整个块一起分配意味着如果寄存器数量不能满足最后一个块，就会造成相当程度的占用率浪费。
- nvcc --ptxas options=v描述了每个kernel每个thread占用的寄存器数量。
- NVIDIA provides an occupancy calculator in the form of an **Excel spreadsheet called CUDA_Occupancy_Calculator.xls ** that enables developers to hone in on the optimal balance and to test different possible scenarios more easily.

- CUDA runtime Occupancy API: **cudaOccupancyMaxActiveBlocksPerMultiprocessor**, 动态选择启动参数。

### 9.2 隐藏寄存器依赖

- 当指令使用之前指令写入的寄存器中存储的结果时，就会产生寄存器依赖。在计算能力7.0的设备上，大多数算术指令的延迟通常为4个周期。因此线程在使用算术结果之前必须等待大约4个周期。然而，这种延迟可以通过执行其他warp中的线程完全隐藏

### 9.3 线程和块的启发

- 延迟隐藏和占用依赖于每个多处理器的活动warp数，这是由执行参数和资源(寄存器和共享内存)约束隐式决定的。选择执行参数是在延迟隐藏(占用)和资源利用之间取得平衡的问题。
- grid size选择的主要考虑因素是让GPU繁忙。一个grid的block数应该大于多处理器数量，保证每个multiprocessor都至少有一个block运行。更进一步，一个multiprocessor应该有多个活跃的块，来防止同步API造成多处理器不忙。这也视情况而定，比如block size和shared mem的选择和分配。
  - 为了向前兼容，内核启动的block数应该是数千的。
- 更高的占用率并不总是等同于更高的性能。在某些情况下，使用高度暴露的指令级并行性(ILP)可以在低占用率的情况下完全覆盖延迟。

- block size的选择涉及很多因素，应该实验来决定，但如下规则应该被遵守：

  - block size应该是warp size的倍数
  - 在能保证一个多处理器能有多个并发块的情况下，一个块应该有至少64个线程
  - 每个块的线程数在128到256之间是一个很好的初始范围。

  - 如果延迟是性能瓶颈，应该用多个小block来替代一个大block

### 9.4 共享内存的影响

- 确定性能对占用的敏感性的一种有用技术是通过实验动态分配的共享内存的数量，如执行配置的第三个参数所指定的那样。通过简单地增加这个参数(不修改内核)，就可以有效地减少内核的占用并衡量它对性能的影响。

### 9.5 并行内核执行

- 非默认流的多流可以允许并行内核执行。

### 9.6 多上下文

- CUDA工作发生在特定GPU的进程空间内，称为上下文。
- 上下文封装了该GPU的内核启动和内存分配，以及诸如页表之类的支持结构。上下文在CUDA驱动程序中是显式的API，但完全隐含在CUDA运行时API中，它自动创建和管理上下文。
- 使用CUDA Driver API, CUDA应用程序进程可以为给定的GPU创建多个上下文。如果多个CUDA应用程序进程并发访问相同的GPU，这几乎总是意味着多个上下文，因为一个上下文绑定到一个特定的主机进程，除非使用多进程服务。

- 多上下文共享GPU是时分的，虽然多个上下文(以及它们相关的资源，如全局内存分配)可以在给定的GPU上并发分配，但在任何给定时刻，只有一个上下文可以在该GPU上执行工作。创建额外的上下文会导致每个上下文数据的内存开销和上下文切换的时间开销。此外，当来自多个上下文的工作可以并发执行时，对上下文切换的需求可以降低利用率。
- 因此，最好避免同一个CUDA应用程序中每个GPU有多个上下文。为了帮助实现这一点，CUDA Driver API提供了一些方法来访问和管理每个GPU上的一个特殊上下文，称为primary context。当线程还没有当前上下文时，CUDA运行时隐式地使用这些相同的上下文。

~~~C++
// When initializing the program/library
CUcontext ctx;
cuDevicePrimaryCtxRetain(&ctx, dev);
// When the program/library launches work
cuCtxPushCurrent(ctx);
kernel<<<...>>>(...);
cuCtxPopCurrent(&ctx);
// When the program/library is finished with the context
cuDevicePrimaryCtxRelease(dev);
~~~

## 10.指令优化

- 了解指令是如何执行的，通常可以进行有用的低级优化，特别是在频繁运行的代码中(程序中所谓的热点)。建议在完成所有高级优化之后执行指令优化。

### 10.1 算术指令

- 单精度浮点提供最佳性能，并高度鼓励使用它们。

#### 10.1.1 除和模

- 除和模的开销大。如果n是2的幂，那么$i/n == i >> log_{2}n$，i % n == i &(n-1)

#### 10.1.2 循环计数 signed vs. unsigned

- C语言标准里，无符号整数溢出有很好的定义，但有符号整数溢出是未定义的。因此编译器使用有符号算法可以比使用无符号算法更积极地进行优化。所以，在循环计数里，i应该定义为signed

~~~C++
for (i = 0; i < n; i++) {
	out[i] = in[offset + stride*i];
}
/* 这里stride*i可能会溢出32bit integer，如果用unsigned，溢出语义会阻止编译器进行可能的优化，如长度截断。但如果是signed，编译器有更多的余地来进行优化 */
~~~

#### 10.1.3 Reciprocal Square Root

- The reciprocal square root should always be invoked **explicitly** as rsqrtf() for single precision and rsqrt() for double precision. The compiler optimizes 1.0f/sqrtf(x) into rsqrtf() only when this does not violate IEEE-754 semantics.

#### 10.1.4 其他算术指令

- 编译器有时必须插入转换指令，引入额外的执行周期：
  - 对char或short操作的函数，其操作数通常需要转换为int类型
  - 用作单精度计算输入的双精度浮点常量，如3.1415
- 要避免这种转换，比如，直接3.1415f

#### 10.1.5 小分数幂

- 有特定的函数，又快有准，如：r = rcbrt(rcbrt(x)) 代表 x^1/9^

#### 10.1.6 数学库

- 带两个下划线的数学库函数会直接映射到硬件。它们快但精度低。

#### 10.1.7 与精度相关的编译器标志

- 默认情况下，nvcc编译器生成符合ieee标准的代码，但它也提供了一些选项来生成一些不太准确但更快的代码：
  - -ftz=true(非规范化数字被刷新为零)
  - -prec-div=false(低精度除)
  - -prec-sqrt=false(低精度平方根)

- 另一个更激进的选项是-use_fast_math，它将每个functionName()调用强制转换为等效的__functionName()调用。这使得代码以降低精度和准确性为代价运行得更快。

### 10.2 内存指令

- When accessing uncached local or global memory, there are hundreds of clock cycles of memory latency.
- 如果在等待全局内存访问完成时可以发出足够多的独立算术指令，那么线程调度程序可以隐藏大部分全局内存延迟。但是，最好尽可能避免访问全局内存。

## 11.控制流

### 11.1 分支和分化

- 当控制流依赖于线程ID的时候，应该写一个能使线程束分化数量最少的控制条件。

- 对于只包含少量指令的分支，线程束分化通常会导致边际性能损失。例如，编译器可以使用预测来避免实际的分支。所有指令都被调度，但是每个线程的条件代码或谓词控制哪些线程执行指令。带有假谓词的线程不会写入结果，也不会计算地址或读取操作数。

- Starting with the Volta architecture, Independent Thread Scheduling allows a warp to remain diverged outside of the data-dependent conditional block. An explicit __syncwarp() can be used to guarantee that the warp has reconverged for subsequent instructions.

### 11.2 分支预测

- 有时，编译器可能会使用分支预测来展开循环或优化if或switch语句。在这种情况下，没有warp会分化。程序员还可以使用控制循环展开：

~~~C++
#pragma unroll
~~~

- 使用分支预测时，不会跳过任何执行依赖于控制条件的指令。相反，每个这样的指令都与每个线程的条件代码或谓词相关联，根据控制条件设置为true或false。尽管这些指令中的每一条都被安排执行，但只有带有真实谓词的指令才会实际执行。带有假谓词的指令不会写入结果，也不会计算地址或读取操作数。
- 只有当分支条件控制的指令数量小于或等于某个阈值时，编译器才会用谓词指令替换分支指令。

## 12.部署CUDA应用

- 完成应用程序的一个或多个组件的GPU加速后，就可以将结果与原始预期进行比较。在解决其他热点以提高总加速之前，开发人员应该考虑采用部分并行实现并将其应用到生产环境中。

## 13.理解编程环境

- 程序员应该注意两个版本号。首先是计算能力，其次是CUDA运行时和CUDA驱动程序api的版本号。

- 计算能力描述了硬件的特性，反映了设备支持的指令集以及其他规格。

- cudaGetDeviceProperties()可以获取额外的设备信息，如asyncEngineCount反映了内核执行和数据传输覆盖的可能；canMapHostMemory反映了零拷贝的可能。
- 要针对特定版本的NVIDIA硬件和CUDA软件，请使用-arch、-code和-gencode选项的nvcc。例如，使用warp shuffle操作的代码必须使用-arch=sm_30(或更高的计算能力)进行编译。

- CUDA主机运行时包括：
  - 设备管理
  - 按上下文管理
  - 内存管理
  - 代码模块管理
  - 执行控制
  - 纹理引用管理
  - 与OpenGL和Direct3D的互操作性

- CUDA运行时通过提供隐式初始化、上下文管理和设备代码模块管理，极大地简化了设备管理。由nvcc生成的c++主机代码利用CUDA运行时，因此链接到该代码的应用程序将依赖于CUDA运行时;类似地，任何使用cuBLAS、cuFFT和其他CUDA Toolkit库的代码也将依赖于CUDA运行时，由这些库在内部使用。
- CUDA运行时处理内核加载，在内核启动之前设置内核参数和启动配置。隐式驱动版本检查、代码初始化、CUDA上下文管理、CUDA模块管理(cubin到函数映射)、内核配置和参数传递都由CUDA执行运行时。

- CUDA运行时分为：C风格的函数接口cuda_runtime_api.h和C风格包装成的C++风格的cuda_runtime.h

## 14.CUDA兼容性开发人员指南（待完成）

- CUDA运行时API为开发人员提供了高级c++接口，以简化设备管理，内核执行等。而CUDA驱动程序API提供一种用于针对NVIDIA硬件的应用程序的低级编程接口。

- 





## 15.准备部署



## 16.部署的基础设施工具



## 17.建议和最佳实践

- 性能优化主要围绕以下三个基本策略:
  - **最大化并行执行**
  - **优化内存使用以获得最大内存带宽**
  - **优化指令使用以获得最大指令吞吐**
- 最大化并行执行：
  - 仔细选择每次内核启动的执行配置。
  - 应用程序还应该通过流显式地公开设备上的并发执行，以及最大化主机和设备之间的并发执行，从而在更高级别上最大化并行执行。

- 优化内存使用：

  - 最小化主机和设备之间的数据传输。
  - 内核对全局内存的访问也应该通过最大化使用设备上的共享内存来最小化。
  - 有时，最好的优化甚至可能是从一开始就避免任何数据传输，只需在需要时重新计算数据。

  - 根据最佳内存访问模式组织内存访问，全局内存访问优化很重要，共享内存的bank冲突严重时也值得优化。

- 优化指令使用：

  - 应该避免使用低吞吐量的算术指令。这意味着在不影响最终结果的情况下，可以用精度来换取速度，例如使用intrinsic而不是常规函数，或者使用单精度而不是双精度。
  - 由于设备的SIMT特性，必须特别注意控制流指令。

## 18.nvcc编译器开关

- -maxrregcount=N specifies the maximum number of registers kernels can use at a perfile level.

- --ptxas-options=-v or -Xptxas=-v lists per-kernel register, shared, and constant memory usage.
- -ftz=true(非规范化数字被刷新为零)
- -prec-div=false(低精度除)
- -prec-sqrt=false(低精度平方根)

- -use_fast_math，它将每个functionName()调用强制转换为等效的__functionName()调用。这使得代码以降低精度和准确性为代价运行得更快。