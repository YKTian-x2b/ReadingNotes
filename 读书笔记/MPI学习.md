# MPI学习

## 0.什么是MPI？

- 跨语言通信接口，用于编写并行计算程序。
- 消息传递模型用作并行计算的原因是：
  - 主从模式下，主进程 可以将 task的描述 通过 消息传递模型 发送给 从进程，实现并行。
  - 对等模型下，进程可以通过汇总并行计算结果来完成计算任务，如大规模排序分治为本地排序和归并。




## 1.MPI的六个基础接口

- MPI_Init是MPI程序的第一个调用它完成MPI程序所有的初始化工作所有MPI程序的第一条可执行语句都是这条语句。
  - MPI常量可以在MPI_Init之前。
  - MPI_Initialized可以在MPI_Init调用之前出现，用来判断MPI_Init是否已经执行。
  
- MPI_Finalize是MPI程序的最后一个调用它结束MPI程序的运行它是MPI程序的最后一条可执行语句否则程序的运行结果是不可预知的。
- MPI_Comm_rank这一调用返回调用进程在给定的通信域中的进程标识号。
- MPI_Comm_size这一调用返回给定的通信域中所包括的进程的个数。
- MPI_Send将发送缓冲区中的count个datatype数据类型的数据发送到目的进程目的进程在通信域中的标识号是dest 本次发送的消息标志是tag 使用这一标志就可以把本次发送的消息和本进程向同一目的进程发送的其它消息区别开来。
- MPI_Recv从指定的进程source接收消息并且该消息的数据类型和消息标识和本接收进程指定的datatype和tag相一致接收到的消息所包含的数据元素的个数最多不能超过count。

------

- MPI_Status *status：状态变量是由至少三个域组成的结构类型这三个域分别是MPI_SOURCE，MPI_TAG和MPI_ERROR。还可以有其他附加域。

- 基本数据类型：

|         MPI          |                           C                            |
| :------------------: | :----------------------------------------------------: |
|       MPI_CHAR       |                      signed char                       |
|      MPI_SHORT       |                    signed short int                    |
| MPI_INT~ˈɪntɪdʒə(r)~ |                       signed int                       |
|       MPI_LONG       |                      signed long                       |
|  MPI_UNSIGNED_CHAR   |                     unsigned char                      |
|  MPI_UNSIGNED_SHORT  |                   unsigned short int                   |
|     MPI_UNSIGNED     |                      unsigned int                      |
|  MPI_UNSIGNED_LONG   |                   unsigned long int                    |
|      MPI_FLOAT       |                         float                          |
|      MPI_DOUBLE      |                         double                         |
|   MPI_LONG_DOUBLE    |                      long double                       |
|       MPI_BYTE       |                    8 bits / 1 byte                     |
|      MPI_PACKED      | data packed or unpacked with MPI_Pack() / MPI_Unpack() |

- 在MPI中，类型匹配要求：宿主语言的类型和通信操作指定的类型要匹配；发送方和接收方的类型匹配。
- MPI的数据转换包含两种含义：数据类型的转换和数据表示的转换。
  - 数据类型转换：MPI严格要求类型匹配，所以不存在数据类型转换的问题。
  - 数据表示转换：即异构系统通信时，改变值的二进制表示。大端字节序和小端字节序、浮点数的位数等。
    - 无类型数据发送的是二进制表示，所以是正确的。
    - 有类型数据，发送方和接收方的数据表示可能不同，编码译码的值也可能不同。

- Message的组成：MPI消息包括信封和数据两个部分，信封指出了发送或接收消息的对象及相关信息，而数据是本消息将要传递的内容。信封和数据又分别包括三个部分可以用一个三元组来表示：
  - 信封 <源/目，标识，通信域>
  - 数据 <起始地址，数据个数，数据类型>
- Tag用于区分发送方发送给同一个接收方的多条相同类型的数据。
- 接收者通过信封的三个字段判断是否接收。MPI_ANY_SOURCE和MPI_ANY_TAG可以豁免对应的匹配，类似于通配符。
- MPI通信域包括两部分：进程组和通信上下文。
  - 进程组即所有参加通信的进程的集合，如果一共有N个进程参加通信，则进程的编号从0到N-1。 
  - 通信上下文提供一个相对独立的通信区域，不同的消息在不同的上下文中进行传递，不同上下文的消息互不干涉。
- 用户可以在原有的通信域的基础上定义新的通信域通信域为库和通信模式提供一种重要的封装机制他们允许各模式有其自己的独立的通信域和它们自己的进程计数方案。

> 死锁的通信顺序

~~~fortran
! pro1的send在等pro1的recv，pro1的recv在等pro2的send，pro2的send在等pro2的recv，pro2的recv在等pro1的send。环路等待死锁。
CALL MPI_COMM_RANK(comm, rank, ierr)
IF (rank.EQ.0) THEN
CALL MPI_RECV(recvbuf, count, MPI_REAL, 1, tag, comm, status, ierr)
CALL MPI_SEND(sendbuf, count, MPI_REAL, 1, tag, comm, ierr)
ELSE IF( rank .EQ. 1)
CALL MPI_RECV(recvbuf, count, MPI_REAL, 0, tag, comm, status, ierr)
CALL MPI_SEND(sendbuf, count, MPI_REAL, 0, tag, comm, ierr)
END IF
~~~

> 不安全的通信顺序

~~~fortran
! 如果某个进程要发送的数据大于系统缓冲区，则该方法不安全
CALL MPI_COMM_RANK(comm, rank, ierr)
IF (rank.EQ.0) THEN
CALL MPI_SEND(sendbuf, count, MPI_REAL, 1, tag, comm, ierr)
CALL MPI_RECV(recvbuf, count, MPI_REAL, 1, tag, comm, status, ierr)
ELSE rank .EQ.1
CALL MPI_SEND(sendbuf, count, MPI_REAK, 0, tag, comm, status, ierr)
CALL MPI_RECV(recvbuf, count, MPI_REAL, 0, tag, comm, status, ierr)
END IF
~~~

> 安全的通信顺序

~~~Fortran
! send的同时，对应的pro发起recv 避免了缓冲区的使用
CALL MPI_COMM_RANK(comm, rank, ierr)
IF (rank.EQ.0) THEN
CALL MPI_SEND(sendbuf, count, MPI_REAL, 1, tag, comm, ierr)
CALL MPI_RECV(recvbuf, count, MPI_REAL, 1, tag, comm, status, ierr)
ELSE rank .EQ. 1
CALL MPI_RECV(recvbuf, count, MPI_REAL, 0, tag, comm, status, ierr)
CALL MPI_SEND(sendbuf, count, MPI_REAL, 0, tag, comm, ierr)
END IF
~~~



## 2.并行程序的两种基本模式

- 并行程序设计模式：对等模式和主从模式。
  - SPMD实现对等模式是比较容易理解的。
  - 但从实践表明：主从模式也完全可以用SPMD来高效实现。


### 2.1 对等模式

- 对等模式的各个进程完成的任务基本一致，所以安全的通信是重点。
- 安全通信是指，在考虑到内存缓冲区不足以存放一次Send的数据量的情况下，是否可以成功发送。也就是说，基本上同一时间需要一个Send对应一个Recv。
- 虚拟进程(MPI_PROC_NULL)：不存在的假想进程，主要充当真实进程通信的源或目的进程，为了编程方便。一个真实进程向虚拟进程MPI_PROC_NULL发送消息时会立即成功返回一个真实进程从虚拟进程MPI_PROC_NULL的接收消息时也会立即成功返回并且对接收缓冲区没有任何改变。

### 2.2 主从模式

- 主从模式的主进程和从进程完成的任务不同，就是if-else分别写不同的逻辑。



## 3.四种通信模式

- MPI共有**四种通信模式**：标准通信模式、缓存通信模式、同步通信模式和就绪通信模式。

- 四种通信模式的划分标准：
  - 是否需要 **对发送的数据进行缓存**
  - 是否 **只有当接收调用执行后才可以执行发送操作**
  - **什么时候发送调用可以正确返回**
  - **发送调用正确返回是否意味着发送已完成**，即发送缓冲区是否可以被覆盖，发送数据是否已到达接收缓冲区。

| 通信模式                      | 发送      | 接收     |
| ----------------------------- | --------- | -------- |
| 标准通信模式 standard mode    | MPI_Send  | MPI_Recv |
| 缓存通信模式 buffered mode    | MPI_Bsend |          |
| 同步通信模式 synchronous mode | MPI_Ssend |          |
| 就绪通信模式 ready mode       | MPI_Rsend |          |

- 标准通信模式下，是否缓存要发送的消息，取决于MPI。
  - 如果 **MPI决定缓存则发送操作一定可以正确返回**，不要求接收操作收到发送的数据。
  - 如果 **MPI不缓存数据**，则只有接收调用被执行且 **数据完全到达接收缓冲区时，发送操作才算完成**。
  - 非阻塞发送的发送调用可以在发送操作没有完成的情况下，正确返回。

- 缓存通信模式下，用户直接对通信缓冲区进行申请、使用和释放。
  - 不管接收操作是否启动，发送操作都可以执行，但是在发送消息之前**必须有缓冲区可用，有则正确返回**，否则失败返回。
  - *只有当缓冲区中的消息发送出去后才可以释放该缓冲区*。
  - 对于非阻塞发送，正确退出并不意味着缓冲区可以被其它的操作任意使用。但阻塞发送返回后其缓冲区是可以重用的。
  - 用户自己申请(malloc)缓冲区，attach挂载，detach卸载，free释放掉。

- 同步通信模式下，通信的开始不依赖于接收进程相应的接收操作是否已经启动。但是，**同步发送却必须等到相应的接收进程开始后才可以正确返回**。
  - 同步发送返回后意味着发送缓冲区中的数据已经全部被系统缓冲区缓存并且已经开始发送。
  - 因此，当 *同步发送返回后发送缓冲区可以被释放或重新使用*。
  - 应用程序缓冲区(也称用户缓冲区)是保存要发送的信息或要接收信息的位置的缓冲区。应用程序缓冲区是传递给MPI通信调用的缓冲区。
  - 系统缓冲区(也称内部缓冲区)是MPI运行时系统的一部分，并由MPI运行时系统管理，并且对应用程序代码不可见。在许多MPI实现和互连中，内部缓冲区在程序地址空间中分配，并计入程序内存使用量。不必在内核或应用程序空间之外的其他位置分配系统缓冲区。
  - 这里的同步可能是指消息的实际发送和实际接收操作是同步的。
  
- 就绪通信模式下，**只有当接收进程的接收操作已经启动时，才可以在发送进程启动发送操作**，否则当发送操作启动而相应的接收还没有启动时发送操作将出错。
  - *对于非阻塞发送操作的正确返回并不意味着发送已完成，但对于阻塞发送的正确返回则发送缓冲区可以重复使用*。

> 同步和异步、阻塞和非阻塞

- 阻塞和非阻塞的区别在于：发起请求后，到获得数据（响应）前，是否挂起进程。
- 同步和异步的区别在于：发起请求后，到获取数据（响应）前，是否需要轮询内核的接收情况。



## 4.常见错误

- 不要在MPI_Init前和MPI_Finalize后写可执行程序代码。
  - 这些位置的代码如何执行，在MPI标准中是未定义的。

- MPI_Send和MPI_Recv的顺序问题：
  - 一方在发送时，另一方应该处于接收状态。
  - 成对的交互发送和接收，鼓励用Sendrecv替代。
  - 鼓励用MPI_Isend和MPI_Irecv来替代对应的阻塞操作。
  - 用MPI_Buffer_attach来显示分配用户自己的存储空间。
- 接收缓冲区溢出



## 5.非阻塞通信

- 非阻塞式编程要注意调用返回并不代表动作执行结束。如果出现异常值，一定想到是否调用了MPI_Wait。

### 5.0 非阻塞通信简介

- 针对一个循环中重复执行通信的优化，MPI引入了重复非阻塞通信方式。

|    通信模式    |      发送      |     接收      |
| :------------: | :------------: | :-----------: |
|   非阻塞标准   |   MPI_Isend    |   MPI_Irecv   |
|   非阻塞缓存   |   MPI_Ibsend   |               |
|   非阻塞同步   |   MPI_Issend   |               |
|   非阻塞就绪   |   MPI_Irsend   |               |
| 重复非阻塞标准 | MPI_Send_init  | MPI_Recv_init |
| 重复非阻塞缓存 | MPI_Bsend_init |               |
| 重复非阻塞同步 | MPI_Ssend_init |               |
| 重复非阻塞就绪 | MPI_Rsend_init |               |

- MPI提供了检测和完成非阻塞接收和发送情况的方法，可以检测一个、多个和全部。

| 非阻塞通信数量 | 检测         | 完成         |
| :------------: | ------------ | ------------ |
| 一个非阻塞通信 | MPI_Test     | MPI_Wait     |
|    任意一个    | MPI_Testany  | MPI_Waitany  |
|    一到多个    | MPI_Testsome | MPI_Waitsome |
|      所有      | MPI_Testall  | MPI_Waitall  |

- 只要消息信封相吻合并且符合有序接收的语义约束，任何形式的发送和任何形式的接收都可以匹配。

### 5.1 非阻塞通信完成

- 非阻塞通信需要专门的通信语句来完成或检测，完成调用不因非阻塞通信形式的改变而改变。非阻塞 **完成调用** 结束后，非阻塞通信完成。
- MPI_Wait以非阻塞通信对象为参数一直等到与该非阻塞通信对象相应的非阻塞通信完成后才返回，同时释放该阻塞通信对象。
- MPI_Test调用可以直接返回当前通信完成情况，flag=true/false。同样会释放已完成的通信对象。

~~~c++
// array_of_requests中任一通信对象(request)完成，释放该对象并返回。idx指出了该对象在array中的索引
int MPI_Waitany(int count, MPI_Request array_of_requests[], int *indx, MPI_Status *status);
// 只要一个或多个通信完成，就返回
int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);
// 全部通信完成，返回
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);
~~~

~~~C++
int MPI_Testany(int count, MPI_Request array_of_requests[], int *indx, int *flag,
                MPI_Status *status);
int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount,
                 int array_of_indices[], MPI_Status array_of_statuses[]);
int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag,
                MPI_Status array_of_statuses[]);
~~~

### 5.2 非阻塞通信对象

- **MPI_Request**：使用非阻塞通信对象可以识别非阻塞通信操作的各种特性，例如：发送模式，和它联结的通信缓冲区，通信上下文，用于发送的标
  识和目的参数或用于接收的标识和源参数。此外非阻塞通信对象还存储关于这个挂起通信操作状态的信息。

- MPI_Cancel：允许取消已调用的非阻塞通信，该调用会立即返回，但不一定真的取消通信。
  - 当通信已经开始，它会正常完成，不受cancel的影响。
  - 当通信未开始，则取消通信，释放通信占用的资源。
  - 此外，cancel不负责释放非阻塞通信对象(request)，必须调用查询或完成操作来释放。

- MPI_Request_free：当程序员能够确认一个非阻塞通信操作完成时，可以直接调用MPI_Request_free，将该对象所占用的资源释放。
  - 一旦释放，request对象就变为MPI_REQUEST_NULL，任何其他调用将无法访问request对象。
  - 如果与该非阻塞通信对象相联系的通信还没有完成，则该对象的资源并不会立即释放，它将等到该非阻塞通信结束后再释放。因此非阻塞通信对象的释放并不影响该非阻塞通信的完成。
  - MPI_Wait和MPI_Test不会将request对象变为MPI_REQUEST_NULL，只是置他们于非激活状态。

### 5.3 消息到达检查

- MPI提供MPI_Probe和MPI_Iprobe，用于在不实际接收消息的情况下，检查给定的消息是否达到。
- MPI_Iprobe：调用该函数时，如果有消息可被接收，且该消息的消息信封和函数参数信封相匹配，则返回flag = true，否则返回false。
  - status和调用MPI_Recv得到的status相同。
  - status中包含source、tag和消息长度。
- MPI_Probe：只有找到一个匹配的消息到达时才返回。

### 5.4 非阻塞通信有序接收的语义约束

- 不管是阻塞通信还是非阻塞通信，都保持顺序接收的语义约束。即，根据程序的书写顺序先发送的消息一定被先匹配的接收调用接收，若
  在实际运行过程中后发送的消息先到达，它也只能等待。

### 5.5 非阻塞通信和重复非阻塞通信

- 为了实现计算与通信的最大重叠，一个通用的原则就是：“尽早开始通信，尽晚完成通信“。在开始通信和完成通信之间进行计算。
- 这样就有可能有更多的计算任务可以和通信重叠，也使通信可以在计算任务执行期间完成而不需要专门的等待时间。
- 如果一个通信会被重复执行，比如循环结构内的通信调用，可以使用MPI_*send_init函数来降低不必要的通信开销。重复通信都是非阻塞的。
- 重复通信范式：
  - 通信的初始化，比如MPI_SEND_INIT
  - 启动通信MPI_START
  - 完成通信MPI_WAIT
  - 释放查询对象MPI_REQUEST_FREE

- 重复非阻塞通信下，当不再进行通信时，**必须**显式的MPI_Request_free掉非阻塞通信对象。
- MPI_Wait和MPI_Test会使非阻塞通信对象处于非激活状态，对象未被释放。MPI_Start和MPI_Startall会激活该对象。



## 6.组通信

- 组通信就是一个进程组中所有进程都参与通信。组通信在各个不同进程的调用形式完全相同。
- 组通信和点对点通信共用一个通信域，MPI保证组通信调用产生的消息不会和点对点调用产生的消息混淆。
- 组通信的三个功能：通信、同步和计算。
  - 同步：当进程完成同步调用后，可以保证所有进程都已执行同步点前的操作。
  - 计算：各进程收发消息，执行指定的计算操作，最后将结果汇总到指定的缓冲区。

### 6.1 消息通信功能

- 分为多对一通信、一对多通信和多对多通信。
  - 多：组内所有进程。

- 一对多通信：广播(MPI_Bcast)、散发([MPI_Scatterv](#MPI_Scatterv))。
- 多对一通信：收集([MPI_Gatherv](#MPI_Gatherv))。
- 多对多通信：组收集([MPI_Allgatherv](#MPI_Allgatherv))、全互换([MPI_Alltoallv](#MPI_Alltoallv))。

### 6.2 同步

- [MPI_BARRIER](#MPI_Barrier)阻塞所有的调用者直到所有的组成员都调用了它各个进程中这个调用才可以返回。

### 6.3 归约

- 归约操作必须是可结合的，MPI定义的归约是可交换的，用户自定的归约可以是不可交换的。

- 预定义的归约操作：

|    函数    |          作用           |
| :--------: | :---------------------: |
|  MPI_MAX   |         最大值          |
|  MPI_MIN   |         最小值          |
|  MPI_SUM   |          求和           |
|  MPI_PROD  |          求积           |
|  MPI_LAND  | 逻辑与（boolean level） |
|  MPI_BAND  |   按位与（bit level）   |
|  MPI_LOR   |         逻辑或          |
|  MPI_BOR   |         按位或          |
|  MPI_LXOR  |        逻辑异或         |
|  MPI_BXOR  |        按位异或         |
| MPI_MAXLOC |    最大值且相应位置     |
| MPI_MINLOC |    最小值且相应位置     |

- 组归约MPI_ALLREDUCE：相当于组中每一个进程都作为ROOT分别进行了一次归约操作，每个进程都有归约结果。
- 归约并发散MPI_REDUCE_SCATTER：将归约结果分散到组内的所有进程中去。
- 扫描MPI_SCAN：每一个进程都对排在它前面的进程（以及它）进行归约操作。
- MPI_MAXLOC和MPI_MINLOC会返回第一个全局最值及其进程序列号。但需要提供表示这个值对的参数类型。

|        名字         |       描述       |
| :-----------------: | :--------------: |
|    MPI_FLOAT_INT    |    float和int    |
|   MPI_DOUBLE_INT    |       ...        |
|    MPI_LONG_INT     |       ...        |
|      MPI_2INT       |       ...        |
|    MPI_SHORT_INT    |       ...        |
| MPI_LONG_DOUBLE_INT | long double和int |

~~~C++
// MPI_FLOAT_INT和下列类型定义等价
// [WARNING] yaksa: 1 leaked handle pool objects
int block[2];
MPI_Aint disp[2];
MPI_Datatype type[2];
MPI_Datatype *mpi_float_int;
MPI_Init(&argc, &argv);
type[0] = MPI_FLOAT;
type[1] = MPI_INT;
disp[0] = 0;
disp[1] = sizeof(float);
block[0] = 1;
block[1] = 1;
MPI_Type_struct(2, block, disp, type, mpi_float_int);
MPI_Finalize();
~~~

- 自定义归约操作：[MPI_Op_create](#MPI_Op_*)。
- 释放自定义归约操作：[MPI_Op_free](#MPI_Op_*)。
- 自定义数据结构：

~~~C++
MPI_Datatype ctype;
MPI_Type_contiguous(2, MPI_DOUBLE, &ctype);
MPI_Type_commit(&ctype);
~~~





## 7.不连续数据发送

- 处理不连续数据的两种方式：派生数据类型（用户自定义新的数据类型）、数据打包和解包（不连续数据打包到连续区域，打包的连续数据解包到不连续区域）。

### 7.1 新数据类型的定义

- 连续：[MPI_Type_contiguous](#MPI_Type_contiguous) 多个旧数据类型连续。
- 向量：[MPI_Type_vector](#MPI_Type_vector) 多个块，每个块都连接相同数量的旧数据类型，块间空间是旧数据类型extent的倍数。
- 索引：[MPI_Type_indexed](#MPI_Type_indexed) 多个块，每个块包含不同数量的旧数据类型，块偏移是旧数据类型extent的倍数。
- 结构体：[MPI_Type_struct](#MPI_Type_create_struct) 允许各个块包含不同类型数据。
- 新类型提交和释放：[MPI_Type_commit && MPI_Type_free](#MPI_Type_com_free) 新定义的数据类型在使用之前必须先递交给MPI系统。free调用可以释放已提交的数据类型。
  - 一个递交后的数据类型可以作为一个基本类型，用在数据类型生成器中产生新的数据类型。
  - 释放一个数据类型不会影响根据该数据类型定义的其他数据类型。


### 7.2 地址函数

- [MPI_Get_address](#MPI_Get_address)：可以返回某一个变量在内存中相对于预定义的地址MPI_BOTTOM的偏移地址。

### 7.3 与数据类型有关的Func

- MPI_Type_extent：返回一个数据类型的跨度extent，单位是字节。
- MPI_Type_size：返回一个数据类型有用部分所占空间大小，单位是字节。
  - 空隙 + 有用空间 = 跨度。

- MPI_Get_count：返回接收操作收到的 指定数据类型的 数据个数。
- MPI_Get_elements：返回接收操作收到的 基本数据类型的 数据个数。

### 7.4 上下界标记类型

- 上界标记类型MPI_UB、下界标记类型MPU_LB。
- MPI_Type_ub和MPI_Type_lb：返回上/下界距离origin的字节偏移量。

- [MPI_Type_get_extent](#MPI_Type_get_extent)：上述函数和数据类型已废弃，用该函数替代。返回数据类型的下界和跨度。

### 7.5 打包和解包

- [MPI_Pack](#MPI_Pack)和[MPI_Unpack](#MPI_Unpack)操作是为了发送不连续的数据，发送前显式地把数据包装在一个连续的缓冲区，接收后从连续缓冲区解包。
- MPI_PACKED：打包单元的数据类型。以该类型发送，可以用任意数据类型接收，只要和实际类型匹配。以任意类型发送，都可以用该类型接收。
  - 接收到的MPI_PACKED类型，需要连续调用MPI_Unpack来解包。
- MPI_Pack_size：返回参数个数个参数类型需要的空间，单位是字节。



## 8.MPI的进程组和通信域

- 通信域包括 通信上下文、进程组、虚拟处理器拓扑、属性等内容。
  - 通信域分为 组内通信域 和 组间通信域。
- 进程组：不同进程的有序集合。组内每个进程和一个整数rank联系。rank从0开始递增。
  - MPI_GROUP_EMPTY是空组的有效句柄
  - MPI_GROUP_NULL是无效句柄
- 通信上下文：对通信空间进行划分。一个上下文所发的消息不能被另一个上下文接收。允许集合操作独立于点对点操作。
  - 不是显式的MPI对象
- MPI_Init调用时，会产生一个预定义组内通信域MPI_COMM_WORLD，包含所有进程。还有一个MPI_COMM_SELF，是仅包含自身的通信域。
- 可以通过MPI_COMM_GROUP访问MPI_COMM_WORLD对应的通信组。

### 8.1 进程组管理

- MPI_Group_size：返回进程组包含的进程个数。
- MPI_Group_rank：返回调用进程在进程组中的rank。如果不在进程组中，返回MPI_UNDEFINED。
- MPI_Group_translate_ranks：Group1中n个进程在Group2中对应的编号。
- MPI_Group_compare：对比两个进程组。
  - 进程及编号完全相同==MPI_IDENT~(identical)~；
  - 进程相同编号不对应==MPI_SIMILAR；
  - 否则MPI_UNEQUAL。
- MPI_Comm_group：指定通信域包含的进程组。
- MPI_Group_union：两个进程组的并集。
- MPI_Group_intersection：两个进程组的交集。
- MPI_Group_difference：两个进程组的差集。
- MPI_Group_incl：将已有进程组中的n个进程按ranks的索引顺序组成一个新的进程组。
- MPI_Group_excl：将已有进程组中的n个进程删除，形成新的进程组。
- MPI_Group_range_incl：将已有进程组中的n组由 三元组数组ranges 指定的进程组成一个新的进程组。
- MPI_Group_range_excl：将已有进程组中的n组由 三元组数组ranges 指定的进程删除后，形成一个新的进程组。
- MPI_Group_free：释放一个已有进程组，置句柄group为MPI_GROUP_NULL，任何正在使用此组的操作将正常完成。

### 8.2 通信域管理

- MPI_Comm_size：通信域包含的进程个数。
- MPI_Comm_rank：返回调用进程在通信域中的rank。
- MPI_Comm_compare：对比两个通信域。
  - comm1和comm2是同一个对象的句柄==MPI_IDENT；
  - 进程组成员及编号完全相同== MPI_CONGRUENT~（全等）~；
  - 进程组成员相同编号不对应==MPI_SIMILAR；
  - 否则MPI_UNEQUAL。
- MPI_Comm_dup：复制通信域，新通信域拥有新的上下文，与旧域相同的进程组和缓冲信息。
- MPI_Comm_create：创建指定通信域进程组子集group对应的新通信域，新域拥有新的上下文。
- [MPI_Comm_split](#MPI_Comm_split)： 根据color划分指定通信域，新域中的进程编号的相对大小同key的相对大小。
- MPI_Comm_free：释放指定通信域，将comm置为MPI_COMM_NULL。任何使用此通信域的挂起操作都会正常完成。

### 8.3 组间通信域

- 组间通信域是一种特殊的通信域，该通信域包括两个进程组，通过组间通信域实现这两个不同进程组内进程之间的通信。
  - 一般把调用进程所在的进程组叫做本地组，而把另一个组叫做远程组。
- MPI_Comm_test_inter：判断给定通信域是组内通信域还是组间通信域。
- MPI_Comm_remote_size：返回组间通信域内远程进程组的进程个数。
- MPI_Comm_remote_group：返回组间通信域中的远程进程组。
- [MPI_Intercomm_create](#MPI_Intercomm_create)：用两个通信域创建一个组间通信域。每个进程都需要提供本地组内通信域及本地特定进程标识local_leader，以及远程对等通信域及remote_leader。
  - 一般地，用MPI_COMM_WORLD的复制品来做peer_comm。
- MPI_Intercomm_merge：将一个组间通信域的两个通信域合并成一个组内通信域。

### 8.4 属性信息

- MPI提供类似于cache的手段，允许一个应用将任意的信息（属性），附加到通信域上。属性对于进程是本地的，只专属于它依附的通信域。
- 关于属性的调用包括：
  - 存储或查询属性值。
  - 生成一个关键字值，用户指定回调函数，通过回调函数通知应用程序通信域被破坏/复制。

- [MPI_Keyval_create](#MPI_Keyval_create)：创建一个属性关键字，用于和属性绑定在本地定义的任一通信域上。关键字是个整数，本地唯一。
  - 用MPI_Comm_dup复制通信域时，copy_fn被唤醒。
  - 用MPI_Comm_free释放通信域或MPI_Attr_delete删除属性时，delete_fn被唤醒。

- MPI_Keyval_free：释放属性关键字，将keyval置为MPI_KEY_INVALID。
- MPI_Attr_put：设置指定关键字的属性值，绑定到指定通信域上。
- MPI_Attr_get：通过关键字获得属性值。
  - 没有关键字，则调用出错。
  - 有关键字，但没在comm上，则flag=false。
  - 有关键字，且在comm上，flag=true，属性值可获取。

- MPI_Attr_delete：删除关键字对应的属性值。




## 9.虚拟进程拓扑

- 虚拟拓扑就是进程的逻辑排列。拓扑是针对组内通信域的，不能附加在组间通信域上。
- 常见的两种拓扑：笛卡尔(Cartesian)拓扑和图拓扑，分别用来表示简单规则的拓扑和更通用的拓扑。简单调用对比：

| 操作         | 笛卡尔拓扑      | 图拓扑            |
| ------------ | --------------- | ----------------- |
| 创建         | MPI_Cart_create | MPI_Graph_create  |
| 获得维数     | MPI_Cartdim_get | MPI_Graphdims_get |
| 获得拓扑信息 | MPI_Cart_get    | MPI_Graph_get     |
| 物理映射     | MPI_Cart_map    | MPI_Graph_map     |

### 9.1 笛卡尔拓扑

- [MPI_Cart_create](#MPI_Cart_create)：创建一个基于oldcomm的新笛卡尔拓扑通信域。
- MPI_Dims_create：根据用户指定的总维数ndims和总进程数nnodes，帮助用户选择每一维的进程个数。用户可以提前在dims里指定进程数，划分算法只修改0值。
- MPI_Topo_test：返回通信域的拓扑类型。MPI_CART、MPI_GRAPH和MPI_UNDEFINED。
- MPI_Cart_get：返回通信域的拓扑信息。包括当前进程的笛卡尔坐标coords。
- MPI_Cart_rank：返回卡氏坐标对应的一维线性坐标。
- MPI_Cart_coords：返回一维线性坐标对应的卡氏坐标。
- MPI_Cartdim_get：返回指定笛卡尔结构的维数ndims。
- MPI_Cart_shift：获得调用进程 向dir维平移disp单位 得到的左右邻居，可以是MPI_PROC_NULL。
- MPI_Cart_sub：将指定通信域划分为不同的子通信域，remain_dims指出保留的维。
- MPI_Cart_map：尽可能为当前进程计算一个优化的映射位置，并进行进程重排序。

### 9.2 图拓扑

- [MPI_Graph_create](#MPI_Graph_create)：创建一个有nnodes、index、edges定义的图。
- MPI_Graphdims_get：返回图的节点数和边数。
- MPI_Graph_get：获取图信息。
- MPI_Graph_neighbors_count：获取指定rank的邻居个数。
- MPI_Graph_neighbors：获取指定rank的邻居。
- MPI_Graph_map：计算一个优化的映射位置，返回映射后的坐标newrank。



## 10.MPI的错误处理

- 句柄：标识资源/服务/对象的一个整数。类似ID。
- [MPI_Errhandler_create](#MPI_Errhandler)：将用户例程function向MPI注册，作为MPI异常句柄。返回的errhandler作为指向该例程的句柄。
- MPI_Errhandler_set：将指定错误句柄和给定的通信域相联系。
- MPI_Errhandler_get：返回与通信域相联系的错误句柄。
- MPI_Errhandler_free：释放错误句柄，并直句柄为MPI_ERRHANDLERNULL。
  - 实际的释放操作是在所有与其相联的通信域都释放后进行。
- MPI_Errhandler_string：返回与错误代码相联的错误字符串。
  - 调用前先为该字符串申请至少MPI_MAX_ERROR_STRING字符长的空间。
- MPI_Errhandler_class：返回错误代码相联的错误类。有效的错误类如下：

|      错误码      |       含义       |
| :--------------: | :--------------: |
|   MPI_SUCCESS    |      无错误      |
|  MPI_ERR_BUFFER  |  无效缓冲区指针  |
|  MPI_ERR_COUNT   |   无效计数参数   |
|   MPI_ERR_TYPE   | 无效数据类型参数 |
|   MPI_ERR_ TAG   |   无效标识参数   |
|   MPI_ERR_COMM   |    无效通信域    |
|   MPI_ERR_RANK   |    无效标识数    |
| MPI_ERR_REQUEST  |   无效请求句柄   |
|   MPI_ERR_ROOT   |      无效根      |
|  MPI_ERR_GROUP   |      无效组      |
|    MPI_ERR_OP    |     无效操作     |
| MPI_ERR_TOPOLOGY |     无效拓扑     |
|   MPI_ERR_DIMS   |    无效维参数    |
|   MPI_ERR_ARG    | 其它无效种类参数 |
| MPI_ERR_UNKNOWN  | 不知道原因的错误 |
| MPI_ERR_TRUNCATE | 接受被截断的消息 |
|  MPI_ERR_OTHER   |    其它的错误    |
|  MPI_ERR_INTERN  |   内部MPI错误    |
| MPI_ERR_LASTCODE |  最后标准错误码  |



## 11.MPI2-动态进程管理



## 12.MPI2-远程存储访问



## 13.MPI2-并行I/O











## 100.API

> API说明

- 参数

  - IN：仅用作输入

  - OUT：仅用作输出

  - INOUT：既作为输入又作为输出。或对于并行执行的一部分进程是IN，另一部分是OUT。

- 如果某参数对一部分进程没有意义，则传递任意值都行。
- 在MPI中OUT或INOUT类型的参数不能被其它的参数作为别名使用。

> MPI_Reduce

~~~C++
int MPI_Reduce(
void *input_data, /*指向发送消息的内存块的指针 */
void *output_data, /*指向接收（输出）消息的内存块的指针 */
int count，/*数据量*/
MPI_Datatype datatype,/*数据类型*/
MPI_Op operator,/*规约操作*/
int dest，/*要接收（输出）消息的进程的进程号*/
MPI_Comm comm);/*通信器，指定通信范围*/

// MPI_COMM_WORLD组中0号进程的recSum内存区接收发送来的sum内存里的1个LONG_DOUBLE，并作MPI_SUM规约。
MPI_Reduce(&sum, &recvSum, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
~~~

> MPI_Get_processor_name和MPI_Get_version

~~~C++
// MPI_MAX_PROCESSOR_NAME = 128
MPI_Get_processor_name(processor_name, &namelen);
// MPI版本号
int MPI_Get_version(int* version, int* subversion);
~~~

> MPI_Wtime 返回从过去某时刻到调用时所经历的时间，是浮点数表示的秒数。

~~~C++
double MPI_Wtime(void);
// 获取
double t1 = MPI_Wtime();
// 返回MPI_Wtime的精度，单位是s。可以认为是一个时钟滴答占用的时间。
double MPI_Wtick();
~~~

> MPI_Barrier 							<a name="MPI_Barrier"></a>

~~~C++
// 用于一个通信Group中所有进程的同步，调用函数时进程将处于等待状态，直到Group中所有进程都调用该函数后才继续执行。
int MPI_Barrier(MPI_Comm comm)
~~~

> MPI_Abort 在遇到不可恢复的严重错误时调用，以退出MPI程序的执行。

~~~C++
// errorcode 返回到所嵌环境的错误码
int MPI_Abort(MPI_Comm comm, int errorcode);
~~~

> MPI_Sendrecv* 捆绑发送

~~~C++
// 执行一个阻塞的发送和接收，接收和发送使用同一个通信域，但是可能使用不同的标识
// 发送缓冲区和接收缓冲区必须分开，他们可以是不同的数据长度和不同的数据类型
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);
// MPI_Sendrecv_replace 只有一个缓冲区，同时是发送缓冲区和接收缓冲区
int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);
~~~

> MPI_Bcast 从root进程向通讯子中的所有其他进程广播一条消息

~~~C++
// buffer是个inout参数，对于root来说是out，对组内所有其他进程是in
// 组内所有进程在调用形式上是完全一致的
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
~~~

> MPI_Bsend 阻塞缓冲发送

~~~C++
// 缓冲通信
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
~~~

> MPI_Buffer_attach

```C++
// 把申请好的缓冲区提交给MPI
int MPI_Buffer_attach(void *buffer, int size)
```

> MPI_Buffer_detach

~~~C++
// 从MPI收回缓冲区
int MPI_Buffer_detach(void *buffer, int *size);
~~~

> MPI_Pack_size & MPI_BSEND_OVERHEAD

~~~C++
// 返回 打包一个消息的 空间上限
// 打包incount个datatype变量需要的空间
int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);
// MPI_BSEND_OVERHEAD 是 BSEND例程使用缓冲区的最大可能空间量。
// bufsize指出了用BSEND发送一条消息所需内存的总用量。
bufsize = MPI_BSEND_OVERHEAD + packSize;
~~~

> MPI_Ssend 阻塞同步发送

~~~C++
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
~~~

> MPI_Rsend 阻塞就绪发送

~~~C++
int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
~~~

> MPI_Irecv 非阻塞式接收

~~~C++
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
              MPI_Request *request);
~~~

> MPI_Wait 等待一个MPI请求的完成 && MPI_BOTTOM

~~~C++
int MPI_Wait(MPI_Request *request, MPI_Status *status);
//
MPI_BOTTOM // 用于指出地址空间的底部
~~~

> MPI_Isend

~~~C++
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
              MPI_Request *request)
~~~

> MPI_Test

~~~C++
int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
~~~

> MPI_Cancel

~~~C++
int MPI_Cancel(MPI_Request *request);
~~~

> MPI_Test_cancelled 检查通信操作是否被取消

~~~C++
int MPI_Test_cancelled(MPI_Status status, int *flag);
~~~

> MPI_Request_free 用来释放一个MPI_Request对象

~~~C++
int MPI_Request_free(MPI_Request *request)
~~~

> MPI_Probe && MPI_Iprobe

~~~C++
int MPI_Probe(int source,int tag,MPI_Comm comm,MPI_Status *status);
int MPI_Iprobe(int source,int tag,MPI_Comm comm,int *flag, MPI_Status *status);
~~~

> MPI_Send_init && MPI_Recv_init && MPI_Startall

~~~C++
int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                  MPI_Comm comm, MPI_Request *request);
int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
                  MPI_Request *request);
int MPI_Startall(int count, MPI_Request array_of_requests[]);
~~~

> MPI_Gather && MPI_Gatherv					<a name="MPI_Gatherv"></a>

~~~C++
// 接收个数是指接收每个进程的元素个数
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
               int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root,
                MPI_Comm comm);
~~~

> MPI_Scatter~ˈskætə(r)~ && MPI_Scatterv			<a name="MPI_Scatterv"></a>

~~~C++
// 根发送缓冲区、发送给每个进程的元素个数 接收缓冲区、接收元素个数
int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
// 根发送缓冲区、发送给各个进程的元素个数数组、发送缓冲区偏移量数组 接收缓冲区、接收元素个数
int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[],
                 MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int root, MPI_Comm comm);
~~~

> MPI_Allgather && MPI_Allgatherv					<a name="MPI_Allgatherv"></a>

~~~C++
// 每一个进程都收集到其他所有进程的数据。
int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                  int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
// for each j：进程j的第j块数据将被所有进程接收到它们接收区的第j块。
int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                   MPI_Comm comm);
~~~

> MPI_Alltoall && MPI_Alltoallv							<a name="MPI_Alltoallv"></a>

~~~C++
/*
MPI_ALLGATHER每个进程散发一个相同的消息给所有的进程，MPI_ALLTOALL散发给不同进程的消息是不同的。因此它的发送缓冲区也是一个数组。MPI_ALLTOALL的每个进程可以向每个接收者发送数目不同的数据，第i个进程发送的第j块数据将被第j个进程接收并存放在其接收消息缓冲区的第i块
接收缓冲区组成的矩阵是发送缓冲区矩阵的转置
*/
int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
// 自定义发送和接受的位置和数量
int MPI_Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[],
                  MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[],
                  MPI_Datatype recvtype, MPI_Comm comm)
~~~

> MPI_Reduce

~~~C++
// 组内每个进程sendbuf中的数据按给定的操作op进行运算并将其结果返回到序列号为root的进程的recvbuf中
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
               int root, MPI_Comm comm);
~~~

> MPI_Type_create_struct 创建一种struct数据类型					<a name="MPI_Type_create_struct"></a>

~~~C++
// count个块，每个块都有指定的数据类型、元素个数和字节偏移
int MPI_Type_struct(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[],
                    MPI_Datatype array_of_types[], MPI_Datatype *newtype);
// 上述函数已被移除，用 MPI_Type_create_struct 替代
int MPI_Type_create_struct(int count, const int array_of_blocklengths[],
                           const MPI_Aint array_of_displacements[],
                           const MPI_Datatype array_of_types[], MPI_Datatype *newtype);
~~~

> MPI_Op_create && MPI_User_function && MPI_Op_free		<a name="MPI_Op_*"></a>

~~~C++
// commute=true是可交换的。
int MPI_Op_create(MPI_User_function *function,int commute,MPI_Op *op);
// C function to combine values
// invec和inoutvec的len个data做op归约 写入inoutvec缓冲区
typedef void MPI_User_function(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);
// 释放用户自定义操作op 将op置为MPI_OP_NULL
int MPI_Op_free(MPI_Op *op)
~~~

> MPI_Type_contiguous												<a name="MPI_Type_contiguous"></a>

~~~c++
int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
~~~

> MPI_Type_vector && MPI_Type_hvector															<a name="MPI_Type_vector"></a>

~~~C++
// stride:number of elements between start of each block
int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype,
                    MPI_Datatype *newtype);
// stride:number of bytes between start of each block
int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
// 本函数已被移除，用 MPI_Type_hvector 替代
~~~

> MPI_Type_indexed && MPI_Type_hindexed												<a name="MPI_Type_indexed"></a>

~~~C++
// count个块，每个块array_of_blocklengths[i]个ele，对应偏移量array_of_displacements[i]
int MPI_Type_indexed(int count, const int array_of_blocklengths[],
                     const int array_of_displacements[], MPI_Datatype oldtype,
                     MPI_Datatype *newtype);
// 偏移量不再是extent的倍数，而是字节数
int MPI_Type_hindexed(int count, int array_of_blocklengths[], MPI_Aint array_of_displacements[],
                      MPI_Datatype oldtype, MPI_Datatype *newtype);
// 本函数已被移除，用 MPI_Type_hindexed 替代
~~~

> MPI_Type_commit && MPI_Type_free															<a name="MPI_Type_com_free"></a>

~~~C++
int MPI_Type_commit(MPI_Datatype *datatype);
// 将该变量设为MPI_DATATYPE_NULL
int MPI_Type_free(MPI_Datatype *datatype);				
// 释放一个数据类型并不影响另一个根据这个被释放的数据类型定义的其它数据类型
~~~

> MPI_Get_address																									<a name="MPI_Get_address"></a>

~~~C++
int MPI_Get_address(const void *location, MPI_Aint *address);
~~~

> MPI_Type_get_extent																									<a name="MPI_Type_get_extent"></a>

~~~C++
// 返回数据类型的下界和跨度
int MPI_Type_get_extent(MPI_Datatype datatype, MPI_Aint *lb, MPI_Aint *extent);
~~~

> MPI_Pack																									<a name="MPI_Pack"></a>

~~~C++
// 发送缓冲区inbuf中的inbount个datatype类型的消息放到起始为outbuf的连续空间，该空间共有outcount个字节
// position 缓冲区outbuf当前位置，用于打包的起始地址，打包后它的值根据打包消息的大小来增加
int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize,
             int *position, MPI_Comm comm);
~~~

> MPI_Unpack

~~~C++
// 从inbuf和insize指定的缓冲区空间将不连续的消息解开放到outbuf,outcount,datatype指定的缓冲区
// position 缓冲区inbuf当前位置，输出缓冲区中被打包消息占用的起始地址，解包后它的值根据打包消息的大小来增加
int MPI_Unpack(const void *inbuf, int insize, int *position, void *outbuf, int outcount,
               MPI_Datatype datatype, MPI_Comm comm);
~~~

> MPI_Errhandler_*																									<a name="MPI_Errhandler"></a>

~~~C++
// MPI_Handler_function: 
typedef void (MPI_Handler_function) (MPI_Comm *, int *, ...);
int MPI_Errhandler_create(MPI_Comm_errhandler_function *comm_errhandler_fn, MPI_Errhandler *errhandler);
// 上述函数已废弃 用 MPI_Comm_create_errhandler 代替
int MPI_Comm_create_errhandler(MPI_Comm_errhandler_function *comm_errhandler_fn,
                               MPI_Errhandler *errhandler);

int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
int MPI_Errhandler_free(MPI_Errhandler *errhandler);
int MPI_Error_string(int errorcode, char *string, int *resultlen);
int MPI_Error_class(int errorcode, int *errorclass);
~~~

> MPI_Comm_split																									<a name="MPI_Comm_split"></a>

~~~C++
int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
~~~

> MPI_Intercomm_create																									<a name="MPI_Intercomm_create"></a>

~~~C++
// peer_comm一般是MPI_COMM_WORLD，remote_leader是远程通信域中特定进程在MPI_COMM_WORLD里的rank
int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm,
                         int remote_leader, int tag, MPI_Comm *newintercomm);
~~~

> MPI_Keyval_create																									<a name="MPI_Keyval_create"></a>

~~~C++
// 本API已被废弃，用 MPI_Comm_create_keyval 替代
int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval,
                      void *extra_state);
// MPI_Copy_function
typedef int MPI_Copy_function(MPI_Comm *oldcomm,int *keyval, void *extra_state,
                              void *attribute_val_in, void **attribute_val_out,int *flag);

// MPI_Delete_function
typedef int MPI_Delete_function(MPI_Comm *comm,int *keyval, void*attribute_val,void *extra_state);
~~~

> MPI_Cart_create																									<a name="MPI_Cart_create"></a>

~~~C++
// ndims维的处理器阵列，每一维分别有dims[i]个处理器 periods指出每个维度是否首尾相邻 reorder是否重排序rank
int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[],
                    int reorder, MPI_Comm *comm_cart);
~~~

> MPI_Graph_create																									<a name="MPI_Graph_create"></a>

~~~C++
// nnodes=节点数 节点i的度数是index[i]-index[i-1] 所有结点的边都按照结点编号的次序存入edges
int MPI_Graph_create(MPI_Comm comm_old, int nnodes, const int indx[], const int edges[],
                     int reorder, MPI_Comm *comm_graph);
~~~



