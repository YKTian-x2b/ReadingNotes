# 深入理解计算机系统

## 1.课程概述

- 课程内容包括：数据表示、机器码、内存、性能、网络等。

## 2.位、字节和整型

- 计算机用补码表示数据。
  - 原码需要单独考虑符号位，反码有正负0问题。

- 逻辑左移和算术左移：从bit的角度看是一样的，从右填充0。
- 逻辑右移和算术右移：逻辑右移，从左填充0。算术右移从左填充符号位。

- 正数的原码、反码、补码都是真值的二进制表示。
- 负数的原码是真值绝对值的二进制表示，符号位置1。负数的反码是原码除符号位外各位取反。负数的补码是负数的反码+1。
- 根据上述规则，计算机里二进制数据转换为真值：
  - 无符号数或符号位为0：补码就是原码，就是真值的二进制表示。
  - 有符号最高位为1：除符号位外，-1再取反，就是负数的原码。
    - 或者，最高位代表的十进制数直接变负数，其余位不变，相加得真值。

- 有符号数和无符号数同时参与的运算，**将会转换为全是无符号数的运算**。这是一个容易产生错误的点。

- 符号扩展：在不改变低位bit的情况下，左复制符号位，可以达到不改变补码值，同时扩展位数的效果。
  - 例如：1110 is -2。11110 is also -2。
  - short int 转 int 之类的，就可以直接扩展8*n个符号位，真值不变，转化简易。

- 截断(truncating)：高位截断，保留低位。
  - unsigned 转 unsigned short：取mod。
    - 11011 is 27。 1011 is 11。 27 % 16 == 11。
  - int 转 short int：都看做无符号二进制表示，取mod之后，再看做有符号数，得真值。
    - 10011 is -13。 0011 is 3。 19 % 16 == 3。其中 19和3是看做的无符号数，-13和3是真值。

- 除法置换成算术右移时会产生一个问题：
  - -3 / 2在c和java里都是向零舍入的，如果用-3 >> 1来替代的话，就会出现-2的不同结果。
  - 所以，应该在移位之前加一个bias，它的值是2^k-1^-1，k是移的位数。（移码）
  - 把乘除替换为相应的移位运算是优化的一种策略。

- 数字取负：
  - 所有位取反，+1就得到一个数的负数了。~x+1。

- 无符号数的使用场景：
  - 模运算居多的地方，比如hash、加密算法
  - 数字用来表示集合而不是数，比如位图

- 做复杂判断的时候，0和TMIN是俩突破点。

## 3.浮点数

- IEEE浮点数标准：(-1)^S^M2^E^，32位单精度和64位双精度。

  - 单精度：S符号位占1bit，E指数占8位，M小数占23位。
  - 双精度：S符号位占1bit，E指数占11位，M小数占52位。

- (-1)^S^M2^E^

  - E是移码表示的数，所以其真值E‘=Exp-Bias。其中Exp就是该段区域的无符号值，Bias是2^k-1^-1。
    - 移码和真值表示的数的二进制相对大小是一致的，利于比较。

  - M是1.0到2.0间的小数，即1.xxxxxx。是原码表示的数。
    - 小数点前面的1没有被存储。
  - NaN ：阶码的每个二进制位全为1 并且 尾数不为0；
  - 无穷：阶码的每个二进制位全为1 并且 尾数为0；符号位为0，是正无穷，符号位为1是负无穷。

- 非标准浮点数的M，整数位是0，所以更接近真值0。

- 浮点数的四种舍入：向零舍入、向上舍入、向下舍入和最近偶数舍入。
  - IEEE默认是**最近偶数舍入**。即如果浮点数在其整数值和其最近的偶数区间内过半则入，否则舍。
- 对于二进制小数的舍入，有效位之后的数，小于中间值就舍，大于中间值就入，等于中间值就向偶舍入。
  - 对于两位有效位：10.00011舍，10.00110入，10.11100入，10.10100舍。

- 浮点数运算不满足结合律。

## 4.机器级编程I-基础

~~~shell
gcc -Og -S sum.c		# -O 开启优化 g优化级别是debug -S 停止
~~~

- 常用工具：gcc、gdb（GNU symbolic debugger）
- movq：在复制数据的时候，源端和目的端可以是立即数、寄存器和内存。因为某些原因，不是2的3次。比如：目的端是立即数是没有意义的。
  - 立即数 到 寄存器或内存
  - 寄存器 到 寄存器或内存
  - 内    存 到 寄存器

~~~assembly
movq Src, Dest
# temp = 0x4 立即数 到 寄存器
movq $0x4, %rax
# *p = -147 立即数 到 内存
movq $-147, (%rax)
# temp2 = temp1 寄存器 到 寄存器
movq %rax, %rdx
# *p = temp 寄存器 到 内存
movq %rax, (%rdx)
# temp = *p 内存 到 寄存器
movq (%rax), %rdx
~~~

- (R)指 取寄存器里的内存地址R
- D(R)指 R还要加Displacement 才是结果地址

~~~c
void swap(long *xp, long *yp){
    long t0 = *xp;
    long t1 = *yp;
    *xp = t1;
    *yp = t0;
}
~~~

~~~assembly
swap:
	movq	(%rdi), %rax	; r means 64bit, d means destination, i means idx, a means accumulation, x means what?
	movq	(%rsi), %rdx	; rdx i/o pointer
	movq	%rdx, (%rdi)
	movq	%rax, (%rsi)
	ret
~~~

- D(Rb,Ri,S)指 Mem[Reg[Rb]+S*Reg[Ri]+D];	Register base, Register index
- lea：加载有效地址，把 Src表达式计算到的结果存到Dst里。

~~~assembly
leaq Src, Dst
# 这条语句就是把 rdi和rsi内容相加 得到的结果 放到rax里
leaq (%rdi, %rsi), %rax
~~~

## 5.Machine-Level Programming II：Control

- rsp: stack pointer.	保存栈顶元素地址
- rip: instruction pointer.    保存当前指令的地址
- CF：carry flag for unsigned。unsigned overflow cause  CF set
- SF：sign flag for signed。t < 0
- ZF：zero flag。t == 0
- OF：overflow flag for signed。正溢或负溢

- leaq指令不会设置这些flag。

~~~assembly
cmpq src2, src1		; src1 - src2 without setting dest
testq src2, src1	; src1 & src2
~~~

- setX指令

~~~C
int gt(long x, long y){
    return x > y;
}
~~~

~~~assembly
cmpq	%rsi, %rdi	; %rsi for y and %rdi for x	compute x - y
setg	%al			; if greater then set %al 1
movzbl	%al, %eax	; move with zero extension byte to long, set all other bytes to 0
# %eax 是因为 如果不涉及高32位操作的话，默认设0
ret
~~~

- jX指令
  - ja是无符号大于，jg是有符号大于。


~~~c
long absdiff(long x, long y){
    long result;
    if(x > y)
        result = x - y;
    else
        result = y - x;
    return result;
}
~~~

~~~assembly
absdiff:
	cmpq	%rsi, %rdi
	jle		.L4
	movq	%rdi, %rax
	subq	%rsi, %rax
	ret
.L4:
	movq	%rsi, %rax
	subq	%rdi, %rax
	ret
~~~

- conditional move
  - 比较通用的一种优化方法：把分支的俩结果都算出来，最后选一个。
  - 如果遇到两分支都复杂、两分支有耦合、两分支同时计算会得到错误结果的情况，则不能用该方法。

~~~assembly
absdiff:
	movq	%rdi, %rax
	subq	%rsi, %rax	; result=x-y
	movq	%rsi, %rdx
	subq	%rdi, %rdx	; eval=y-x
	cmpq	%rsi, %rdi	; x:y
	cmovle	%rdx, %rax	; if <=, result=eval
	ret
~~~

- for、while
- switch：jumpTable+jumpTargets。

## 6.Machine-Level Programming III：Procedures

- procedures: 类似于方法、函数、过程

### 6.1 Passing Control

- 栈stack的生长方向和地址空间相反，栈底在高位地址空间，栈顶在地位地址空间。

- 当在一个函数(procedure)中调用(call)另一个函数时，rip(instruction pointer)会替换为被调用函数的首地址，rsp(stack pointer)会 -=8，新增内容当前函数的下一指令地址。
- 当被调用函数返回(return)时，rsp +=8 进行pop操作，rip获取该栈顶地址值，会返回调用函数的下一指令地址。

### 6.2 Passing Data

- ABI(Application binary interface)规定了前六个参数用的寄存器：**%rdi、%rsi、%rdx、%rcx、%r8、%r9**。返回值寄存器：**%rax**。
  - 再多的参数都在stack里存着。
  - 浮点数是一组特定寄存器(FPU)来传递的。

### 6.3 Local data Management

- stack frame：用于特定call的每个块，是procedure执行的环境。包括：
  - 函数的返回地址和参数（7+）
  - 临时变量: 包括函数的非静态局部变量以及编译器自动生成的其他临时变量
  - 函数调用的上下文
- frame pointer：%rbp (base pointer)。rbp是可选的，只有在不知道函数会花费多大空间（buffer、动态数组）的情况下，会依赖rbp。
- esi：32位 source index。
- ABI定义了调用函数和被调用函数需要保持的一组寄存器，当它们的值在函数调用过程中被“不情愿”的修改时，会借助栈帧来保持。
  - caller saved 暂时值会在 call之前保存在栈帧中。
  - called saved 暂时值会在 被使用前保存在栈帧中，返回前写回。
  - %r10和%r11 用于 caller saved temporaries
  - %rbx、%r12-%r14 用于 callee saved
  - %rbp 用于 callee saved && 栈帧指针
  - %rsp 不要乱搞它
- 递归和函数调用也是一样，完全依赖栈规则和栈，没有特殊的地方。

## 7.Machine-Level Programming IV：Data

### 7.1 数组

~~~C++
#define ZLEN 5
typedef int zip_dig[ZLEN];
int get_digit(zip_dig z, int digit){
	return z[digit];
}
// %rdi for z, %rsi for digit
// %rid + 4*%rsi --> %eax
// movl	(%rdi, %rsi, 4), %eax
~~~

### 7.2 Struct

- 结构体内的成员变量，会有对齐操作。对齐的尺度，取决于所有成员数据类型的最大字节长度。

~~~C++
// 4B for c; 4B for i; 4B for d;
struct S4{
    char c;
    int i;
    char d;
}
// 4B for i; 4B for c and d;
struct S5{
    int i; 
    char c;
    char d;
}
~~~

### 7.3 Float Point

- 8086的协处理器8087，可以在单芯片上实现完整的IEEE浮点标准所需的所有硬件。
- SSE（streaming SIMD extension）FP：指令集，特用于vector指令。
- AVX：更新版本的SIMD。

------

- addss：单精度浮点(single scalar)加法。

- addps：add packed single scalar。SSE的128bit=16B，16个B，4个Float。同时进行add操作。
- addsd：双精度浮点加。
- 全是caller saved。参数依次放在%xmm0、%xxm1...上。返回值在%xmm0里。


## 8.Machine-Level Programming V：Advanced Topics

### 8.1 Memory Layout

- 地址空间的地址只有47位可用，就是2^47^个B。128TB差不多。
- 在linux的地址空间里，stack从最高地址（7FFFFFFFFFFF）开始向下生长，限制为8MB。在低地址端，依次是Text（文本/代码段）、Data（数据段）和Heap。内存的某个位置，是被引用的库函数代码（shared Libs）。
  - 文本段：代码所在的可执行程序位置
  - heap向上生长

~~~C
char big_array[1L<<24];		// 全局数组被看做程序本身一部分 放在BSS段	CSAPP课程里没有说BSS，结论是放在Data段
char huge_array[1L<<31];

int global = 0;				// 全局已初始化变量 放在数据段Data 

int useless()	{return 0;}	// 程序执行代码 放在代码段Text 只读 也有可能包含一些只读的常数变量，例如字符串常量等。

int main(){					// 程序执行代码 放在代码段Text
    
    int local = 0;			// 局部变量 放在栈里	函数调用的参数、返回值也在栈里
    void *p1, *p2, *p3;
    p1 = malloc(1L << 28);	// 动态分配空间 放在堆里
    p2 = malloc(1L << 28);
    p3 = malloc(1L << 28);
    return 0;
}
// BSS（block started by symbol）段 用来存放程序中未初始化的全局变量、静态局部变量的内存区域，属于静态内存分配。
// Data段 用来存放程序中已初始化的全局变量、静态局部变量的内存区域，属于静态内存分配。
~~~

### 8.2 Buffer Overflow

~~~C++
void echo(){
    char buf[4];
    gets(buf);
    puts(buf);
}
void call_echo(){
    echo();
}
~~~

~~~assembly
sub		$0x18, %rsp			; 这里实际上分配了 24B 所以当输入超过24B的时候 会有缓冲区溢出
mov		%rsp, %rdi
callq	400680 <gets>		
mov		%rsp, %rdi
callq	400520 <puts@plt>
add		$0x18, %rsp
retq
~~~

- 代码注入攻击：将一段刻意设计的二进制代码通过类似于gets这种有漏洞函数读入内存，溢出到另一个函数中，这样就会改变另一个函数的行为，实现攻击。
- 代码注入攻击的应对方式：
  - ASLR：地址 空间 布局 随机化。stack空间里 变量的起始位置是随机分配的，以防止溢出攻击。
  - 标记stack空间不可执行（硬件层面）。
  - stack canary（金丝雀）：在内存的某个地方拿一个8字节的数据，放到stack分配的空间里，如果有溢出，则该数据段被修改。通过对比源内存空间的数据，可以预测溢出。
    - 矿井在下人之前，放一只金丝雀进去，如果鸟死了，说明瓦斯泄漏了。用金丝雀的安危来预测人的安危。
- return-oriented programming attacks：找一些系统里已存在的gadgets（小工具/可执行程序一部分的字节序列），然后把这些gadgets的地址填充到stack里。每执行到一个gadgets的末尾就会退回下一条gadgets。这样gadgets就连在一起顺序执行了。
  - gadgets的最后一个字节有十六进制值c3，这是x86返回指令ret的编码方式。

### 8.3 Unions

- 这个的使用要注意小端字节序。

~~~C
union{
	unsigned char c[8];
    unsigned short s[4];
    unsigned int [2];
    unsigned long l[1];
}
// 小端字节序 数据的低字节放在地址空间低位 
char[0] = 0xf0;		// 低地址空间
char[1] = 0xf1;
char[2] = 0xf2;
char[3] = 0xf3;
char[4] = 0xf4;
char[5] = 0xf5;
char[6] = 0xf6;
char[7] = 0xf7;		// 高地址空间
long[0] = 0xf7f6f5f4f3f2f1f0;
~~~

## 9.程序优化

~~~C++
for(int i = 0; i < n; i++){
    int ni = n * i;
    for(int j = 0; j < n; j++){
        a[ni+j] = b[j];
    }
}
// 强度降低 乘转加
int ni = 0;
for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
        a[ni+j] = b[j];
    }
    ni += n;
}
~~~

~~~C++
int up = val[(i-1)*n+j];
int down = val[(i+1)*n+j];
int left = val[i*n+j-1];
int right = val[i*n+j+1];
int sum = up + down + left + right;
//
int inj = i*n+j;
int up = val[inj-n];
int down = val[inj+n];
int left = val[inj-1];
int right = val[inj+1];
int sum = up + down + left + right;
~~~

~~~C++
void lower(char *s){
    size_t i;
    // here call n times strlen and strlen is O(N)
    for(i = 0; i < strlen(s); i++){
        if(s[i] >= 'A' && s[i] <= 'Z'){
            s[i] -= ('A'-'a');
        }
    }
}
// 
size_t len = strlen(s);
// 这里编译器可能不会自动优化的原因有二：
// 一、它可能无法确实strlen的版本，所以，无法提供优化。编译器必须保证版本改变时，代码也能运行
// 二、s每次循环里都会被改变，即使strlen结果相同，但编译器可能无法确定
~~~

~~~C++
void sum_rows1(double *a, double *b, long n){
    long i, j;
    for(i = 0; i < n; i++){
        b[i] = 0;
        for(j = 0; j < n; j++){
            // 汇编里 b[i]的值会被一次次从内存中读出来 在寄存器里加上新值 再写进回内存 而不是始终在寄存器里完成
            // 这是因为 a[i*n+j]和b[i]可能指向同一个内存位置，这将导致不一致性 called Memory Aliasing
            b[i] += a[i*n+j];
        }
    }
}

void sum_rows2(double *a, double *b, long n){
    long i, j;
    for(i = 0; i < n; i++){
        double val = 0;
        for(j = 0; j < n; j++){
            val += a[i*n + j];
        }
        b[i] = val;
    }
}
~~~

~~~C++
// optimization blockers 性能阻碍因素的优化
// OP&IDENT -> mul&1 or add&0
void combine4(vec_ptr v, data_t *dest){
    long i;
    long length = vec_length(v);
    data_t *d = get_vec_start(v);
    data_t t = IDENT;
    for(i = 0; i < length; i++){
        t = t OP d[i];
    }
    *dest = t;
}
// 循环展开+指令流水线
void unroll2a_combine(vec_ptr v, data_t *dest){
    long length = vec_length(v);
    long limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x = IDENT;
    long i;
    for(i = 0; i < limit; i+=2){
        x = x OP (d[i] OP d[i+1]);
    }
    for(; i < length; i++){
        x = x OP d[i];
    }
}
// 更进一步利用流水线
void unroll2a_combine(vec_ptr v, data_t *dest){
    long length = vec_length(v);
    long limit = length - 1;
    data_t *d = get_vec_start(v);
    data_t x0 = IDENT;
    data_t x1 = IDENT;
    long i;
    for(i = 0; i < limit; i+=2){
        x0 = x0 OP d[i];
        x1 = x1 OP OP d[i+1];
    }
    for(; i < length; i++){
        x0 = x0 OP d[i];
    }
    *dest = x0 OP x1;
}
~~~

## 10.内存层次结构

- RAM：SRAM and DRAM

  - SRAM要四或六个晶体管组成一个cell（存储1bit的基本存储单元）；存取速度是DRAM的10倍；不需要刷新；贵；cache；

  - DRAM只要一个晶体管组成一个cell；需要刷新；main mem、frame buffer；

- flash mem：EEPROM（电可擦除可编程ROM）；
- 固件（firmware）：编程进ROM里的软件

- 寻道时间（main）、旋转延迟、读写时间

- CPU读取磁盘时，会发出一个三元组(r/w, logical block, mem addr)
- 固态硬盘（solid state disks）完全由闪存和固件构成，不是用机械硬盘的机械部件。闪存转换层（固件）替代了磁盘控制器的作用。
  - flash mem的更新以block为单位，但写入以page为单位，多个page组成block。这就意味着，每次写page都需要新的block来承载旧block的复制，并擦除旧block。
- cache miss：cold miss、conflict miss、capacity miss

## 11.Cache Memories

- cache memory 包含在CPU芯片内，完全由硬件管理，可以和register file、bus interface交互来避免频繁内存访问。
  - 就是cache和memory间的又一层次
- cache memory的构成：set array -> line array -> block data + tag + valid
- cache address：tag + set idx + block offset
- cache查找：set idx + tag 查看命中 -> valid 查看有效 -> block offset拿到数据首地址
- write：
  - write-hit：write-through or write-back
  - write-miss：write-allocate or no-write-allocate
    - write-allocate：load into cache and update line in cache
    - no-write-allocate：write straight to mem，donot load into cache

- 在任何存储系统中，写都比读更灵活，因为写可以延迟写，读不行。

- 空间局部性的改善方法就是
  - 避免stride过大的读
- 时间局部性的改善方法就是
  - 让读的数据多留一会儿，就是短时间内把局部先算完。

## 12.Linking

- Linker的意义：
  - 模块化：可以把大程序文件划分成小的模块。
  - 可修改性：某个模块的修改，可以只重新编译该模块，而不是整体编译。


- Linker干的事：
  - 1.符号解析：把符号定义和符号引用对应起来
  - 2.重定向：把所有模块整合到一个可执行的目标模块，该模块可以直接在系统上加载执行。
  - 

- 目标文件种类：
  - 可重定向目标文件(.o file)，由一个个.c文件生成的，未经过linker操作的。
  - 可执行目标文件(a.out file)，由linker重定向过的，可以直接复制到内存并执行的。
  - 共享目标文件(.so file)，特殊的.o文件，可以加载到内存并动态链接的（在运行时或加载的时候），也称Dynamic Link Libraries(DLLs)。
- object模块采用ELF格式（可执行可链接格式），

- Global Symbol：任何不带静态属性的全局变量或函数定义，本模块定义，可以给其他模块引用
- Externel Symbol：其他模块定义，本模块引用。
- Local Symbol：本模块定义，本模块引用。静态属性声明的全局变量或函数。
  - 以上三个定义，都不是指C语言的术语，而是更抽象层次的。
  - 静态的意义：属于和不变

~~~C
int f(){
    // 静态局部变量 某种意义上，它像一个全局变量，被放在.data section 而不是stack里
    // 某种意义上，它的作用域是局部的
    // 它会被分配 在ELF格式的table entry里 拥有唯一的名字 maybe x.1 x.2
    static int x = 0;
    return x;
}
~~~

- strong symbol：函数和已初始化的变量（定义）
- weak symbol：未初始化的变量（声明）
- Linker的解析规则：对于多个同名symbol，reference应该链接到哪个symbol
  - 不允许多个strong symbol，也就是说只能定义一次
  - 给定一个强symbol和多个弱symbol，会选强symbol
  - 给定多个弱symbol，随机选一个

- 重定向：Linker把启动代码（用来call main函数）、用户代码放入ELF的.text区域；把启动数据和用户全局数据放入ELF的.data区域，并加入Headers、.symtab、.debug等，形成可执行目标文件。

- 编译器会为Linker在汇编代码中添加标识，以支持重定向的地址偏移补全。

- Executable Object File的 .init/.text/.rodata section 可以直接复制到内存的Read-only code segment；.data/.bss section 可以直接复制到内存的data segment；

- static lib：把用到的每个库函数的.o取出来聚合成.a(archive)文件，然后链接到用户代码生成的.o文件。

~~~bash
gcc -L. libtest.o -lmine	# gcc link libtest.o libmine.a; 其中libtest.o call了 libmine.a的foo(); this is ok
gcc -L. -lmine libtest.o	# this is error
~~~

- dynamic lib(shared lib)：运行时或executable file load到内存里再link共享的库函数，所有.o用户代码共享一个副本，避免冗余复制。

***

- 运行时动态插入：

~~~C
#include<stdio.h>
#include<stdlib.h>
#include<dlfcn.h>

int x[2] = {1, 2};
int y[2] = {3, 4};
int z[2];

int main(){
    void *handle;
    void (*addvec)(int *, int *, int *, int);
    char *error;  
    handle = dlopen("./libvector.so", RTLD_LAZY);
    if(!handle){
        fprintf(stderr, "%s/n", dlerror());
        exit(1);
    }
    addvec = dlsym(handle, "addvec");
    if((error = dlerror()) != NULL){
        fprintf(stderr, "%s/n", error);
        exit(1);
    }
    addvec(x, y, z, 2);
    printf("z = [%d %d]\n", z[0], z[1]);
    if(dlclose(handle) < 0){
        fprintf(stderr, "%s/n", dlerror());
        exit(1);
    }
    return 0;
}
~~~

- 动态加载的一个example：Library interpositioning（库插入）技术，可以在编译、加载和运行时中断库函数调用，在不修改源码的情况下，加入代码，完成类似运行分析统计、异常检查等功能。类似于java的代理。

~~~C
// 初始代码
#include<stdio.h>
#include<malloc.h>
int main(){
    // 现在我们要保证
    int *p = malloc(32);
    free(p);
    return 0;
}
~~~

- 编译时库插入

~~~C
/// mymalloc.h
#ifdef COMPILETIME
#include <stdio.h>
#include <malloc.h>
void *mymalloc(size_t size){
    void *ptr = malloc(size);
    printf("malloc(%d)=%p\n", (int)size, ptr);
    return ptr;
}
void myfree(void *ptr){
    free(ptr);
    printf("free(%p)\n", ptr);
}
#endif

/// malloc.h
#define malloc(size) mymalloc(size)
#define free(ptr) myfree(ptr)
void *mymalloc(size_t size);
void myfree(void *ptr);
~~~

~~~bash
# 插入c
make intc
# -Wall 警告所有可能的风险； -D define ...； -c 只激活预处理,编译,和汇编,生成obj文件； -IDir 指定额外头文件搜索路径
gcc -Wall -DCOMPILETIME -c mymalloc.c
gcc -Wall -I. -o intc int.c mymalloc.c

make runc
./intc
malloc(32)=0x1edc010
free(0x1edc010)
~~~

- 链接时插入

~~~C++
/// mymalloc.h
#ifdef LINKTIME
#include <stdio.h>
void *__real_malloc(size_t size);
void __real_free(void *ptr);
void *__wrap_malloc(size_t size){
    void *ptr = __real_malloc(size);	/* call libc malloc */
    printf("malloc(%d) = %p\n", (int)size, ptr);
    return ptr;
}
void __wrap_free(void *ptr){
    __real_free(ptr);					/* call libc free */
    printf("free(%p)\n, ptr");
}
#endif
~~~

~~~bash
make intl
gcc -Wall -DLINKTIME -c mymalloc.c
gcc -Wall -c int.c
# -Wl.option
# 此选项传递 option 给连接程序; 如果 option 中间有逗号, 就将 option 分成多个选项, 然后传递给会连接程序。
# "--wrap,malloc"指示linker将malloc调用解析为__wrap_malloc,然后将__real_malloc解析为libc的malloc函数
gcc -Wall -Wl, --wrap,malloc, -Wl, --wrap,free -o intl int.o mymalloc.o
...
~~~

- 加载或运行时插入：

~~~C++
#ifdef RUNTIME
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

void *malloc(size_t size){
    void *(*mallocp)(size_t size);
    char *error;
    // 找下一个malloc
    mallocp = dlsym(RTLD_NEXT, "malloc");
    if((error = dlerror()) != NULL){
        fputs(error, stderr);
        exit(1);
    }
    char *ptr = mallocp(size);
    printf("malloc(%d) = %p\n", (int)size, ptr);
    return ptr;
}
void free(void *ptr){
    void (*freep)(void *) = NULL;
    char *error;
    if(!ptr)
        return;
    freep = dlsym(RTLD_NEXT, "free");
    if((error = dlerror()) != NULL){
        fputs(error, stderr);
        exit(1);
    }
    freep(ptr);
    printf("free(%p)\n", ptr);
}
#endif
~~~

~~~bash
make intr
# -shared 尽量使用动态库 -fpic -ldl
gcc -Wall -DRUNTIME -shared -fpic -o mymalloc.so mymalloc.c -ldl
gcc -Wall -o intr int.c
~~~

## 13.异常控制流：异常和处理

- 控制流：指令序列。
- 物理控制流：硬件正在执行的实际指令序列。
- 异常控制流存在于系统的所有level：
  - low level：Exceptions：系统事件响应、由硬件和OS软件合作实现
  - high level：
    - 进程上下文切换：硬件时钟+OS
    - 信号：OS
    - 非局部跳转：C运行时库

- 内核：OS的始终驻留在内存中的一部分。

- 异步异常（中断）：由处理器外部的事件引发，如Timer interrupt。
- 同步异常：由指令执行造成的事件引发，如Trap、Fault、Abort


### 13.1进程上下文切换

- A process is an instance of a running program. Two key abstractions:
  - Logical control flow, 每个进程都像是在独占CPU和寄存器运行，通过上下文切换来实现这一机制
  - Private address space, 每个进程都像是在独占内存，通过虚拟内存来实现这一机制

- 上下文切换，就是从上一个进程运行的地址空间，切换到下一个进程。同时需要把保存的寄存器值载入到物理寄存器。

- 每个进程是一个逻辑控制流，所以在判断进程**并发**的时候，我们考虑两个逻辑控制流是否在时间上重叠，重叠即**并发**。但他们的物理控制流可能并不重叠，会在不停的切换。

- kernel不是单独的进程，而是运行在一些已存在进程里。上下文切换就由kernel code完成。
- 系统级函数的调用会通过返回值（-1）和全局变量errno来指明异常情况，**系统级函数的调用必须检查返回值**。

~~~C
void unix_error(char *msg){
    fprintf(stderr, "%s, %s\n", msg, strerror(errno));
}
//
if((pid = fork()) < 0){
    unix_error("fork error");
}
~~~

- 程序员角度的程序运行状态：Running（scheduled|executing）、Stopped（suspend）、Terminated。
- exit()调用一次，返回零次。fork()调用一次，返回两次。

~~~C
// fork()之后的代码会在父子进程中分别并行执行一次；fork返回时无法预测调度顺序；会分开地址空间但内容一样；共享打开的文件
// 实际上fork会复制它的内核映像给子进程，代码、变量和程序都一样
int main(){
    pid_t pid;
    int x = 1;
    pid = fork();	// 没有写异常检测处理函数，为了简约
    if(pid == 0){
        printf("child process x = %d\n", ++x);
        exit(1);
    }
    printf("parent process x = %d\n", --x);
    exit(1);
}
// parent process x = 0
// child process x = 2
~~~

- 一个进程结束时，OS会保存它直到它的父进程结束。它的一些状态会保存到OS表里。可能是父进程想查看子进程的退出状态。
  - 半死半活状态的子进程叫僵尸进程
- 父进程wait或waitpid会收割僵尸进程，残留的僵尸进程会被始祖进程init收割。
  - 提这个是因为，僵尸进程占用内存空间，数百万的僵尸进程可能导致内存爆掉（在父进程持续相当久的情况下）。

- 要是子进程一直活着，就得init最后收割。


~~~C
// 父进程会阻塞等待它的任一个子进程结束 返回pid并由child_status指明退出状态
int wait(int *child_status);
// 父进程会阻塞等待它的pid子进程结束
pid_t waitpid(pid_t pid, int &status, int options);
~~~

- execve

~~~c
// 在当前进程中执行可执行文件filename（脚本 二进制文件） 用参数列表argv 和 环境变量列表 envp
// filename开头 #!interpreter or #!/bin/bash
// 命令会重写code data和stack 保留 PID open files和signal context 执行一个全新的内容
// called once never return except error
int execve(char *filename, char *argv[], char *envp[]);
//
// Exe "/bin/ls -lt /usr/include"
if((pid=fork())==0){
	if(execve(myargv[0], myargv, environ) < 0){
        printf("%s: Command not found. \n", myargv[0]);
        exit(1);
    }
}
~~~

### 13.2 signal

- Linux进程层次结构：init => Daemon + Login shell s => child... => Grandchild...
  - Daemon 通常是提供服务的长期运行程序
  - Login shell 命令行界面（linux 默认 bash）

~~~C
int main(){
    char cmdline[MAXLINE];
    while(1){
        printf("> ");
        // F大写表示 系统级库函数要错误处理
        Fgets(cmdline, MAXLINE, stdin);
        if(feof(stdin))
            exit(0);
        eval(cmdline);
    }
}
void eval(char * cmdline){
    char *argv[MAXARGS];
    char buf[MAXLINE];
    int bg;			// 是否是后台任务
    pid_t pid;
    strcpy(buf, cmdline);
    // 没有内置或可执行命令 什么都没 直接返回
    if(argv[0] == NULL)		return;
    if(!builtin_command(argv)){
        if((pid = Fork()) == 0){
            if(execve(argv[0], argv, environ) < 0){
                printf("%s: Commmand not found.\n", argv[0]);
                exit(0);
            }
        }
        // 不是后台任务 需要等该子进程结束并收割
        if(!bg){
            int status;
            if(waitpid(pid, $status, 0) < 0)
                unix_error("waitfg: waitpid error");
        }
        else{
            // 是后台任务，则不管它 后续任务执行完成，通过signal通知shell 再收割
            printf("%d %s", pid, cmdline);
        }
    }
    return;
}
~~~

- 信号signal是内核传递给进程以通知它系统发生了某些事情的message。
  - 内容是一个唯一的整数id
  - 实际的通知过程就是：kernel更新目标进程上下文的一些state，that is all

| ID   | Name    | Default action | Corresponding  Event                   |
| ---- | ------- | -------------- | -------------------------------------- |
| 2    | SIGINT  | Terminate      | User typed ctrl-c                      |
| 9    | SIGKILL | Terminate      | Kill program(cannt override or ignore) |
| 14   | SIGALRM | Terminate      | Timer signal                           |
| 17   | SIGCHLD | Ignore         | Child stopped or terminated            |

- signal发送的原因：
  - 内核检测到系统事件（除零异常或子进程结束）
  - 其他进程要求内核发送signal给目标进程，通过kill system call

- signal接收：目标进程被kernel强制要求对signal的传递做出反应。反应包括：
  - Ignore signal
  - Terminate the process(with optional core dump)
  - catch the signal by executing a user-level function called signal handler

- 一个信号被发送但未被接收会处于pending（待定/挂起）状态，且最多有一个特定类型的pending信号，不会排队。后续的同类信号会被丢弃。

- 进程可以阻塞接收信号。
- kernel维护每个进程上下文的pending、blocked位向量
  - signal vector is also called signal mask

~~~bash
/bin/kill -9 24818
# 发送给进程组的每个进程
/bin/kill -9 -24817
~~~

- ctrl+c是发送SIGINT给每个前台进程，要求terminate。ctrl+z是发送SIGSTP给每个前台进程，要求挂起，直到SIGCONT才恢复。

~~~C
void sigint_handler(int sig){
    printf;
    sleep(2);
    printf;
    fflush(stdout);
    sleep(1);
    pirntf;
    exit(0);
}
int main(){
    if(signal(SIGINT, sigint_handler) == SIG_ERR)
        unix_error("signal error");
    pause();		// wait for the receipt of a signal
    return 0;
}
~~~

- handler和主程序在一个进程里执行，会共享全局状态。handler可以被其他类型的handler打断（handler嵌套）。
  - sigprocmask显式阻塞或解除阻塞一个信号

~~~C
sigset_t mask, prev_mask;
Sigemptyset(&mask);
Sigaddset(&mask, SIGINT);
// Block SIGINT and save previous blocked set 
Sigprocmask(SIG_BLOCK, &mask, $prev_mask);
/* Code region that will not be interrupted by SIGINT */
// restore previous blocked set, unblocking SIGINT
Sigprocmask(SIG_SETMASK, &prev_mask, NULL);
~~~

- 写安全handler的指南
  - 保持handler简单
  - 在handler里只调用异步信号安全的函数
  - 保存和重置errno在进入和退出时
  - 通过暂时阻塞信号来保护共享数据结构的访问
  - 把全局变量声明为volatile
  - 对于只进行读和写的全局变量（不会递增递减），标记为volatile和sig_atomic

- 函数的异步信号安全：函数是可重入的或不能被信号中断。
  - 是：_exit/write/wait/waitpid/sleep/kill/sio_putl/sio_puts/sio_error
  - 不是：printf/sprintf/malloc/exit

- 像read这种慢系统调用，在信号来的时候，会被打断并返回错误。需要检查这种情况，并redo。（while）
  - 慢系统调用：类似read这种，发起磁盘访问请求，然后调度别的进程，等读完，再返回。

- signal有不安全、语义易误解、不可移植、打断慢系统调用等问题。
- 不可移植、不可预测的解决方案：sigaction。

~~~C
void handler(int sig){
    int olderrno = errno;
    sigset_t mask_all, prev_all;
    pid_t pid;
    
    Sigfillset(&mask_all);
    while((pid = waitpid(-1, NULL, 0)) > 0){
        Sigprocmask(SIG_BLOCK, &mask_all, &prev_all);
        deletejob(pid);
        Sigprocmask(SIG_SETMASK, &prev_all, NULL);
    }
    if(errno != ECHILD){
        Sio_error("waitpid error");
    }
    errno = olderrno;
}
int main(int argc, char **argv){
    int pid;
    sigset_t mask_all, prev_all;
    Sigfillset(&mask_all);
    Signal(SIGCHLD, handler);
    initjobs();
    
    while(1){
        // 这个的缺陷在于，如果子进程先called且退出，那么父进程后add的job将无法被delete
        if((pid = Fork()) == 0){
            Execve("/bin/date", argv, NULL);
        }
        Sigprocmask(SIG_BLOCK, &mask_all, &prev_all);
        addjob(pid);
        Sigprocmask(SIG_SETMASK, &prev_all, NULL);
    }
    exit(0);
}
///
// 改进
int main(int argc, char **argv){
    int pid;
    sigset_t mask_all, mask_one, prev_one;
    Sigfillset(&mask_all);
    Sigemptyset(&mask_one);
    Sigaddset(&mask_one, SIGCHLD);
    Signal(SIGCHLD, handler);
    initjobs();
    
    while(1){
        // 即使子进程先开始并提前退出，SIGCHLD也被两次block延迟到addjob之后了。（第一个unblock不会被父进程执行）
        // 而且这样不会影响子进程释放它的子进程
        Sigprocmask(SIG_BLOCK, &mask_one, &prev_one);		// block sigchld
        if((pid = Fork()) == 0){
            Sigprocmask(SIG_SETMASK, &prev_one, NULL);		// unblock sigchld
            Execve("/bin/date", argv, NULL);
        }
        Sigprocmask(SIG_BLOCK, &mask_all, NULL);
        addjob(pid);
        Sigprocmask(SIG_SETMASK, &prev_all, NULL);
    }
    exit(0);
}
~~~

- sigsuspend

~~~C
// 暂停直到有信号来
int sigsuspend(const sigset_t *mask);
~~~

## 14.系统IO

- RIO（robust I/O)包值得去学习一下，异常处理、错误处理等。

- Unix一切都是文件，无论从磁盘读取字节序列（文件）还是网络数据发送、接收，都是一样的IO。

~~~C
// rio_readn 健壮的读取n个字节 没有缓冲的 处理了short count问题
ssize_t rio_readn(int fd, void *usrbuf, size_t n){
    size_t nleft = n;
    ssize_t nread;
    char *bufp = usrbuf;
    while(nleft > 0){
        if((nread = read(fd, bufp, nleft)) < 0){
            if(errno == EINTR)		// error interrupted by sig handler return
                nread = 0;
            else	
                return -1;			// errno set by read()
        }
        else if(nread == 0)
            break;					// EOF
        nleft -= nread;
        bufp += nread;
    }
    return (n - nleft);				// Return >= 0
}
~~~

- 文件元数据

~~~C
int main(int argc, char **argv){
    struct stat stat;	// 元数据的表示
    char *type， *readok;
    Stat(argv[1], &stat);
    if(S_ISREG(stat.st_mode)){
        type = "regular";
    }
    else if(S_ISDIR(stat.st_mode)){
        type = "directory";
    }
    else
        type = "other";
    if((stat.st_mode & S_IRUSR)){
        readok = "yes";
    }
    else
        readok = "no";
    printf("type: %s, read: %s\n", type, readok);
    exit(0);
}
~~~

~~~bash
./statcheck foo.c
type: regular, read: yes
chmod 000 foo.c
./statcheck foo.c
type: regular, read: no
~~~

## 15.虚拟内存

- 好处：更有效的使用物理内存、简化内存管理，每个进程都有相同统一独立受保护的线性内存空间、独立地址空间（用户空间和内核空间、各进程空间）
- 虚拟内存可以把多个虚拟页映射到单个物理页，以实现数据和资源共享。
  - 共享库
- 虚存简化了Linking：有个进程都有相似的内存空间代码段、数据段、堆的起始位置相同

- 虚存简化了Loading：execve为.text和.data部分分配虚存页表项并标记为invalid，这俩部分按需进行页到页复制。
  - 按需加载的意思：如果.data有非常大的数组，那么不访问该数组的时候不会加载。


- 虚拟地址空间的48位的，但存的是64位的，高位全0或全1。可以设置一些位来控制虚拟地址的访问权限。如内核还是用户、rwx。

- TLB(Translation Lookaside Buffer)硬件缓存PTE，in MMU。
- 48位虚拟地址空间，每个page占4KB，一个页表项8B的情况下，需要512GB的内存才能存下所有的PTE。但大部分虚拟地址空间没用，所以，用多级页表。一级页表常驻内存，存虚拟地址所在PT指针。二级页表同其他数据一样，换入换出，存PTE。

- Intel core i7用四级页表存了512GB的虚存空间。

- 虚存areas通过将它们连接到磁盘对象来初始化，执行进程叫内存映射。

- fork用到了private copy-on-write技术。fork调用时会创建mm_struct/vm_area_struct/page tables这些小的数据结构，但不会实际复制虚拟内存空间。而是共享父虚拟空间。当进程read该空间的时候，始终共享不复制。只有进程write的时候，开辟新的虚拟空间，重新映射到新的物理空间。
  - 最初两个进程的pt都是只读的、vm_area_struct都是COW

- evecve会重用当前虚拟地址空间运行新程序。free  and new vm_area_struct/page tables。

~~~C
void mmapcopy(int fd, int size){
    char *bufp;
    bufp = Mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    write(1, bufp, size);
    return;
}
// void *mmap(void *start, int len, int port, int flags, int fd, int offset);
// 打开文件读并写入标准输出流，通过mmap可以避免读到用户空间
int main(int argc, char **argv){
    struct stat stat;
    int fd;
    fd = Open(argv[1], O_RDONLY, 0);
    Fstat(fd, &stat);
    mmapcopy(fd, stat.st_size);
    exit(0);
}
~~~

## 16.动态内存分配

- App => Dynamic Mem Allocator => heap

- 内部碎片化：为了 对齐 和 存空间分配所需的额外数据结构 而产生的Block - Payload的额外消耗。

- 外部碎片化：heap有足够空闲block空间，但是没有单个block可以满足当前请求的size。

- 外部碎片化的解决方案：隐式链表、显式链表。
- 隐式：在block头、尾占用一个B去表示，block size和占用情况。尾字节（boundary tag）是用来被下一个block合并的。
  - 已分配的块中，可以不要尾字节。下一个block通过头字节倒数第二低位来判定占用情况。因为要字节对齐，所以头字节低位一定是000。
  - 如果是空闲块，那它就需要一个尾字节，但那无所谓，因为它就是空的。如果是占用块，就能省一个字节。

- malloc是需要立即执行的，但free不一定，可以通过延时合并来换取更好的效率。类似于read和write。

- 峰值内存利用率U(K)：H(K)指代第k+1次malloc或free请求后的堆大小，P(K)指代第k+1次malloc或free请求后的所有payloads总和。
  - U(K) = maxP(K) / H(K), K = 0, 1, 2, 3...
  - 对任意的内存请求序列，maxP(K)是固定的，但H(K)取决于堆存储的效率。
- 显式空闲链表：挑个空闲空间存所有空闲空间的双向链表。




















## 100.额外的知识

- volatile 的意思是让编译器每次操作修饰的变量时一定要从内存中真正取出，而不是使用已经存在寄存器中的值。

- Intel处理器，8086/8286这些统称 x86。所以x86可以是Intel处理器的代称。是一种CISC。
- ARM处理器，是一种RISC。
- 汇编指令

~~~assembly
addq	Src, Dest	; Dest = Dest + Src
subq	Src, Dest	; Dest = Dest - Src
imulq	Src, Dest	; Dest = Dest * Src
salq	Src, Dest	; Dest = Dest << Src
shlq	Src, Dest	; Dest = Dest << Src
sarq	Src, Dest	; Dest = Dest >> Src 	Arithmetic
shrq 	Src, Dest	; Dest = Dest >> Src	Logic
xorq	Src, Dest	; Dest = Dest ^ Src
andq	Src, Dest	; Dest = Dest & Src
orq		Src, Dest	; Dest = Dest | Src

incq	Src, Dest	; Dest += 1
decq 	Src, Dest	; Dest -= 1
negq	Src, Dest	; Dest = -Dest
notq	Src, Dest	; Dest = ~Dest

mov(1)(2)(3)	; (1)处为s或z或没有，表示对操作数进行sign符号位扩展或zero0扩展或不操作
				;(2)(3)处为bwlq，表示源、目的操作数的大小 分别对应 1/2/4/8字节

imulq	S		;八字有符号乘法 结果低位保存在上一行代码给出的寄存器中，高位保存在操作数给出的寄存器中
mulq	S		;八字无符号乘法 结果低位保存在上一行代码给出的寄存器中，高位保存在操作数给出的寄存器中
idivq	S		;八字有符号除法，用上一行给出的寄存器的值除以寄存器S的值 余数存在本行给出的寄存器中，商存在上一行给出的寄存器
divq	S		;八字无符号除法，用上一行给出的寄存器的值除以寄存器S的值 余数存在本行给出的寄存器中，商存在上一行给出的寄存器中

cqto	无		;符号扩展转换为八字 结果低位高位分别保存在两个寄存器中

testq 	a, b	;对两个操作数进行逻辑（按位）与操作 并根据运算结果设置符号标志位、零标志位和奇偶标志位。
~~~

- Linux进程的虚拟内存空间

![Linux进程虚拟内存空间](https://raw.githubusercontent.com/JiXuanYu0823/ReadingNotes/main/assets/Linux%E8%BF%9B%E7%A8%8B%E8%99%9A%E6%8B%9F%E5%86%85%E5%AD%98%E7%A9%BA%E9%97%B4.png)

