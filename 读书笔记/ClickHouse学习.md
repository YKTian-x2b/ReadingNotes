# ClickHouse原理解析与应用实践

- 一个高性能OLAP数据库

## 0.序言

> OLTP和OLAP

- OLTP(On-Line Transaction Processing)：联机事务处理
  - 实时性要求高、数据量不很大、并发读写、事务一般是确定的
  - 用于日常操作
- OLAP(On-Line Analytical Processing)：联机分析处理
  - 数据仓库的核心部分，用于历史数据的加工处理分析。
  - 读多写少、数据量大、动态查询
  - 用于决策

- OLAP分为ROLAP(relational)和MOLAP(Multidimensional)，以及HOLAP(Hybrid)。

> 查询加速的方法

- 查询加速的方法：**减少数据扫描范围** 和 **数据传输时的大小**。
  - 列式数据存储完成了第一点、数据压缩完成了第二点。

> 分区和分片

- 分区：将数据库中已有的表拆分开存储，分为水平分区和垂直分区;
  - 水平分区是将表按时间或按季度等划分开;
  - 垂直分区是将表按列分开,减少每张表的宽度。分区之后他们还是一个整体,只不过分开存储。

- 分片：分布式存储

## 1.架构

<img src="https://raw.githubusercontent.com/JiXuanYu0823/ReadingNotes/main/assets/ClickHouse%E6%9E%B6%E6%9E%84.png" alt="ClickHouse架构" style="zoom:50%;" />

### 1.1 Column和Field

- Column：内存中的一列数据由Column对象表示。
  - 通过泛化实现各种关系运算，
  - ColumnString、ColumnArray和ColumnTuple等实现如插⼊数据的insertRangeFrom和insertFrom⽅法、⽤于分⻚的cut，以及⽤于过滤的filter⽅法等。
- Field：Field对象表示一个单值。
  - 使⽤了聚合的设计模式。
  - 在Field对象内部聚合了Null、UInt64、String和Array等13种数据类型及相应的处理逻辑。

### 1.2 DateType

- 通过DataType实现数据的序列化和反序列化。
  - 泛化实现
- DataType不直接进行数据读取：在DataType的实现类中，聚合了相应数据类型的Column对象和Field对象。

### 1.3 Block和Block流

- ClickHouse内部的数据操作是面向Block对象进行的，并采用了流的形式。
- Block对象的本质是（数据对象，数据类型，列名称）三元组，即（Column，DataType，ColName）。
- Block通过ColumnWithTypeAndName对象间接引用Column和DataType。
- 顶层接口：IBlockInputStream负责数据的读取和关系运算，IBlockOutputStream负责将数据输出到下⼀环节。
- Block流也使⽤了泛化的设计模式。

### 1.4 Table

- 直接用IStorage接口指代数据表。
- IStorage接⼝定义了DDL（如ALTER、RENAME、OPTIMIZE和DROP等）、read和write⽅法，它们分别负责数据的定义、查询与写⼊。
- 在数据查询时，IStorage负责根据AST查询语句的指示要求，返回指定列的原始数据。后续对数据的进⼀步加⼯、计算和过滤，则会统⼀交由Interpreter解释器对象处理。对Table发起的⼀次操作通常都会经历这样的过程，接收AST查询语句，根据AST返回指定列的数据，之后再将数据交由Interpreter做进⼀步处理。

### 1.5 Interpreter与Parser

- Parser分析器 负责创建AST对象；⽽Interpreter解释器 则负责解释AST~抽象语法树~，并进⼀步创建查询的执行管道。它们与IStorage⼀起，串联起了整个数据查询的过程。
- Parser分析器可以将⼀条SQL语句以递归下降的⽅法解析成AST语法树的形式。

- Interpreter解释器的作⽤就像Service服务层⼀样，起到串联整个查询过程的作⽤，它会根据解释器的类型，聚合它所需要的资源。
  - 首先它会解析AST对象；
  - 然后执⾏“业务逻辑”（例如分⽀判断、设置参数、调⽤接⼝等）；
  - 最终返回IBlock对象，以线程的形式建⽴起⼀个查询执⾏管道。

### 1.6 Functions和Aggregate Functions

- 主要包括普通函数和聚合函数
- 普通函数：由IFuntion接口定义，函数会直接向量化地作用到一整列数据上。
- 聚合函数：由IAggregateFunction接口定义，聚合函数是有状态的，该状态支持序列化和反序列化，可以在分布式节点间进行传输。

### 1.7 Cluster和Replication

- ClickHouse的1个节点只能拥有1个分片，也就是说如果要实现1分片、1副本，则⾄少需要部署2个服务节点。
- 分⽚只是⼀个逻辑概念，其物理承载还是由副本承担的。

### 1.8 ClickHouse为什么快？

> 硬件层面的思考

- 基于将硬件功效最大化的⽬的，ClickHouse会在内存中进⾏GROUP BY，并且使⽤ **HashTable** 装载数据。
- 与此同时，他们⾮常在意CPU L3级别的缓存，因为⼀次L3的缓存失效会带来70~100ns的延迟。

> 算法层面的思考

- ClickHouse最终选择了这些算法：
  - 对于常量，使⽤Volnitsky算法；
  - 对于非常量，使⽤CPU的向量化执⾏SIMD，暴力优化；
  - 正则匹配使⽤re2和hyperscan算法。
- 性能是算法选择的首要考量指标。



## 2. 安装和部署

### 2.1 客户端访问接口

> CLI(Command Line Interface)拥有两种执行模式

- 交互式执⾏可以⼴泛⽤于调试、运维、开发和测试等场景，它的使⽤⽅法是直接运⾏clickhouse-client进⾏登录，就可以进行一问一答的交互式查询。
- ⾮交互式模式主要⽤于批处理场景，在执⾏脚本命令时，需要追加--query参数指定执⾏的SQL语句。
  - 可以追加--multiquery参数，它可以⽀持⼀次运⾏多条SQL查询，多条查询语句之间使⽤分号间隔。

> JDBC

- 可以通过各种语言的接口访问该数据库。

### 2.2 实用工具

> clickhouse-local

- clickhouse-local可以独⽴运⾏⼤部分SQL查询，不需要依赖任何ClickHouse的服务端程序。

> clickhouse-benchmark

- clickhouse-benchmark是基准测试的⼩⼯具，它可以⾃动运⾏SQL查询，并⽣成相应的运⾏指标报告

### 2.3 目录结构

- **核心目录**：
  - /etc/clickhouse-server：服务端的配置文件⽬录，包括全局配置config.xml和⽤户配置users.xml等。
  - /var/lib/clickhouse：默认的数据存储目录。
  - /var/log/clickhouse-server：默认保存日志的目录。
- 配置文件：
  - /etc/security/limits.d/clickhouse.conf：⽂件句柄数量的配置，也可用config.xml的max_open_files修改。
  - /etc/cron.d/clickhouse-server：cron定时任务配置，⽤于恢复因异常原因中断的ClickHouse服务进程。

- 在/usr/bin路径下的可执⾏文件：
  - clickhouse：主程序的可执行文件。
  - clickhouse-client：⼀个指向ClickHouse可执行⽂件的软链接，供客户端连接使⽤。
  - clickhouse-server：⼀个指向ClickHouse可执行⽂件的软链接，供服务端启动使⽤。
  - clickhouse-compressor：内置提供的压缩⼯具，可⽤于数据的正压反解。

## 3.数据定义

### 3.1 数据类型

- 分为基本类型、复合类型和特殊类型。

#### 3.1.1基础类型

- 基础类型只有数值、字符串和时间三种。

- 没有Boolean类型，可以用整型0/1替代。

1. 数值类型分为整数、浮点数和定点数三类。

> Int

- Int8、Int16、Int32和Int64
- 支持无符号整数，用前缀U表示

> Float

- Float32和Float64
- ClickHouse的浮点数支持正无穷、负无穷以及非数字的表达方式（inf、-inf、nan）

> Decimal

- Decimal32、Decimal64、Decimal128.
- 两种形式声明定点：简写方式有Decimal32(S)、Decimal64(S)、Decimal128(S)三种，原⽣方式为Decimal(P,S)，其中：
  - P代表精度，决定总位数（整数部分+小数部分），取值范围是1~38；
  - S代表规模，决定小数位数，取值范围是0~P。

~~~mysql
# toDecimal128(value, S)
SELECT toDecimal64(2,4) / toDecimal32(2,2)
~~~

2. 字符串分为String、FixedString和UUID三类。

> String

- 长度不限、字符集不限。
  - “统一保持UTF-8编码”是一种很好的约定。

> FixedString

- FixedString(N) 约等于 MySQL的Char(N)，表示固定长度的字符串。
  - 用NULL填充末尾字符。

> UUID

- UUID是⼀种数据库常⻅的主键类型，未赋值时会依照格式用0填充。

3. 时间类型分为DateTime、DateTime64和Date三类。目前没有时间戳类型。最高精度是s。

> DateTime

- DateTime类型包含时、分、秒信息，精确到秒，⽀持使⽤字符串形式写⼊。

~~~mysql
CREATE TABLE Datetime_TEST(
	c1 Datetime
)ENGINE = Memory;
INSERT INTO Datetime_TEST VALUES('2019-06-22 00:00:00');
~~~

> DateTime64

- DateTime64可以记录亚秒。

> Date

- Date类型不包含具体的时间信息，只精确到天，它同样也⽀持字符串形式写入。

#### 3.1.2 复合类型

- ClickHouse还提供了数组、元组、枚举和嵌套四类复合类型

1. **Array**

- 在查询时并不需要主动声明数组的元素类型。
  - 因为ClickHouse的数组拥有类型推断的能力，推断依据：以最小存储代价为原则，即使⽤最小可表达的数据类型。

~~~mysql
# 类型间必须兼容
SELECT [1, 2.0, null] as a, toTypeName(a);	# Array(Nullable(UInt8))
~~~

- 在定义表字段时，数组需要指定明确的元素类型。

~~~mysql
CREATE TABLE Array_TEST(
	c1 Array(String)
)engine=Memory
~~~

2. **Tuple**

- 元组同样⽀持类型推断，其推断依据仍然以最小存储代价为原则

~~~mysql
SELECT (1,2.0,null) AS x, toTypeName(x) # Tuple(UInt8, Float64, Nullable(Nothing))
~~~

- 在定义表字段时，元组也需要指定明确的元素类型

~~~mysql
CREATE TABLE Tuple_TEST (
	c1 Tuple(String,Int8)
) ENGINE = Memory;
~~~

3. **Enum**

- Enum8和Enum16，分别对应(String:Int8)和(String:Int16)
- Key和Value要保证唯一性，不能为NULL，Key允许是空串。

~~~mysql
CREATE TABLE Enum_TEST(
	c1 Enum8('ready'=1, 'start'=2, 'success'=3, 'error'=4)
) ENGINE=Memory;
INSERT INTO Enum_TEST VALUES('ready');
INSERT INTO Enum_TEST VALUES('start');
~~~

4. **Nested**

- 嵌套表内不能继续使⽤嵌套类型
- 嵌套类型本质是一种多维数组

~~~mysql
CREATE TABLE nested_test (
    name String,
    age UInt8 ,
    dept Nested(
        id UInt8,
        name String
    )
) ENGINE = Memory;

┌─name──────┬─type──────────┬─default_type─┬─default_expression─┬─comment─┬─codec_expression─┬─ttl_expression─┐
│ name      │ String        │              │                    │         │                  │                │
│ age       │ UInt8         │              │                    │         │                  │                │
│ dept.id   │ Array(UInt8)  │              │                    │         │                  │                │
│ dept.name │ Array(String) │              │                    │         │                  │                │
└───────────┴───────────────┴──────────────┴────────────────────┴─────────┴──────────────────┴────────────────┘

# 
INSERT INTO nested_test VALUES ('bruce' , 30 , [10000,10001,10002], ['研发部','技术⽀
持中⼼','测试部']);
# ⾏与⾏之间,数组⻓度⽆须对⻬
INSERT INTO nested_test VALUES ('bruce' , 30 , [10000,10001], ['研发部','技术⽀持中
⼼']);
~~~

#### 3.1.3 特殊类型

1. **Nullable**

- Nullable类型与Java8的Optional对象有些相似，它表⽰某个基础数据类型可以是Null值。
  - 只能和基础类型搭配
  - Nullable的存储方案会造成读写数据性能的降低

~~~mysql
CREATE TABLE Null_TEST (
    c1 String,
    c2 Nullable(UInt8)
) ENGINE = TinyLog;
# 通过Nullable修饰后c2字段可以被写⼊Null值：
INSERT INTO Null_TEST VALUES ('nauu', null)
INSERT INTO Null_TEST VALUES ('bruce', 20)
# 
SELECT c1 , c2 ,toTypeName(c2) FROM Null_TEST
~~~

2. **Domain**

- 域名类型分为IPv4和IPv6两类
- 如果需要返回IP的字符串形式，则需要显式调⽤IPv4NumToString或IPv6NumToString函数进⾏转换。

### 3.2 定义数据表

#### 3.2.1 数据库

- 5种引擎
  - Ordinary：默认引擎，在此DB下可以使用任意类型表引擎。
  - Dictionary：字典引擎，自动为所有数据字典创建它们的数据表。
  - Memory：内存引擎，用于存放临时数据。此类数据库下的数据表只会停留在内存中，不会涉及任何磁盘操作。
  - Lazy：日志引擎，此类数据库下只能使⽤Log系列的表引擎。
  - MySQL：MySQL引擎，此类数据库下会⾃动拉取远端MySQL中的数据，并为它们创建MySQL表引擎的数据表。

~~~mysql
# 默认数据库的实质是一个文件目录，在/var/lib/clickhouse/data下
CREATE DATABASE IF NOT EXISTS db_name [ENGINE = engine]
#
DROP DATABASE [IF EXISTS] db_name
~~~

#### 3.2.2 数据表

- 表引擎决定了数据表的特性，也决定了数据将会被如何存储及加载。
- 三种建表方式

~~~mysql
#
CREATE TABLE [IF NOT EXISTS] [db_name.]table_name (
    name1 [type] [DEFAULT|MATERIALIZED|ALIAS expr],
    name2 [type] [DEFAULT|MATERIALIZED|ALIAS expr],
    ...
) ENGINE = engine
# 复制表
CREATE TABLE [IF NOT EXISTS] [db_name1.]table_name AS [db_name2.] table_name2
[ENGINE = engine]
# 不仅会根据SELECT⼦句建⽴相应的表结构，同时还会将SELECT⼦句查询的数据顺带写⼊
CREATE TABLE [IF NOT EXISTS] [db_name.]table_name ENGINE = engine AS SELECT
~~~

#### 3.2.3 默认值表达式

- 表字段⽀持三种默认值表达式的定义⽅法，分别是DEFAULT、MATERIALIZED（物化表达式）和ALIAS（别名）。
  - 数据写入：在数据写⼊时，只有DEFAULT类型的字段可以出现在INSERT语句中。⽽MATERIALIZED和ALIAS都不能被显式赋值，它们只能依靠计算取值。
  - 数据查询：在数据查询时，只有DEFAULT类型的字段可以通过SELECT *返回。⽽MATERIALIZED和ALIAS类型的字段不会出现在SELECT *查询的返回结果集中。
  - 数据存储：在数据存储时，只有DEFAULT和MATERIALIZED类型的字段才支持持久化。

~~~mysql
CREATE TABLE dfv_v1 (
    id String,
    c1 DEFAULT 1000,
    c2 String DEFAULT c1
) ENGINE = TinyLog
# 修改默认值
ALTER TABLE [db_name.]table MODIFY COLUMN col_name DEFAULT value
~~~

#### 3.2.4 临时表

- 只支持Memory表引擎，不属于任何数据库
- 通过添加TEMPORARY关键字实现

~~~mysql
CREATE TEMPORARY TABLE [IF NOT EXISTS] table_name (
    name1 [type] [DEFAULT|MATERIALIZED|ALIAS expr],
    name2 [type] [DEFAULT|MATERIALIZED|ALIAS expr],
)
~~~

#### 3.2.5 分区表

- ⽬前只有合并树（MergeTree）家族系列的表引擎才⽀持数据分区。

~~~mysql
CREATE TABLE partition_v1 (
    ID String,
    URL String,
    EventTime Date
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(EventTime)
ORDER BY ID;
#
INSERT INTO partition_v1 VALUES
('A000','www.nauu.com', '2019-05-01'),
('A001','www.brunce.com', '2019-06-02');
# 
SELECT table,partition,path from system.parts WHERE table = 'partition_v1';
~~~

#### 3.2.6 视图

- 普通视图是SELECT查询的映射。
- 物化视图支持表引擎，在视图创建好之后，如果原表被写入新数据，物化实物也同步更新。物化视图⽬前并不⽀持同步删除，如果在源表中删除了数据，物化视图的数据仍会保留。本质是一张特殊的数据表。

### 3.3 数据表的基本操作

- ⽬前只有MergeTree、Merge和Distributed这三类表引擎⽀持ALTER查询

~~~mysql
# 增加字段
ALTER TABLE tb_name ADD COLUMN [IF NOT EXISTS] name [type] [default_expr] [AFTER name_after]
# 修改字段
ALTER TABLE tb_name MODIFY COLUMN [IF EXISTS] name [type] [default_expr]
# 修改备注
ALTER TABLE tb_name COMMENT COLUMN [IF EXISTS] name 'some comment'
# 删除字段
ALTER TABLE tb_name DROP COLUMN [IF EXISTS] name
# RENAME类似linux的mv命令 RENAME可以修改数据表的名称，如果将原始数据库与⽬标数据库设为不同的名称，那么就可以实现数据表在两个数据库之间
# 移动的效果
RENAME TABLE [db_name11.]tb_name11 TO [db_name12.]tb_name12, [db_name21.]tb_name21
TO [db_name22.]tb_name22, ...
# 清空表数据但不删除表
TRUNCATE TABLE [IF EXISTS] [db_name.]tb_name
~~~

### 3.4 数据分区的基本操作

- ClickHouse内置了许多system系统表，⽤于查询自身的状态信息。其中parts系统表专⻔⽤于查询数据表的分区信息。
- partition_id或者name等同于分区的主键，可以基于它们的取值确定⼀个具体的分区

~~~mysql
# 查询
SELECT partition_id,name,table,database FROM system.parts WHERE table = 'partition_v2';
# 删除
ALTER TABLE tb_name DROP PARTITION partition_expr
# 复制 replace 要求两表拥有相同的分区键、结构完全一样
ALTER TABLE B REPLACE PARTITION partition_expr FROM A
# 重置分区数据 clear
ALTER TABLE tb_name CLEAR COLUMN column_name IN PARTITION partition_expr
# 卸载分区 detach 分区被卸载后，它的物理数据并没有删除，⽽是被转移到了当前数据表⽬录的detached⼦⽬录下。
ALTER TABLE tb_name DETACH PARTITION partition_expr
# 装载分区 attach	⽽装载分区则是反向操作，它能够将detached⼦⽬录下的某个分区重新装载回去
# 常⽤于分区数据的迁移和备份场景
ALTER TABLE tb_name ATTACH PARTITION partition_expr
~~~

### 3.5 数据写入

- 三种

~~~mysql
# 常规插入
INSERT INTO [db.]table [(c1, c2, c3…)] VALUES (v11, v12, v13…), (v21, v22, v23…),...
# 导入指定格式
INSERT INTO [db.]table [(c1, c2, c3…)] FORMAT format_name data_set
# SELECT
INSERT INTO [db.]table [(c1, c2, c3…)] SELECT ...
# VALUES和SELECT⼦句的形式都⽀持声明表达式或函数
~~~

- 因为ClickHouse所有数据操作都是面向Block数据块的。所以最大数据块行数以内的写入操作是原子的。
  - 只有ClickHouse-Server处理数据具有原子写入特征，如HTTP接口。
  - CLI命令行或INSERT SELECT子句写入时则不具备该特性。

### 3.6 数据的删除和修改

- ClickHouse提供了DELETE和UPDATE的能力，这类操作被称为Mutation查询，可以看作ALTER语句的变种。
- Mutation语句是⼀种“很重”的操作，更**适⽤于批量数据的修改和删除**。
- **不支持事务**，一旦语句提交，就会产生影响，无法回滚。
- 是**异步**的，会立即返回。

~~~mysql
# 删除 是异步的，不会立即删除！！！
/* 
异步的实现方法：
1.每执⾏⼀条ALTER DELETE语句，都会在mutations系统表中⽣成⼀条对应的执行计划，当is_done等于1时表⽰执⾏完毕。
SELECT database, table ,mutation_id, block_numbers.number as num ,is_done FROM system.mutations
2.与此同时，在数据表的根⽬录下，会以mutation_id为名⽣成与之对应的日志⽂件用于记录相关信息
3.数据删除的过程是以数据表的每个分区目录为单位，将所有目录重写为新的目录，新目录的命名规则是在原有名称上加上system.mutations.block_numbers.number
4.数据在重写的过程中会将需要删除的数据去掉。旧的数据目录并不会⽴即删除，⽽是会被标记成非激活状态（active为0）
5.等到MergeTree引擎的下⼀次合并动作触发时，这些⾮激活⽬录才会被真正从物理意义上删除
*/
ALTER TABLE [db_name.]table_name DELETE WHERE filter_expr

# 更新 UPDATE⽀持在⼀条语句中同时定义多个修改字段，分区键和主键不能作为修改字段
ALTER TABLE partition_v2 UPDATE URL = 'www.wayne.com',OS = 'mac' WHERE ID IN
(SELECT ID FROM partition_v2 WHERE EventTime = '2019-06-01')
~~~



## 4.数据字典

- 数据字典以键和属性映射的形式定义数据。字典中的数据常驻内存，适合保存常量和常使用的维度表数据。
- 字典分为内置和扩展两种。
  - 用户通过自定义配置实现的字典叫外部扩展字典。
  - 默认自带的叫内置字典。

- 正常情况下，字典数据只能通过字典函数访问。
- 字典表引擎是特例，在该引擎下，数据字典可以挂载到一张代理的数据表下，实现数据表和字典数据的JOIN查询 。

### 4.1 内置字典

- ClickHouse目前的内置字典，只是提供了字典的定义机制和取数函数，⽽没有内置任何现成的数据。

### 4.2 外部扩展字典

- **数据字典能够有效地帮助我们消除不必要的JOIN操作（例如根据ID转名称），优化SQL查询，为查询性能带来质的提升。**

- 外部扩展字典是以插件形式注册到ClickHouse中的，有用户自定义数据模式和来源。目前支持7种类型的内存布局和4类数据来源。
  - 7：flat、hashed、cache、complex_key_hashed和complex_key_cache、range_hashed、ip_trie。
  - 4：本地文件、可执行文件、HTTP(s)、DBMS

- 在默认的情况下，ClickHouse会自动识别并加载/etc/clickhouseserver⽬录下所有以_dictionary.xml结尾的配置⽂件。同时ClickHouse⽀持不停机在线更新配置⽂件。
- 在单个字典配置文件内可以定义多个字典，其中每⼀个字典由⼀组dictionary元素定义。在dictionary元素之下⼜分为5个子元素，均为必填项。

~~~xml
<?xml version="1.0"?>
<dictionaries>
    <dictionary>
        <name>dict_name</name>
        <structure>
        	<!-- 字典的数据结构 -->
        </structure>
        <layout>
        	<!-- 在内存中的数据格式类型 -->
        </layout>
        <source>
        	<!-- 数据源配置 -->
        </source>
        <lifetime>
        	<!-- 字典的自动更新频率 -->
        </lifetime>
    </dictionary>
    <!-- 省略... -->
</dictionaries>

<!-- structure -->
<structure>
    <!-- id 或 key -->
    <id>
    <!-- Key属性 -->
    </id>
    <attribute>
    <!-- 字段属性 -->
    </attribute>
    <!-- ... -->
</structure>

<!-- structure实例 -->
<structure>
    <!-- 复合型 Tuple -->
    <key>
        <attribute>
            <name>field1</name>
            <type>String</type>
        </attribute>
        <attribute>
            <name>field2</name>
            <type>UInt64</type>
        </attribute>
    <!-- 省略... -->
    </key>
    <!-- 数值型 UInt64 -->
    <id>
        <!--名称自定义-->
        <name>Id</name>
    </id>
    
    <!-- attribute 一个到多个属性字段 -->
    <attribute>
        <name>Name</name>
        <type>DataType</type>
        <!-- 空字符串 -->
        <null_value></null_value>
        <expression>generateUUIDv4()</expression>
        <hierarchical>true</hierarchical>
        <injective>true</injective>
        <is_object_id>true</is_object_id>
    </attribute>
    <!-- 省略... -->
</structure>

<!-- layout实例 -->
<layout>
    <cache>
    <!-- 缓存大小 -->
    <size_in_cells>10000</size_in_cells>
    </cache>
</layout>

<!-- source实例 -->
<source>
    <!--  cache字典使用本地文件作为源，需要通过 executable形式 -->
    <executable>
        <command>cat /chbase/data/dictionaries/organization.csv</command>
        <format>CSV</format>
    </executable>
</source>

<!-- lifetime实例 -->
<lifetime>
    <!-- 时间区间内随机触发更新，在线更新，无须重启服务，单位是s -->
    <min>300</min>
    <max>360</max>
</lifetime>
~~~

~~~mysql
SELECT name,type,key,attribute.names,attribute.types FROM system.dictionaries
~~~

#### 4.2.1 内存布局

- 根据key键类型的不同，可以将它们划分为两类：⼀类是以flat、hashed、range_hashed和cache组成的单数值key类型，因为它们均使⽤单个数值型的id；另⼀类则是由complex_key_hashed、complex_key_cache和ip_trie组成的复合key类型。complex_key_hashed和complex_key_cache字典在功能⽅⾯与hashed和cache并⽆⼆致，只是单纯地将数值型key替换成了复合型key⽽已。

> flat

- 性能最高，只能使用UInt64数值型key。
- 在内存中使用数组结构保存，数组初始大小1024，最大限制500000。

> hashed

- 只能使用UInt64数值型key。
- 在内存中通过散列结构保存，没有存储上限。

> range_hashed

- 它在hashed的基础上增加了指定时间区间的特性，数据会以散列结构存储并按照时间排序。时间区间通过range_min和range_max元素指定，所指定的字段必须是Date或者DateTime类型。

> cache

- 只能使用UInt64数值型key。
- 它的字典数据在内存中会通过固定长度的向量数组保存。定长的向量数组又称cells，它的数组长度由size_in_cells指定。
- 当从cache字典中获取数据的时候，它⾸先会在cells数组中检查该数据是否已被缓存。如果数据没有被缓存，它才会从源头加载数据并缓存到cells中。
- 因为性能不稳定，所以如果⽆法做到99%或者更⾼的缓存命中率，则最好不要使⽤此类型。

> complex_key_hashed

- complex_key_hashed字典在功能方面与hashed字典完全相同，只是将单个数值型key替换成了复合型。

> complex_key_cache

- complex_key_cache字典同样与cache字典的特性完全相同，只是将单个数值型key替换成了复合型.

> ip_trie

- ip_trie字典的key只能指定单个String类型的字段，⽤于指代IP前缀。ip_trie字典的数据在内存中使⽤trie树结构保存，且专门⽤于IP前缀查询的场景。

#### 4.2.2 数据源

- 本地文件、可执行文件、远程文件和三种DBMS(MySQL、MongoDB、ClickHouse)。

#### 4.2.3 数据更新策略

- 扩展字典支持数据的在线更新，更新后无须重启服务。
- 更新频率是配置文件中lifetime指定时间间隔内的随机数。
- 字典内部有版本的概念。更新过程中，旧版本会持续提供服务，直至更新完成，新版本才会替代旧版本。
- 部分数据源可以依据previous标识判断数据源是否发生实质变化，以此跳过固定更新节奏。
  - 系统文件的修改时间
  - DB的字段updatetime

~~~mysql
SYSTEM RELOAD DICTIONARY [dict_name]
~~~

#### 4.2.4 基本操作

~~~mysql
# 字典查询
SELECT name, type, key, attribute.names, attribute.types, source FROM system.dictionaries;
# 数据查询 如果字典使⽤了复合型key，则需要使⽤元组作为参数传⼊
SELECT dictGet('dict_name','attr_name',key);
# IPv4格式为A.B.C.D IPv6格式同下
SELECT dictGetString('test_ip_trie_dict', 'asn', tuple(IPv6StringToNum('2620:0:870::')))
# 通过字典表获取字典数据
CREATE TABLE tb_test_flat_dict (
    id UInt64,
    code String,
    name String
) ENGINE = Dictionary(test_flat_dict);
# DDL创建字典
CREATE DICTIONARY test_dict(
    id UInt64,
    code String,
    name String
)
PRIMARY KEY id
LAYOUT(FLAT())
SOURCE(FILE(PATH '/var/lib/clickhouse/user_files/organization.csv' FORMAT CSV))
LIFETIME(320)
~~~



## 5.MergeTree原理解析

- MergeTree不是LSM-Tree，因为它不包含memtable和log，插入的数据直接写入文件系统，仅适合于批量插入，不适合频繁插入。

### 5.1 MergeTree的创建和存储

- MergeTree表的创建方法

~~~mysql
CREATE TABLE [IF NOT EXISTS] [db_name.]table_name (
    name1 [type] [DEFAULT|MATERIALIZED|ALIAS expr],
    name2 [type] [DEFAULT|MATERIALIZED|ALIAS expr]
) ENGINE = MergeTree()
[PARTITION BY expr]	# 支持单个列字段、元组和列表达式
[ORDER BY expr]		# 默认情况下排序键和主键相同，建议用排序键来替代主键 支持多字段
[PRIMARY KEY expr]	# 会依照主键生成一级索引	MergeTree主键允许存在重复数据
[SAMPLE BY expr]	# 抽样表达式 用于声明数据以何种标准进行采样。主键的配置中也需要声明同样的表达式
[SETTINGS name=value, 省略...]	# 设置索引粒度、TTL、多路径存储策略等
~~~

- 数据表的物理层级：数据表目录、分区目录及各分区下具体的数据文件。

### 5.2 数据分区

> 分区ID的生成规则

- 不指定分区键：没有用PARTITION BY声明，所有数据都会被写入all分区。
- 整型分区键：直接转化为该整型的字符形式作为分区ID
  - 无法转换为日期类型YYYYMMDD格式
- 日期类型：转化为YYYYMMDD格式作为分区ID
  - 分区键能转化为YYYYMMDD格式，也按本规则
- 其他类型：通过128位Hash算法取其Hash值作为分区ID的取值

~~~mysql
PARTITION BY (length(Code),EventTime)
# 2-20190501
# 2-20190611
~~~

> 分区目录命名规则

- 完整分区目录的命名公式：PartitionID_MinBlockNum_MaxBlockNum_Level。
- MinBlockNum_MaxBlockNum：数据块最小编号与数据块最大编号，是一个全局自增的整型编号。
  - 新分区的MinBlockNum和MaxBlockNum是一样的。
  - 同一个分区的多个目录及其子文件，会合并成新目录。MinBlockNum取所有目录中最小的MinBlockNum值。Max同理。
  - 它不是Block数，是编号。

- Level：分区合并的次数。新目录初始值为0，分区合并后，取多目录最大Level值+1。

> 分区目录合并过程

- 分区目录是在数据写入过程中被创建的。不是在数据表被创建之后就存在的。
- 伴随着每一批数据的写入（一次INSERT语句），MergeTree都会生成一批新的分区。也就是说，同一个分区也会存在多个分区目录的情况。
- 写入后10~15分钟，ClickHouse会通过后台任务再将属于相同分区的多个目录合并成一个新的目录。已经存在的旧分区目录并不会立即被删除，而是在之后的某个时刻通过后台任务被删除（默认8分钟）。该目录会处于未激活状态（active=0）。
- 属于同一个分区的多个目录，在合并之后会生成一个全新的目录，目录中的索引和数据文件也会相应地进行合并。

### 5.3 一级索引

- 稀疏索引占用空间小，所以primary.idx内的索引数据常驻内存。
- 多个主键的索引数据是对应列值的拼接。
- MarkRange在ClickHouse中是用于定义标记区间的对象。MergeTree按照index_granularity的间隔粒度，将⼀段完整的数据划分成
  了多个⼩的间隔数据段，⼀个具体的数据段即是⼀个MarkRange。
- **索引查询** 其实就是两个数值区间的交集判断。其中，⼀个区间是由基于主键的查询条件转换⽽来的条件区间；⽽另⼀个区间是刚才所讲述的与MarkRange对应的数值区间。
  - 生成查询条件区间：首先，将查询条件转换为条件区间。
  - 递归交集判断：以递归的形式，依次对MarkRange的数值区间与条件区间做交集判断。从最⼤的区间开始。
    - 如果不存在交集，剪枝。
    - 如果存在交集，且MarkRange步长大于8个index_granularity，则将此区间拆分成8个子区间。继续递归。
    - 如果存在交集，且不大于8个，则记录MarkRange并返回。
  - 合并MarkRange区间：将最终匹配的MarkRange聚在一起，合并它们的范围。

### 5.4 二级索引

- 二级索引又称跳数索引，由数据的聚合信息构建而成。

~~~mysql
# 支持使用元组和表达式。
INDEX index_name expr TYPE index_type(...) GRANULARITY granularity
~~~

- granularity定义了一行跳数索引能跳过多少个index_granularity区间的数据。
- MergeTree共支持4种跳数索引，分别是minmax、set、ngramf_v1和tokenf_v1。

### 5.5 数据存储

- 列存储，每个列都有一个对应的.bin数据文件。.bin文件保存了当前分区内的某列数据。
- 数据会在（1）按order by排序和（2）默认经过LZ4算法压缩 后被写入.bin文件。
- 压缩数据由多个压缩数据块组成。
- 每个压缩数据块由头信息和压缩数据两部分组成。
  - 头信息9个字节：压缩方法1个UInt8、压缩后前数据字节尺寸4个UInt32。
  - 压缩数据块在压缩前 体积在64KB~1MB之间。写入数据会按该区间进行聚合和划分。

### 5.6 数据标记

- **数据标记和索引区间是对齐的**，均按照index_granularity粒度间隔。
- 每个.bin对应一个.mrk数据标记文件。
- 一行标记数据包括 压缩数据块的起始偏移量 和 未压缩数据的起始偏移量。
- 标记数据不会常驻内存，采用LRU缓存策略加速。
- 数据读取分为两个步骤：读取压缩数据块和读取数据。

### 5.7 分区、索引、标记和压缩数据的协同

- 每8192行数据的某列会凑成一个逻辑块，对应一个稀疏索引和一个标记数据。
- 每64KB~1MB的数据会凑成一个物理块，对应一个压缩块。
- 那么就存在逻辑块和物理块的一对一、一对多和多对一关系。映射方法由标记数据指出。

- 块偏移量和块内偏移量（压缩文件中的偏移量和解压缩块的偏移量）
  - 块是按未压缩数据的size(64KB~1MB)划分的。
- MergeTree可以以多线程形式同时读取多个压缩数据块，以提高性能。

> 下次合并是如何在本次合并结果之上重新编排压缩文件数据位置的？

- 



## 6.MergeTree系列表引擎

- 六大表引擎：合并树、外部存储、内存、文件、接口和其他。
- MergeTree系列表引擎会在Merge的时候触发特殊处理逻辑。
  - ReplacingMergeTree、SummingMergeTree、AggregatingMergeTree、CollapsingMergeTree和VersionedCollapsingMergeTree。

### 6.1 MergeTree

- TTL：MergeTree表引擎支持 表和列 级别的数据TTL。到时间删除整个表数据或列数据。
  - 设置TTL后，会生成ttl.txt文件，以json格式保存 列或表的信息 和 时间戳区间。
  - MergeTree合并分区时，触发删除TTL过期数据的逻辑。
  - 选择删除的分区时用的贪心算法，贪心选择策略是最早过期同时最老的分区。
  - 列被删除后不会参与Merge。

~~~mysql
# TTL = time_col时间的三天之后
TTL time_col + INTERVAL 3 DAY;
# SECOND、MINUTE、HOUR、DAY、WEEK、MONTH、QUARTER和YEAR
TTL time_col + INTERVAL 1 MONTH;

# 定义表的时候 设置列TTL
CREATE TABLE ttl_table_v1(
    id String,
    create_time DateTime,
    code String TTL create_time + INTERVAL 10 SECOND,
    type UInt8 TTL create_time + INTERVAL 10 SECOND
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(create_time)
ORDER BY id;
# optimize强制触发TTL清理 FINAL表示 所有分区合并 都被触发
optimize TABLE ttl_table_v1 FINAL;
# 候补列TTL
ALTER TABLE ttl_table_v1 MODIFY COLUMN code String TTL create_time + INTERVAL 1 DAY;
ALTER TABLE ttl_table_v1 MODIFY COLUMN type UInt8 TTL create_time + INTERVAL 1 QUARTER;
# 暂时没有取消列TTL的方法

# 定义表的时候 设置表TTL
CREATE TABLE ttl_table_v2(
    id String,
    create_time DateTime,
    code String TTL create_time + INTERVAL 1 MINUTE,
    type UInt8
)ENGINE = MergeTree
PARTITION BY toYYYYMM(create_time)
ORDER BY create_time
TTL create_time + INTERVAL 1 DAY;
# 候补表TTL
ALTER TABLE ttl_table_v2 MODIFY TTL create_time + INTERVAL 3 DAY;
# 暂时没有取消表TTL的方法
~~~

- 多路径存储策略：数据分区为划分，将分区目录写在一个分区指定的多个磁盘目录下。
  - JBOD策略、HOT/COLD策略
- 默认策略是写在config.xml配置中指定的path路径下。
- JBOD策略：非RAID的多块磁盘情况下，执行INSERT和MERGE时，轮询写入各磁盘。

~~~xml
<!-- config.xml -->
<storage_configuration>
    <!-- ⾃定义磁盘配置 -->
    <disks>
        <disk_hot1> <!--⾃定义磁盘名称 -->
        	<path>/chbase/data</path>
        </disk_hot1>
        <disk_hot2>
        	<path>/chbase/hotdata1</path>
        </disk_hot2>
        <disk_cold>
        	<path>/chbase/cloddata</path>
        	<keep_free_space_bytes>1073741824</keep_free_space_bytes>
        </disk_cold>
    </disks>
    <!-- ⾃定义策略配置 -->
    <policies>
        <default_jbod> <!--⾃定义策略名称 -->
            <volumes>	<!-- 卷组/磁盘组 -->
                <jbod> <!-- ⾃定义名称 磁盘组 -->
                	<disk>disk_hot1</disk>
                	<disk>disk_hot2</disk>
                </jbod>
            </volumes>
        </default_jbod>
    </policies>
</storage_configuration>
~~~

~~~shell
# 授权读写磁盘
sudo chown clickhouse:clickhouse -R /chbase/cloddata /chbase/hotdata1
# 重启服务
sudo service clickhouse-server restart
~~~

~~~mysql
# 查询disks
SELECT
    name,
    path,formatReadableSize(free_space) AS free,
    formatReadableSize(total_space) AS total,
    formatReadableSize(keep_free_space) AS reserved
FROM system.disks;
# storage_policies
SELECT policy_name,
    volume_name,
    volume_priority,
    disks,
    formatReadableSize(max_data_part_size) max_data_part_size ,
    move_factor
FROM system.storage_policies;
# 
CREATE TABLE jbod_table(
	id UInt64
)ENGINE = MergeTree()
ORDER BY id
SETTINGS storage_policy = 'default_jbod';
~~~

- HOT/COLD策略：挂载不同类型磁盘的场景下，HOT区用SSD这种高性能磁盘，注重性能，COLD区用HHD这种高容量磁盘，注重经济性。
  - 数据在写⼊MergeTree之初，⾸先会在HOT区域创建分区目录用于保存数据，当分区数据大小累积到阈值（1MB）时，数据会自行移动到COLD区域。

~~~xml
<policies>

    <moving_from_hot_to_cold><!--⾃定义策略名称 -->
        <volumes>
            <hot><!--⾃定义名称 ,hot区域磁盘 -->
                <disk>disk_hot1</disk>
                <max_data_part_size_bytes>1073741824</max_data_part_size_bytes>
            </hot>
            <cold><!--⾃定义名称 ,cold区域磁盘 -->
            	<disk>disk_cold</disk>
            </cold>
        </volumes>
        <move_factor>0.2</move_factor>
    </moving_from_hot_to_cold>
</policies>
~~~

- MergeTree当前支持分区目录在卷组内或者卷组间移动

~~~mysql
ALTER TABLE hot_cold_table MOVE PART 'all_1_2_1' TO DISK 'disk_hot1';
ALTER TABLE hot_cold_table MOVE PART 'all_1_2_1' TO VOLUME 'cold';
~~~

### 6.2 ReplacingMergeTree

- ReplacingMergeTree：或许叫去重MergeTree。因为MergeTree主键没有唯一性约束，所以可能存在重复数据，该引擎用于合并分区时去重。
  - ENGINE = ReplacingMergeTree(version)。version是可选的UInt*、Date或DateTime字段，决定了去重算法。
  - 去除重复数据：按ORDER BY分组后，每个分组只保留最后一条或版本字段最大的那一条数据。
  - 去除只会在相同分区内发生，不同数据分区的重复数据不会被剔除。

### 6.3 SummingMergeTree

- SummingMergeTree：或许叫求和MergeTree。能够在合并分区的时候按照预先定义的条件聚合汇总数据，将同一分组下的多行数据汇总合并成一行，这样既减少了数据行，又降低了后续汇总查询的开销。

- SummingMergeTree和AggregatingMergeTree的聚合都是根据ORDER BY进行的，所以如果主键和ORDER BY指定的字段不一样时，应该明确声明PRIMARY KEY。除此之外，都用ORDER BY替代。
  - PRIMARY KEY列字段必须是ORDER BY的前缀。

- ENGINE = SummingMergeTree((col1,col2,…))；col1、col2是⼀个选填参数，⽤于设置除主键外的其他数值类型字段，以指定被SUM汇总的列字段。如若不填写此参数，则会将所有⾮主键的数值类型字段进行SUM汇总。
  - 非汇总字段则会使用第一行数据的取值。
- SummingMergeTree也支持嵌套类型的字段，在使用嵌套类型字段时，需要被SUM汇总的字段名称必须以Map后缀结尾。
  - 默认情况下，会以嵌套类型中第⼀个字段作为聚合条件Key。
  - 为了使用复合Key，在嵌套类型的字段中，除第一个字段以外，任何名称是以Key、Id或Type为后缀结尾的字段，都将和第一个字段一起组成复合Key。

~~~mysql
CREATE TABLE summing_table_nested(
    id String,
    nestMap Nested(
        id UInt32,
        key UInt32,
        val UInt64
    ),
    create_time DateTime
)ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(create_time)
ORDER BY id;
~~~

### 6.4 AggregatingMergeTree

- AggregatingMergeTree：或许叫聚合MergeTree。能够在合并分区的时候，按照预先定义的条件聚合数据。并通过二进制的格式存在表内
- 将同一分组下的多行数据聚合成一行，既减少了数据行，又降低了后续聚合查询的开销。

~~~mysql
ENGINE = AggregatingMergeTree();
# AggregateFunction是ClickHouse提供的一种特殊的数据类型，它能够以二进制的形式存储中间状态结果
CREATE TABLE agg_table(
    id String,
    city String,
    code AggregateFunction(uniq,String),
    value AggregateFunction(sum,UInt32),
    create_time DateTime
)ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(create_time)
ORDER BY (id,city)
PRIMARY KEY id;
# AggregateFunction类型的列字段，在写⼊数据时，需要调⽤*State函数；⽽在查询数据时，则需要调⽤相应的*Merge函数
INSERT INTO TABLE agg_table
SELECT 'A000','wuhan', uniqState('code1'), sumState(toUInt32(100)), '2019-08-10 17:00:00';
# 
SELECT id,city,uniqMerge(code),sumMerge(value) FROM agg_table GROUP BY id,city;
~~~

- AggregatingMergeTree更为常见的应用方式是结合物化视图使用，将它作为物化视图的表引擎。
- 通常会使⽤MergeTree作为底表，用于存储全量的明细数据，并以此对外提供实时查询。
- 接着，新建⼀张物化视图：

~~~mysql
CREATE MATERIALIZED VIEW agg_view
ENGINE = AggregatingMergeTree()
PARTITION BY city
ORDER BY (id,city)
AS SELECT
    id,
    city,
    uniqState(code) AS code,
    sumState(value) AS value
FROM agg_table_basic
GROUP BY id, city;
# 新增数据时，数据会自动同步到物化视图
~~~

### 6.5 CollapsingMergeTree

- CollapsingMergeTree：折叠合并树。通过以增代删（改）的思路，支持行级数据修改和删除的表引擎。它通过定义一个sign标记位字段，记录数据行的状态。如果sign标记为1，则表示这是一行有效的数据；如果sign标记为-1，则表示这行数据需要被删除。sign标记为1和-1的一组数据会被抵消删除。

- 折叠规则：
  - 如果sign=1比sign=-1的数据多一行，则保留最后一行sign=1的数据。
  - 如果sign=-1比sign=1的数据多一行，则保留第一行sign=-1的数据。
  - 如果sign=1和sign=-1的数据行⼀样多，并且最后一行是sign=1，则保留第一行sign=-1和最后一行sign=1的数据。
  - 如果sign=1和sign=-1的数据行⼀样多，并且最后一行是sign=-1，则什么也不保留。
  - 其他情况，CH会打印警告日志，但不会报错，查询结果未知。

~~~mysql
CREATE TABLE collpase_table(
    id String,
    code Int32,
    create_time DateTime,
    sign Int8
)ENGINE = CollapsingMergeTree(sign)
PARTITION BY toYYYYMM(create_time)
ORDER BY id;

-- 修改前的源数据, 它需要被修改
INSERT INTO TABLE collpase_table VALUES('A000',100,'2019-02-20 00:00:00',1);
-- 镜像数据, ORDER BY字段与源数据相同(其他字段可以不同),sign取反为-1,它会和源数据折叠
INSERT INTO TABLE collpase_table VALUES('A000',100,'2019-02-20 00:00:00',-1);
-- 修改后的数据 ,sign为1
INSERT INTO TABLE collpase_table VALUES('A000',120,'2019-02-20 00:00:00', 1);
                                        
-- 修改前的源数据, 它需要被删除
INSERT INTO TABLE collpase_table VALUES('A000',100,'2019-02-20 00:00:00',1);
-- 镜像数据, ORDER BY字段与源数据相同, sign取反为-1, 它会和源数据折叠
INSERT INTO TABLE collpase_table VALUES('A000',100,'2019-02-20 00:00:00',-1);
~~~

- 因为Merge时才删除数据，所以查询会获得旧数据。要么查前强制触发合并，要么如下：

~~~mysql
SELECT id,SUM(code * sign),COUNT(code * sign),AVG(code * sign),uniq(code * sign)
FROM collpase_table
GROUP BY id
HAVING SUM(sign) > 0;
~~~

- 如上折叠规则，如果先写sign=-1，再写sign=1，就不能折叠。VersionedCollapsingMergeTree解决了这个问题。

### 6.6 VersionedCollapsingMergeTree

- VersionedCollapsingMergeTree：版本折叠合并树。功能和CollapsingMergeTree一样。但对数据的写⼊顺序没有要求，在同⼀个分区内，任意顺序的数据都能够完成折叠操作。
- ENGINE = VersionedCollapsingMergeTree(sign,ver)。在定义ver字段之后，VersionedCollapsingMergeTree会自动将ver作为排序条件并增加到ORDER BY的末尾。这样，同一版本的1和-1就会严格相邻。

### 6.7 总结

- 在具体的实现逻辑部分，7种MergeTree共用一个主体，在触发Merge动作时，它们调用了各自独有的合并逻辑。其他6个MergeTree的合并逻辑都继承于MergeTree的MergingSortedBlockInputStream。
- MergingSortedBlockInputStream的主要作用是按照ORDER BY的规则保持新分区数据的有序性。
- 在7种MergeTree的基础上+Replicated前缀，能组合出另外7种拥有副本协同能力的表引擎。



## 7.其他类型表引擎

### 7.1 外部存储类型

- 外部存储表引擎直接从其他的存储系统读取数据，数据文件由外部系统提供，表引擎只负责元数据管理和数据查询。
  - MySQL、HDFS、Kafka、File、JDBC等。
  - 可以通过ClickHouse的SQL语句写入数据到外部存储系统。

### 7.2 内存类型

- 内存类型表引擎会在数据表被加载时，将所有数据载入内存，以供查询。
  - Memory和Buffer不支持物理存储，Set和Join会写盘。

- Memory表引擎的数据在内存中的形态和查询时看到的一致，不做压缩和格式转换。支持并行查询，简单查询性能可以媲美MergeTree。
  - 应用场景：ClickHouse内部用作集群间分发数据的载体。

- Set表引擎的数据会同步到磁盘，数据写入时自动去重。它只能作IN查询的右侧条件，不能直接SELECT。
- Join表引擎为JOIN查询而生，也可直接SELECT。
- Buffer表引擎会充当缓冲区的角色。写入数据时，如果并发量大，导致MergeTree的合并操作慢于写速度，可以先写入Buffer表，再溢出到MergeTree。

### 7.3 日志类型

- 日志类型：适合小数量，“一次”写入多次查询的场景。无索引、不分区、不能并发读写、阻塞写、会写盘。
  - TinyLog、StripeLog~（条纹）~、Log的性能依次递增。
- TinyLog：元数据+分列数据文件。按列存储但没有标记数据，所以不能并行读。
- StripeLog：元数据+合并数据文件+标记数据。可以并行查询，所有列合并存储。
- Log：元数据+分列数据文件+标记数据。按列存储且可以并行查询。

### 7.4 接口类型

- 该类表引擎不存储任何数据，是其他表引擎上层的接口。
  - Merge、Dictionary、Distributed。
- Merge：异步并行地代理查询任意数量的相同结构表，最终合成结果集返回。
  - 各表的表引擎和分区定义可以不同。

- Dictionary：生成数据字典的代理表。
- Distributed：相当于分布式数据库分片方案的中间件，用于集群内部自动开展数据的写入分发和查询路由工作。

### 7.5 其他类型

- Live View：不是表引擎，是一种视图，用于监听SQL查询结果，实时反馈目标数据。
- Null：向Null表写数据，系统会正确返回，但是Null表会自动忽略数据，永远不会将它们保存。如果向Null表发起查询，那么它将返回空表。
  - 常和物化视图一起用。
- URL：等价于HTTP客户端，它可以通过HTTP/HTTPS协议，直接访问远端的REST服务。当执行SELECT查询的时候，底层会将其转换为GET请求的远程调用。当执行INSERT查询的时候，会将其转换为POST请求的远程调用。



## 8.数据查询

~~~mysql
# 查询语句格式
[WITH expr |(subquery)]
SELECT [DISTINCT] expr
[FROM [db.]table | (subquery) | table_function] [FINAL]
[SAMPLE expr]
[[LEFT] ARRAY JOIN]
[GLOBAL] [ALL|ANY|ASOF] [INNER | CROSS | [LEFT|RIGHT|FULL [OUTER]] ] JOIN
(subquery)|table ON|USING columns_list
[PREWHERE expr]
[WHERE expr]
[GROUP BY expr] [WITH ROLLUP|CUBE|TOTALS]
[HAVING expr]
[ORDER BY expr]
[LIMIT [n[,m]]
[UNION ALL]
[INTO OUTFILE filename]
[FORMAT format]
[LIMIT [offset] n BY columns]
~~~

### 8.1 with子句

~~~mysql
WITH ( round(database_disk_usage) ) AS database_disk_usage_v1
SELECT database,database_disk_usage, database_disk_usage_v1
FROM (
	-- 嵌套
    WITH (
    	SELECT SUM(data_uncompressed_bytes) FROM system.columns
    ) AS total_bytes
    SELECT database, (SUM(data_uncompressed_bytes) / total_bytes) * 100 AS database_disk_usage FROM system.columns
    GROUP BY database
    ORDER BY database_disk_usage DESC
)
~~~

### 8.2 from子句

~~~mysql
SELECT number FROM numbers(5);
# 省略from 会从虚拟表system.one中取数
# from后 可以跟Final修饰符，配合折叠树，在查询时合并。
~~~

### 8.3 sample子句

- 查询仅返回采样数据，采样是幂等的。
- sample子句只能用于MergeTree系列引擎，且必须在create table时声明sample by表达式。

~~~mysql
# SAMPLE BY所声明的表达式必须同时包含在主键的声明
# Sample Key必须是Int类型
CREATE TABLE hits_v1 (
    CounterID UInt64,
    EventDate DATE,
    UserID UInt64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(EventDate)
ORDER BY (CounterID, intHash32(UserID))
-- Sample Key声明的表达式必须也包含在主键的声明中
SAMPLE BY intHash32(UserID)

# SAMPLE factor
SELECT count() * 10 FROM hits_v1 SAMPLE 0.1;
SELECT count() * any(_sample_factor) FROM hits_v1 SAMPLE 0.1;
# SAMPLE factor OFFSET n
SELECT CounterID,_sample_factor FROM hits_v1 SAMPLE 1/10 OFFSET 1/2;
# SAMPLE rows 实际采样的最小粒度有index_granularity索引粒度决定，小于该粒度的rows没意义
SELECT count() FROM hits_v1 SAMPLE 10000;
SELECT CounterID, _sample_factor FROM hits_v1 SAMPLE 100000 LIMIT 1;
~~~

### 8.4 array join子句

- 将一行数组展开成多行。
- 在数据表内部，与数组或嵌套类型的字段进行JOIN操作。
- 对多个数组字段进行ARRAY JOIN操作时，查询的计算逻辑是按行合并而不是产生笛卡尔积
- 目前支持INNER和LEFT两种JOIN策略

~~~mysql
# 数组类型
SELECT title,value,v FROM query_v1 ARRAY JOIN value AS v;
SELECT title,value,v FROM query_v1 LEFT ARRAY JOIN value AS v;
# 嵌套类型
SELECT title, nest.v1, nest.v2 FROM query_v2 ARRAY JOIN nest
~~~

### 8.5 join子句

- all/any/asof * left/right/full(outer)/inner/cross * join
- all：如果左表内一行在右表内有多行匹配，则返回右表全部连接的数据。
- any：如果左表内一行在右表内有多行匹配，则返回第一行连接的数据。
- asof：模糊查询，在连接键后追加一个模糊连接匹配条件。
- inner：内连接。
- outer：左外、右外、全外。
- cross：笛卡尔积

~~~mysql
SELECT a.id,a.name,b.rate FROM join_tb1 AS a ALL INNER JOIN join_tb2 AS b ON a.id = b.id;
SELECT a.id,a.name,b.rate FROM join_tb1 AS a ANY JOIN join_tb2 AS b ON a.id = b.id;
# asof 等价于 a.id = b.id AND a.time >= b.time
SELECT a.id,a.name,b.rate,a.time,b.time
FROM join_tb1 AS a ASOF INNER JOIN join_tb2 AS b
ON a.id = b.id AND a.time = b.time;
# asof using
SELECT a.id,a.name,b.rate,a.time,b.time
FROM join_tb1 AS a ASOFINNER JOIN join_tb2 AS b 
USING(id,time);
# asof_column必须是有序序列数据类型（整型、浮点型、日期）	且 连接键和asof_column不能是同一个字段

# LEFT RIGHT FULL CROSS
SELECT a.id,a.name,b.rate FROM join_tb1 AS a
FULL JOIN join_tb2 AS b ON a.id = b.id;
~~~

- CH会自动将关联查询转换为join查询
  - 如果查询语句中不包含WHERE条件，则会转为CROSS JOIN。
  - 如果查询语句中包含WHERE条件，则会转为INNER JOIN。

- JOIN查询的注意事项：
  - 左大右小：右表应该是小表，因为JOIN执行的时候，右表会被全部载入内存与左表进行比较。
  - JOIN没有缓存支持
  - ⼤量维度属性补全的查询场景中，则建议使用字典代替JOIN查询
  - 空值填充策略可以通过join_use_nulls策略制定

### 8.6 where和prewhere子句

- 如果where指定的过滤条件是主键字段，则能进一步借助索引加速查询。

- PREWHERE目前只能用于MergeTree系列的表引擎，同where功能一样。不同之处在于：使用PREWHERE时，首先只会读取PREWHERE指定的列字段数据，用于数据过滤的条件判断。待数据过滤之后再读取SELECT声明的列字段以补全其余属。

### 8.7 group by子句

- 如果SELECT后只声明了聚合函数，则可以省略GROUP BY关键字。
- 当聚合函数内的数据存在NULL值，NULL == NULL。
- 聚合查询可以结合WITH ROLLUP、WITH CUBE和WITH TOTALS三种修饰符获取额外的汇总信息。
- WITH ROLLUP：按照聚合键从右向左上卷数据，基于聚合函数依次⽣成分组小计和总计。
- WITH CUBE：会像立方体模型一样，基于聚合键之间所有的组合生成小计信息。如果设聚合键的个数为n，则最终小计组合的个数为2的n次。
- WITH TOTALS：会基于聚合函数对所有数据进行总计。

### 8.8 having子句

- 和group by联合使用，实现二次过滤。

~~~mysql
SELECT table ,avg(bytes_on_disk) as avg_bytes
FROM system.parts GROUP BY table
HAVING avg_bytes > 10000;
~~~

### 8.9 limit by子句

- 运行于order by之后和limit之前，按指定分组，返回最多前n行数据。
- 支持offset

~~~mysql
SELECT database,table,MAX(bytes_on_disk) AS bytes FROM system.parts
GROUP BY database,table ORDER BY database ,bytes DESC
LIMIT 3 OFFSET 1 BY database
~~~

### 8.10 limit子句

~~~mysql
SELECT database,table,MAX(bytes_on_disk) AS bytes FROM system.parts
GROUP BY database,table ORDER BY bytes DESC
LIMIT 3 BY database
LIMIT 10
~~~

### 8.11 select子句

- 不建议在列式数据库中 用 select *。
- 支持正则表达式。

~~~mysql
SELECT COLUMNS('^n'), COLUMNS('p') FROM system.databases;
~~~

### 8.12 DISTINCT子句

- DISTINCT子句能够去除重复数据，使⽤场景广泛。

### 8.13 UNION ALL子句

- UNION ALL子句能够联合左右两边的两组子查询，将结果⼀并返回。在⼀次查询中，可以声明多次UNION ALL以便联合多组查询。
- 左右子查询的要求：
  - 首先，列字段的数量必须相同；
  - 其次，列字段的数据类型必须相同或相兼容；
  - 最后，列字段的名称可以不同，查询结果中的列名会以左边的子查询为准。

### 8.14 查看SQL执行计划

- 通过将ClickHouse服务日志设置到DEBUG或者TRACE级别，可以变相实现EXPLAIN查询，以分析SQL的执行日志。
- 会真正执行查询。
- 所以，尽量使用LIMIT、分区索引、一级索引、二级索引、过滤条件什么的。

~~~mysql 
clickhouse-client -h <host> --port <port> --password <pass> --send_logs_level=trace <<< "SELECT * FROM DB.TABLE" > /dev/null
~~~



## 9.副本和分片

### 9.1 副本

- 只有使用了ReplicatedMergeTree复制表系列引擎，才能应用副本的能力。
- 副本特点：依赖Zookeeper、表级别的副本、多主架构、Block为基本写入单元、写入具有原子性和唯一性、

- 副本定义：ENGINE = ReplicatedMergeTree('zk_path', 'replica_name')
- zk_path用于指定在ZooKeeper中创建的数据表的路径，常见：/clickhouse/tables/{shard}/table_name
- replica_name的作用是定义在ZooKeeper中创建的副本名称，该名称是区分不同副本实例的唯⼀标识，常见：服务器域名

### 9.2 ReplicatedMergeTree

- ReplicatedMergeTree需要依靠ZooKeeper的事件监听机制以实现各个副本之间的协同。在每张ReplicatedMergeTree表的创建过程中，它会以zk_path为根路径，在Zoo-Keeper中为这张表创建一组监听节点。
- Log：常规操作日志节点(INSERT、MERGE和DROP PARTITION)。保存了副本需要执行的任务指令。每个副本实例都会监听/log节点 ，当有新的指令加入时，会把指令加入副本自己的任务队列，并执行任务。
- Mutations：执行ALTER DELETE和ALTER UPDATE查询时，指令被添加到这个节点。功能和log日志相似。
- Log和Mutations节点具体由Log-Entry和MutationEntry实现
- 副本协同核心流程主要有INSERT、MERGE、MUTATION和ALTER四种。INSERT和ALTER查询是分布式执行的。其他不支持分布式（SELECT、CREATE、DROP、RENAME和ATTACH）。
  - 可以使用 on cluster 语法进行DDL查询
- MERGE操作和MUTATION操作无论由哪个副本发起，都由主副本来指定。
- ALTER操作，谁执行谁负责，由发起节点负责对共享元数据的修改和各副本修改进度的监控。

### 9.3 分片

- Distributed表引擎作为透明代理，在集群内部自动开展数据的写入、分发、查询、路由等工作。它本身不存储任何数据，需要和其他数据表协同工作。
- 分布式存储的每个节点都是一个分片。但有N个相同数据节点叫N-1副本。

~~~mysql
# 分布式DLL
create/drop/rename/alter table on CLUSTER cluster_name;
# {shard}和{replica}两个动态宏变量代替了先前的硬编码方式
CREATE TABLE test_1_local ON CLUSTER shard_2(
id UInt64	-- 这里可以使用任意其他表引擎，
)ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/test_1', '{replica}')
ORDER BY id
~~~

### 9.4 Distributed

- 需要创建 _local后缀 的本地表，引擎随意。需要 _all后缀的分布式表，代理操作多张本地表。

- 分布式表和本地表间结构的一致性检查，Distributed表采用读时检查机制。

~~~mysql
ENGINE = Distributed(cluster, database, table, [,sharding_key])
~~~

- sharding_key 要是整型。以此为依据分片。

- 分布式表的查询分类：
  - 不支持MUTATION类型的操作
  - INSERT和SELECT会分布式地作用于local本地表
  - 部分元数据操作（CREATE、DROP、RENAME 和 ALTER的非分区操作），只作用于Distributed表，不用于本地表
- 分片权重越大，被写入数据越多。
- slot数量是所有分片权重之和。slot值确定了数据的分片位置。

- 选择函数：用于判断数据的分片位置。
  - 先找出slot值：slot = shard_value % sum_weight
  - 基于slot确定分片区

- 分布式写入的两种方法：
  - 外部计算系统事先将数据分片，然后该系统将数据写入各个本地表
  - Distributed表引擎代理写入分片
- Distributed写入步骤：
  - 在第一个分片（执行INSERT的）节点写入本地分片数据
  - 第一个分片连接远端分片，并将要发的数据写入分布式表存储目录下的临时bin文件
  - 第一个分片向远端发送数据
  - 远端接收并写入本地
  - 第一个分片确认写入完成
    - 同步写和异步写可选
- 副本复制数据：如果分片有副本，那么 要不就Distributed表写入副本，要么借助ReplicatedMergeTree表实现副本数据的分发。
  - 为了避免Distributed表的单点瓶颈，第二张更好点
- 借助ReplicatedMergeTree表实现副本数据的分发时，Distributed表选择shard的一个replica写入数据，ReplicatedMergeTree表复制数据到其他replicas。
  - replica选择的基本思路：借助ClickHouse服务节点的全局异常计时器，选择异常最少的那个replica。
  - 在上述基础上，对于相同计数的replicas：
    - random策略随机选择replica
    - nearest_hostname策略选择与当前host名字字节基准最像的replica
    - in_order策略按定义顺序逐个选择
    - first_or_random策略先选第一个，不行再随机

- 多分片SELECT查询时，本地查询和远端查询并行进行，然后合并至本地，返回。
- 对于某个场景，没分片查才能得到正确结果的情况下，可以借助Global优化。
  - GLOBAL IN或JOIN
  - 子句返回数据不宜过大

~~~mysql
# CH5节点test_query_local的数据：id && repo: [{1, 100}, {2, 100}, {3, 100}]
# CH6节点test_query_local的数据：id && repo: [{3, 200}, {4, 200}]
# 要求找到同时拥有两个仓库的用户

# 这种就很造成 N^2的查询请求
SELECT uniq(id) FROM test_query_all WHERE repo = 100
AND id IN (SELECT id FROM test_query_all WHERE repo = 200)

# 正确查询
SELECT uniq(id) FROM test_query_all WHERE repo = 100
AND id GLOBAL IN (SELECT id FROM test_query_all WHERE repo = 200)
# 上述查询的执行步骤：1.将IN⼦句单独提出，发起了⼀次分布式查询。 2.将IN子句查询的结果进行汇总，并放入一张临时的内存表保存。
# 3.将内存表发送到远端分片节点。 4.将分布式表转为本地表后，开始执行完整的SQL语句，IN子句直接使用临时内存表的数据。
~~~



## 10.管理与运维

### 10.1用户配置

- users.xml配置文件定义了用户相关的配置项。包括系统参数设定、用户定义、权限和熔断机制。
- 用户profile
- 约束：保证profile内的参数值不会被随意修改
- 自定义用户
- quota：设置用户能使用的资源配额

### 10.2 权限管理

- 访问权限：networks标签定义了网络访问权限、allow_databases和allow_dictionaries标签限制了用户数据库和字典的访问权限、
- 查询权限：决定了用户能执行的查询语句
  - 读权限：SELECT、EXISTS、SHOW、DESCRIBE
  - 写权限：INSERT、OPTIMIZE
  - 设置权限：SET
  - DDL权限：CREATE、DROP、ALTER、RENAME、ATTACH、DETACH、TRUNCATE
  - 其他权限：KILL、USE
  - readonly标签控制读、写、设置权限。0==不限制，1==读，2==读+设置
  - allow_ddl：控制DDL权限

- 数据行级权限：通过databases标签定义了用户级别的查询过滤器，实现数据的行级粒度权限。
  - 只包含符合定义条件的结果。
  - 相当于WHERE，会使PREWHERE失效。

### 10.3 熔断机制

- 当使用的资源数量达到阈值时，那么正在进行的操作会被自动中断。
- 按照时间周期累积统计：当累积量达到阈值，则直到下个计算周期开始之前，该用户将无法继续进行操作。
- 根据单次查询的用量熔断：如果某次查询使用的资源用量达到了阈值，则会被中断。以分区为最小单元进行统计。

### 10.4 数据备份

- 数据副本不能处理误删数据这类行为。
- 导出文件备份：
  - 以dump（转储）形式导出为本地文件
  - 也可以直接复制整个目录文件
- 通过快照表备份：快照表实际就是普通的数据表，按频率创建，然后用INSERT INTO SELECT语句，点对点复制。
- 按分区备份：FREEZE和FETCH。都要用ATTACH装载。
  - FREEZE：实质上是对原始目录文件进行硬链接操作，所以并不会导致额外的存储空间。
  - FETCH：只支持ReplicatedMergeTree系列，选择合适副本，并下载相应的分区数据。
  - 元数据需要单独复制/data/metadata/

### 10.5 服务监考

- system.metrics：当前正在执行的高层次概要信息
- system.events：已经执行过的高层次累积概要信息
- system.asynchronous_metrics：当前正在后台异步运行的高层次的概要信息

- 查询日志：
  - query_log：记录了CH服务中所有已经执行的查询记录
  - query_thread_log：所有线程的执行查询的信息
  - part_log：MergeTree系列的分区操作日志
  - text_log：打印日志
  - metric_log：system.metrics和system.events合并



## 100.其他知识点

### 1.布隆过滤器 BloomFilter

- 布隆过滤器实际上是一个很长的二进制向量和一系列随机映射函数。主要用于判断一个元素是否在一个集合中。
- 散列表是一个key通过一个hash函数映射到数组中的一个位置。布隆过滤器是一个key通过多个hash函数映射到位向量的多个位置。将对应位置都置1。
- 如果布隆过滤器判断元素不存在，则一定不存在。如果判断存在，则可能存在。
- 误判的原因还是哈希冲突，只是概率小了。
- 布隆过滤器不能删除元素，这会增大误判率。

- 应用场景：
  - 数据库防止穿库。使用BloomFilter来减少不存在的行或列的磁盘查找。
  - 业务场景中判断用户是否阅读过某视频或文章。
  - 用布隆过滤器当缓存的索引，只有在布隆过滤器中，才去查询缓存，如果没查询到，则穿透到db。如果不在布隆器中，则直接返回。
  - WEB拦截器，如果相同请求则拦截，防止重复被攻击。

### 2.AST abstract syntax tree 

- 抽象语法树

### 3.TTL Time To Live 

- 数据生命周期

### 4.CTE 

- Common Table Expression 公共表表达式：一个命名的临时结果集。CTE不作为对象存储，仅在查询执行期间持续。

~~~mysql
select pow(pow(2,2), 3);
# CTE
with pow(2,2) as a select pow(a, 3);
~~~

### 5.IColumn

- IColumn接口，几乎所有的操作都是不可变的：这些操作不会更改原始列，但是会创建一个新的修改后的列。
