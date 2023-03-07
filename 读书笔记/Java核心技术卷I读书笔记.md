# Java核心技术卷I读书笔记

## 一、Java程序设计概述

- 运行时自省机制：通过 Java 的反射机制，程序员可以在 Java 程序处于运行态的时候操作任意的类或者对象的属性、方法。
- 即时编译：虚拟机有一个选项，可以将执行最频繁的字节码序列转换成机器码，以弥补解释型虚拟机指令速度慢的问题。
- 可移植性：不仅指JVM，还有恒定4字节的int、Unicode编码、类库可移植接口等。
- JavaSE~(ɪ'dɪʃ(ə)n)~（标准版）、JavaEE（企业版）、JavaME（微型版）。

## 二、Java的基本程序设计结构

- 类名：字母打头的字母数字串、大写字母开头的大驼峰、文件名和公共类名相同。
- 数据类型：
  - 整型：int(4), short(2), long(8), byte(1)
    - java没有无符号整型
  - 浮点型：float(4), double(8)
    - BigDecimal
  - char类型：通常占2个字节，Unicode字符集
    - 不建议用
  - boolean类型
    - 整型和布尔不能相互转换

- var关键字是java10的特性，只用于方法的局部变量。
- 枚举类型

~~~java
enum Size{ SMALL, MEDIUM, LARGE, EXTER_LARGE };
Size s = Size.MEDIUM;
Size ss = null;
~~~

- 类型转换规则

~~~java
// 当二元运算符连接两个值x, y时，先要将两个操作数转换为同一类型，再运算
if(isDouble(x) || isDouble(y))	另一个操作数就会转换为double
else if(isFloat(x) || isFloat(y))	另一个操作数就会转换为float
else if(isLong(x) || isLong(y))	另一个操作数就会转换为long
else	两个操作数都被转换为int
~~~

- 强制类型转换

~~~java
double x = 9.997;
int nx = (int) x;	// 截断
int nxx = (int) Math.round(x);		// 舍入
~~~

- java中，>>（算术右移）用符号位填充高位；>>>（逻辑右移）用0填充高位。
- 算术左移和逻辑左移一样，都是<<。

- 字符串和非字符串拼接(+)，后者会转换为字符串。
- java中字符串时不可变的，不能修改字符串的单个字符。
- java的字符串判相等，不能用==。
- 字符串存在空串和Null串的区别：空串长度为0，内容为空；Null串表示没有关联对象。

- StringBuffer是线程安全的，StringBuilder略快。

- 标准输入流System.in

~~~java
import java.util.*;
public class InputTest{
    public static void main(String[] args) throws IOException{
        Scanner in = new Scanner(System.in);
        String name = in.nextLine();		// Yu Jixuan
        String firstName = in.next();	
        int age = in.nextInt();
        // 文件输入输出
        Scanner in = new Scanner(Path.of("myfile.txt"), StandardCharsets.UTF_8);
        PrintWriter out = new PrintWriter("myfile.txt", StandardCharsets.UTF_8);
        // 虚拟机启动目录的位置
        String dir = System.getProperty("user.dir");
    }
}
~~~

- java不能在嵌套的两个块中声明同名的变量。
- switch

~~~java
switch(choice){
    case 1:			// case的标签是常量：整型常量表达式、枚举常量、字符串字面量
        break;
    case 2:
        break;
    default:
        break;
}
~~~

- BigInteger和BigDecimal

~~~java
BigInteger a = BigInteger.valueof(100);
BigInteger reallyBig = new BigInteger("47192571905819502423423456");
BigInteger c = a.add(b);
~~~

- 数组

~~~java
int[] a = new int[100];		// int[] a 和int a[] 都是可以的; 与C++中 int* a = new int[100];基本等价
int n = 100;
int[] b = new int[n];	// 数组长度不要求是常量
int[] smallPrimes = {1, 2, 4, 7};
var arr = new int[0];	// 长度为0的数组和null并不相同
// 数组默认初始化时，所有元素都是0; boolean is False; Object is null;
int[] luckyNum = smallPrimes;	// 两个变量将引用同一个数组
luckNumber = Arrays.copyof(smallPrimes, smallPrimes.length);	// 拷贝数组
Arrays.sort(a);		// 优化后的快排
int[][] magicSquare = {{1, 2, 3},{4, 5, 6},{7, 8, 9}};
~~~

- 增强for循环适用于数组和实现了Iterable接口的类对象。

## 三、对象与类

### 3.1 类间关系

- 依赖(uses-a)：如果一个类的方法使用另一个类的对象，我们就说一个类依赖于另一个类。
  - controller依赖于serviceImpl，serviceImpl依赖于mapper，mapper依赖于opjo。
  - 应该减少依赖（耦合）。
- 聚合(has-a)：类A的对象包含类B的对象。
- 继承(is-a)：类B由类A继承而来。

### 3.2 对象变量

~~~java
// 对象变量是变量引用对象，所有的java对象都存储在堆中，一个对象包含另一个对象变量时，只是包含另一个堆对象的指针
Date birthday;	// 等同于C++ Date* birthday
~~~

### 3.3 Date和LocalDate

- java中Date类用来表示时间点，日期的日历表示法用LocalDate类

~~~java
LocalDate nw = LocalDate.now();		// 用静态工厂方法 而不是构造器
LocalDate newYearsEve = LocalDate.of(1999, 12, 31);
int year = newYearsEve.getYear();
int month = newYearsEve.getMonthValue();
int day = newYearsEve.getDayofMonth();
~~~

### 3.4 java静态工厂方法

~~~java
// 优势1：有名字，可读性强
String value1 = String.valueOf(0);
String value3 = String.valueOf(true);
String value3 = String.valueOf("Henry");
Optional<String> value1 = Optional.empty();
Optional<String> value2 = Optional.of("Henry");
Optional<String> value3 = Optional.ofNullable(null);
// 优势2：不必每次都创建一个新实例
// 如：Boolean 类的 valueOf() 类就是提前预先构建好的实例，或将构建好的实例缓存起来，进行重复的利用，从而避免创建不必要的重复对象。
public final class Boolean implements java.io.Serializable, Comparable<Boolean>
{
    /**
     * The {@code Boolean} object corresponding to the primitive value {@code true}.
     */
    public static final Boolean TRUE = new Boolean(true);

    /**
     * The {@code Boolean} object corresponding to the primitive value {@code false}.
     */
    public static final Boolean FALSE = new Boolean(false);
    //
    public static Boolean valueOf(String s) {
        return toBoolean(s) ? TRUE : FALSE;
	}
}
// 优势3：可以返回原返回类型的任何子类对象 里氏替换原则
public Class Animal {
    public static Animal getInstance(){
       // 可以返回 Dog or Cat
        return new Animal();
    }
}
Class Dog extends Animal{}
Class Cat extends Animal{}
// 优势4：所返回的对象的类可以随着每次调用而发生变化，这取决于静态工厂方法的参数值
~~~

### 3.5 类的构造

- 可以在一个源文件中包含两个类，以public标识的类为主力类。但一个类一个文件或许比较好。

- 如果要接受一个对象引用作为构造参数，需要考虑null问题。

~~~java
public Employee(String n, double s){
    name = Objects.requireNonNullElse(n, "unknown");	// 如果为空，则赋值"unknown"
    salary = s;
}
public Employee(String n, double s){
    name = Objects.requireNonNull(n, "The name cannot be null");	// 如果为空，产生NullPointerException异常
    salary = s;
}
~~~

- Getter不应该返回可变对象的引用，这会使封装被破坏，应该先克隆。

~~~java
class Employee{
    private Date hireDay;
    public Date getHireDay(){
        return hireDay;		 				//BAD!!!
        //return (Date)hireDay.clone();		//OK!!!
    }
}
public static void main(String[] args){
    Employee harry = ...;
    Date d = harry.getHireDay();
    d.setTime(d.getTime() - (long)1000);	// d和hireDay引用同一个对象，导致私有的实例字段被更改
}
~~~

- final修饰表示常量，一旦创建不可改变且声明时必须赋值，或在类的构造方法中赋值，不可以重新赋值。

- 每个类都可以拥有一个静态main方法。在程序启动时，静态main方法将执行并构造程序所需要的对象。
- java **总是** call by *value* 的，就是浅拷贝(C++)。

### 3.6 java实现swap函数

~~~java
// 只能运用在对象只有两个类变量时，交换这两个确定的变量
public class TestSwap {
	private int a = 3;
	private int b = 5;
	private void swap() {
		int temp = this.a;
		this.a = this.b;
		this.b = temp;
	}
}
// 通过数组
public class TestSwap {
	public static void main(String[] args){
		int a = 3;
		int b = 5;
		int[] arr = swap(a,b);
		a = arr[0];
		b = arr[1];
	}
	private static int[] swap(int x, int y){
		return new int[]{y,x};
	}
}
// 自定义包装类对象
class MyInteger {   
	private int x;   
	public MyInteger(int xIn) { x = xIn; }
	public int getValue() { return x; }
	public void setValue(int xIn) { x = xIn;}
}
public class Swapping {   
	static void swap(MyInteger rWrap, MyInteger sWrap) {         
		int t = rWrap.getValue();      
		rWrap.setValue(sWrap.getValue());      
		sWrap.setValue(t);   
	}   
	public static void main(String[] args) {      
		int a = 23, b = 47; 
		MyInteger aWrap = new MyInteger(a);      
		MyInteger bWrap = new MyInteger(b);      
		swap(aWrap, bWrap);      
		a = aWrap.getValue();      
		b = bWrap.getValue(); 
	}
}
// 通过反射，待补充

~~~

### 3.7 对象构造

- **重载**(overloading)：多个方法有相同的函数名，不同的参数列表。
  - 可以有不同的返回类型；可以有不同的访问修饰符；可以抛出不同的异常。
  - 调用的时候根据函数的参数来区别不同的函数。
  - 不能有两个签名相同（名字和参数相同），但返回值不同的方法。
- **重写**(override)：子类中定义的方法与其父类的有相同的名称、参数和返回值，不同的函数体。
  - 子类方法可见性不能低于超类；子类重写方法一定不能抛出新的检查异常或者比被父类方法申明更加宽泛的检查型异常。
- 字段可以默认初始化；但方法的局部变量不能，必须明确初始化。
- 如果类没有编写构造器，编译器会提供一个无参构造器，对所有实例字段做默认初始化。
- java可以显示初始化字段：

~~~java
class Employee{
    private static int nextId = 0;
    private int id = assignId();
    private static int assignId(){
        int r = nextId;
        nextId++;
        return r;
    }
}
~~~

~~~java
// 一个骚操作，类似于C++里的委托构造函数
public Employee(double s){
    this("Employee #" + nextId, s);		//调用另一个构造函数 只能写在第一句
    nextId++;
}
~~~

- java通过包将类组织在一个集合中，并通过保证包名的唯一性，来区分代码。
  - 比如：因特网域名的逆序作为包名称。工程名做子包名。

- 从编译器角度来看，嵌套的包间没有任何关系，每个包都是独立的类集合。

- 静态导入：导入静态方法和静态字段，而不只是类。

~~~java
import static java.lang.Math.*;
import static java.lang.Math.pow;		//导入静态方法，且不冲突
// 不建议用
public class KaiTest {
    public static void main(String[] args) {
        int x = 3;
        int y = 4;
        System.out.println(sqrt(pow(x, 2) + pow(y, 2)));
    }
}
~~~

### 3.8 类设计技巧

- 一定要保证数据私有
- 一定要对数据进行初始化
- 不要在类中使用过多的基本类型
  - 用其他类替换多个相关的基本类型
- 不是所有字段都需要getter或setter
- 分解过多职责的类
- 名称要有意义
- 优先使用不可变的类

## 四、继承

### 4.1 继承

- 只有Employee方法能直接访问Employee类的私有字段。

~~~java
// 一个比较标准的equals实现
@Override
public boolean equals(Object obj){
    if(this == obj) return true;        
    if(obj == null) return false;
    if(getClass() != obj.getClass())    return false;
    Employee other = (Employee) obj;
    // Object类的equals将确定两个对象引用是否相等。Objects.equals(null, null) == true;
    return Objects.equals(name, other.name) && salary == other.salary && id == other.id;
}
// 访问父类私有字段
public class Manager extends Employee{
    public double getSalary(){
        return super.getSalary() + bonus;
    }
}
~~~

- super调用构造器的语句必须是子类构造器的第一条语句。
- 子类构造器没有显式地调用超类的构造器，将自动调用超类的无参构造器。这时，如果超类没有无参构造器，编译器会报错。
  - 所以是不是都该写一个无参？
- this的两个作用：指示隐式参数的引用；调用本类的其他构造器。
- super的两个作用：调用超类的方法；调用超类的构造器。
- 调用构造器的语句只能作为另一个构造器的第一条语句出现。

- **多态(polymorphism)**：一个对象变量能指示多种实际类型的现象称为多态。
- **动态绑定**：在运行时能自动选择适合的方法。

~~~java
public static void main(String[] args) {
    Employee[] staff  = new Employee[10];
    Manager manager = new Manager("Kai", 1000,100);
    staff[0] = manager;
    staff[0].setBonus(1000);		// ERROR,因为staff[0]是Employee类型的
    manager.setBonus(1000);
    staff[0].getSalary();			// 1100; 调用的是manager的getSalary
}
~~~

- 方法调用的过程：
  - 编译器查看对象的声明类型和方法名，列举该类和超类所有名字匹配且可访问的候选方法。
  - 重载解析：根据参数类型，匹配方法。
  - 静态绑定：如果是private、static、final方法或构造器，那么编译器就能确定方法。
  - 如果方法依赖于隐式参数(this)的实际类型，就需要运行时动态绑定。
- 虚拟机会为每个类维护一个方法表，列出所有 方法签名&实际方法。
- 重写时，子类方法可见性不能低于超类，因为编译器在匹配方法时，会列举超类的方法表，也就是说需要父类方法可访问。
- final类不允许继承，final方法不允许重写，final类中的所有方法自动成为final方法。

- 超类强制转换为子类时，应该使用instanceof进行检查。
  - 最好尽量少用类的强制类型转换和instanceof。

~~~java
staff[1] = new Employee("Kai2", 10000);
if(staff[1] instanceof Manager) {		//false
    Manager man = (Manager) staff[1];
    man.setBonus(1000);
    System.out.println(man.getSalary());
}
~~~

- 使用abstract关键字，可以避免实现缺乏内容的函数。**包含一个或多个抽象方法的类必须被声明为抽象的**。
  - 抽象类可以包含字段和具体方法。
- 抽象类不能实例化。
- 如果希望子类的方法访问超类的某个字段或方法，需要将这些类方法和字段声明为**受保护的**(protected)。
- 访问修饰符及权限：

| 访问修饰符              | 可见性               |
| ----------------------- | -------------------- |
| public                  | 外部完全可见         |
| protected               | 对本包和所有子类可见 |
| default（不需要修饰符） | 对本包可见           |
| private                 | 仅对本类可见         |

- java中只有基本类型不是对象，数组是对象。

- 子类equals方法应该首先调用超类的equals。如果超类字段相等，才需要比较子类的字段。

~~~java
@Override
public boolean equals(Object obj){
    if(!super.equals(obj))  return false;
    Manager other = (Manager) obj;
    return bonus == other.bonus;
}
~~~

- Object类默认的hashCode方法会从对象的存储地址得出散列码，而字符串的散列码由内容导出。

~~~java
// 一个好的hashCode实践
public int hashCode(){
    return Objects.hash(id, name, salary);
}
~~~

- java允许运行时确定数组大小。

- @SuppressWarnings 抑制警告注解，取消显示指定的编译器警告

- 包装器是不可变的，一旦构造，就不允许修改包装在其中的值。
- 可变长度参数

~~~java
public static double max(doubel... values){
	double largest = Double.NEGATIVE_INFINITY;
    for(double v : values)	if(v > largest)	largest = v;
    return largest;
}
~~~

- 枚举类

~~~java
public class EnumTest {
    public static void main(String[] args) {
        System.out.println(Size.SMALL.toString());      // SMALL
        Size s = Enum.valueOf(Size.class, "LARGE");
        System.out.println("size = " + s);      // size = LARGE
        Size[] values = Size.values();  // 包含Size.SMALL... 等元素
        System.out.println(Size.MEDIUM.ordinal());      // 1
    }
}
// 枚举类型 所有枚举类型都是Enum类的子类
enum Size {
    // 4个实例
    SMALL("S"), MEDIUM("M"), LARGE("L"), EXTRA_LARGE("XL");
    private String abbreviation;        // 缩写
    // 枚举类型的构造器总是私有的
    private Size(String abbreviation){
        this.abbreviation = abbreviation;
    }
    public String getAbbreviation(){
        return abbreviation;
    }
}
~~~

### 4.2 反射

- **反射**：能够分析类能力的程序。
  - 在运行时分析类的能力。
  - 在运行时检查对象（例如，编写一个适用于所有类的toString方法）。
  - 实现泛型数组操作代码。
  - 利用Method对象（类似于C++函数指针）。
- 获得class类

~~~java
public static void main(String[] args) throws ClassNotFoundException, NoSuchMethodException, 
										InvocationTargetException, InstantiationException, 
										IllegalAccessException {
    Employee e = new Employee("Harry", 10000);
    Employee m = new Manager("Hacker", 1000, 100);
    // 得到Class的第一种方式
    Class ecls = e.getClass();
    Class mcls = m.getClass();
    System.out.println(ecls.getName() + " " + e.getName()); // extendsLearn.Employee Harry
    System.out.println(mcls.getName() + " " + m.getName()); // extendsLearn.Manager Hacker
    // 得到Class的第二种方式，如果clsName会在运行时变化，就可以用这个方法动态获得类名
    String clsName = "extendsLearn.Manager";
    Class ncls = Class.forName(clsName);
    System.out.println(ncls.toString());    // class extendsLearn.Manager
    // 得到Class的第三种方式
    Class cls1 = Random.class;
    Class cls2 = int.class;
    Class cls3 = Double[].class;		// [Ljava.lang.Double;
    // 虚拟机为每个类型维护一个唯一的Class对象，因此 == 就能判定Class的相等性
    if(clsName == mcls)		System.out.println("Yes");		// Manager.class != Employee.class
    Object obj = cls1.getConstructor().newInstance();   // 获取cls1类型类的实例 无参构造
}
~~~

- 分析类的能力
  - java.lang.reflect包有三个类Field、Method、Constructor分别用于描述类的字段、方法和构造器。
  - getModifiers返回整数，用不同的0/1位描述使用的修饰符：public、static、final
  - Modifier类的isPublic、isFinal方法可以判断修饰符
  - getFields、getMethods、getConstructors返回类的 ***公共*** 字段、方法和构造器数组，包括超类的公共成员。
  - getDeclareFields、getDeclareMethods、getDeclareConstructors返回类的 ***所有*** 字段、方法和构造器。不包括超类的成员。
  - 如下代码可以查看java解释器能加载的任何类，不仅是编译时可以用的类。


~~~java
package extendsLearn;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Scanner;

@SuppressWarnings("all")
public class StructureReflection {
    public static void main(String[] args) throws ClassNotFoundException {
        String name;
        if(args.length > 0) name = args[0];
        else{
            Scanner in = new Scanner(System.in);
            System.out.println("Enter class name(e.g. java.util.Date): ");
            name = in.next();
        }
        Class cl = Class.forName(name);			// 从完整类名获得Class
        Class supercl = cl.getSuperclass();
        String modifiers = Modifier.toString(cl.getModifiers());	// 00000000000000000000000000010001 public final
        if(modifiers.length() > 0) System.out.print(modifiers + " ");
        System.out.print("class" + name);
        if(supercl != null && supercl != Object.class)
            System.out.print(" extends " + supercl.getName());
        System.out.print("\n{\n");
        printConstructors(cl);
        System.out.println();
        printMethods(cl);
        System.out.println();
        printFields(cl);
        System.out.println("}");
    }
    public static void printConstructors(Class cl){
        Constructor[] constructors = cl.getDeclaredConstructors();		// 所有构造器
        for(Constructor c : constructors){
            String name = c.getName();
            System.out.print("   ");
            String modifiers = Modifier.toString(c.getModifiers());
            if(modifiers.length() > 0)  System.out.print(modifiers + " ");
            System.out.print(name + "(");
            Class[] paramTypes = c.getParameterTypes();					// 所有参数Class
            for(int j = 0; j < paramTypes.length; j++){
                if(j > 0)   System.out.print(", ");
                System.out.print(paramTypes[j].getName());
            }
            System.out.println(");");
        }
    }
    public static void printMethods(Class cl){
        Method[] methods = cl.getDeclaredMethods();
        for(Method method : methods){
            Class returnType = method.getReturnType();
            String name = method.getName();
            System.out.print("   ");
            String modifiers = Modifier.toString(method.getModifiers());
            if(modifiers.length() > 0) System.out.print(modifiers + " ");
            System.out.print(name + "(");
            Class[] paramTypes = method.getParameterTypes();
            for(int j = 0; j < paramTypes.length; j++){
                if(j > 0)   System.out.print(", ");
                System.out.print(paramTypes[j].getName());
            }
            System.out.println(");");
        }
    }
    public static void printFields(Class cl){
        for (Field field : cl.getDeclaredFields()) {
            Class type = field.getType();
            String name = field.getName();
            System.out.print("   ");
            String modifiers = Modifier.toString(field.getModifiers());
            if(modifiers.length() > 0) System.out.print(modifiers + " ");
            System.out.println(type.getName() + " " + name + ";");
        }
    }
}
~~~

~~~shell
Enter class name(e.g. java.util.Date): 
java.lang.Double
public final classjava.lang.Double extends java.lang.Number
{
   public java.lang.Double(double);
   public java.lang.Double(java.lang.String);

   public equals(java.lang.Object);
   public static toString(double);
   public toString();
   public hashCode();
   public static hashCode(double);
   public static min(double, double);
   public static max(double, double);
   public static native doubleToRawLongBits(double);
   public static doubleToLongBits(double);
   public static native longBitsToDouble(long);
   public volatile compareTo(java.lang.Object);
   public compareTo(java.lang.Double);
   public byteValue();
   public shortValue();
   public intValue();
   public longValue();
   public floatValue();
   public doubleValue();
   public static valueOf(java.lang.String);
   public static valueOf(double);
   public static toHexString(double);
   public static compare(double, double);
   public static isNaN(double);
   public isNaN();
   public static isFinite(double);
   public static isInfinite(double);
   public isInfinite();
   public static sum(double, double);
   public static parseDouble(java.lang.String);

   public static final double POSITIVE_INFINITY;
   public static final double NEGATIVE_INFINITY;
   public static final double NaN;
   public static final double MAX_VALUE;
   public static final double MIN_NORMAL;
   public static final double MIN_VALUE;
   public static final int MAX_EXPONENT;
   public static final int MIN_EXPONENT;
   public static final int SIZE;
   public static final int BYTES;
   public static final java.lang.Class TYPE;
   private final double value;
   private static final long serialVersionUID;
}
~~~

- 在运行时分析对象
  - Field类对象可以get和set被反射类对象的值，前提是用setAccessible方法覆盖java的访问控制。

~~~java
public static void main(String[] args) throws NoSuchFieldException, IllegalAccessException {
    Employee harry = new Employee("Kaikai", 50000);
    Class< ? extends Employee> cls = harry.getClass();
    Field f = cls.getDeclaredField("name");
    f.setAccessible(true);
    Object o = f.get(harry);
    System.out.println(o);
    f.set(harry, "kait");
    System.out.println(f.get(harry));
}
~~~

~~~java
public class ObjectAnalyzer {
   private ArrayList<Object> visited = new ArrayList<>();

   public String toString(Object obj) throws IllegalAccessException {
       if(obj == null)  return "null";
       if(visited.contains(obj))    return "...";
       visited.add(obj);
       Class cls = obj.getClass();
       if(cls == String.class)      return (String)obj;
       if(cls.isArray()){
           StringBuilder r = new StringBuilder(cls.getComponentType() + "[]{");       // 数组元素Class
           for(int i = 0;i < Array.getLength(obj); i++){    // Array是reflect库的类
               if(i > 0) r.append(",");
               Object val = Array.get(obj, i);
               if(cls.getComponentType().isPrimitive()) r.append(val);       // isPrimitive 判定是否为基本数据类型
               else r.append(toString(val));
           }
           return r.append("}").toString();
       }
       String r = cls.getName();
       while(cls != null){
           r += "[";
           Field[] fields = cls.getDeclaredFields();
           AccessibleObject.setAccessible(fields, true);
           for (Field f : fields) {
               if(!Modifier.isStatic(f.getModifiers())){
                   if(!r.endsWith("[")) r += ",";
                   r += f.getName() + "=";
                   Class t = f.getType();
                   Object val = f.get(obj);
                   if(t.isPrimitive())  r += val;
                   else r += toString(val);
               }
           }
            r += "]";
           cls = cls.getSuperclass();
       }
       return r;
   }
}
public class ObjectAnalyzerTest {
    public static void main(String[] args)throws ReflectiveOperationException{
        ArrayList<Integer> squares = new ArrayList<>();
        for(int i = 1; i <= 5; i++){
            squares.add(i*i);
        }
        System.out.println(new ObjectAnalyzer().toString(squares));
    }
}
/** java.util.ArrayList[elementData=class java.lang.Object[]{
		java.lang.Integer[value=1][][],java.lang.Integer[value=4][][],java.lang.Integer[value=9][][],
		java.lang.Integer[value=16][][],java.lang.Integer[value=25][][],null,null,null,null,null
	},size=5]
	[modCount=5][][]
*/
~~~

- 编写泛型数组代码
  - java数组会记住每个元素的类型（new表达式使用的元素类型）。虽然可以将Employee[]临时转化为Object[]，再转换回来。但，如果一开始就是Object[]，却不能转换成Employee[]。所以泛型数组的获取，必须获得正确的元素类型。

~~~java
//这里参数和返回值用Object而不是Object[]的原因是，int[]可以转换为Object，不能转换为Object[]
public static Object goodCopyof(Object a, int newLength){	
    Class cl = a.getClass();
    if(!cl.isArray())	return null;
    Class componentType = cl.getComponentType();
    int length = Array.getLength(a);
    Object newArray = Array.newInstance(componentType, newLength);
    System.arraycopy(a, 0, newArray, 0, Math.min(length, newLength));
    return newArray;
}
~~~

~~~java
// System.arraycopy
// 当数组为一维数组，且元素为基本类型时，属于深复制，即原数组与新数组的元素不会相互影响
// 当数组为多维数组，或一维数组中的元素为引用类型时，属于浅复制，原数组与新数组的元素引用指向同一个对象
public static void main(String[] args) {
    Employee[] staff  = new Employee[2];
    Manager manager = new Manager("Kai", 1000,100);
    staff[0] = manager;
    staff[1] = new Employee("Kai2", 10000);
    Employee[] copy = new Employee[4];
    // 一维数组中的元素为引用类型，浅拷贝
    System.arraycopy(staff, 0, copy, 0, 2);
    // 返回对象的哈希值，在该对象的类没有重写了hashCode()方法的情况下，与hashCode()方法返回的哈希值相同。
    // 另外，null的哈希值为零。
    // 在hashCode()方法被重写的情况下，就不能根据hashCode()方法来判断两个变量是否引用了同一个对象，
    // 但可以根据System.identityHashCode(Object x)方法来判断两个变量是否引用了同一个对象。
    System.out.println(System.identityHashCode(staff[0]) + " " + System.identityHashCode(copy[0]));
    copy[0].setName("TTT");
    System.out.println(staff[0].getName());			// TTT
    // 一维String 浅拷贝
    String[] strs = {"abc", "def", "hij"};
    String[] strscpy = new String[4];
    System.arraycopy(strs, 0, strscpy, 0, 3 );
    System.out.println(strs[0] == strscpy[0]);		// true
    strscpy[1] = "zzz";		// 因为String不可变，所以重新赋值是更改引用，不会影响原对象变量
    System.out.println(strs[1] + " " + strscpy[1]);		// zzz def
}
~~~

- 调用任意方法和构造器
  - java没有提供途径将一个方法的存储地址传给另一个方法。接口和lambda表达式是一种更好的解决方案。
  - 反射机制允许调用任意的方法。
  - 不要使用回调函数的Method对象！

~~~java
// Object invoke(Object obj, Object... args)
// 第一个参数是隐式参数，对于静态方法是null，其余对象提供了显示参数
// 如果返回类型是基本类型， invoke方法会返回其包装器类型
// Method getMethod(String name, Class... parameterTypes)
// Constructor getConstructor (Class ...paramType);
public class MethodTest {
    public static void main(String[] args) throws ReflectiveOperationException{
        Class math = Class.forName("java.lang.Math");
        Method sqrt = math.getMethod("sqrt", double.class);
        Method square = MethodTest.class.getMethod("square", double.class);
        printTable(1, 10, 10, square);
        printTable(1, 10, 10, sqrt);
    }
    public static double square(double x){
        return x * x;
    }
    public static void printTable(double from, double to, int n, Method f) throws ReflectiveOperationException{
        System.out.println(f);
        double dx = (to - from) / (n - 1);
        for(double x = from; x <= to; x += dx){
            double y = (Double)f.invoke(null, x);
            System.out.printf("%10.4f | %10.4f%n", x, y);		// %n跨平台行分隔符 \n
        }
    }
}
~~~

> 应用场景

- Spring框架的IoC通过反射来创建对象、设置依赖属性
- JDBC加载数据库驱动时，也会用到反射
- 泛型

### 4.3 继承的设计技巧

- 将公共操作和字段放在超类中
- 不要使用受保护的字段
  - java里同一个包的类都能访问protected字段。
  - protected方法指示子类该重新定义该方法还是不错的。
- 使用继承实现"is-a"关系
  - 不是is-a，就不要用继承
- 除非所有继承的方法都有意义，否则不要使用继承
- 在覆盖方法时，不要改变预期的行为
- 使用多态，而不是类型信息
- 不要滥用反射

## 五、接口、lambda表达式和内部类

### 5.1 接口

- 接口用来描述类应该做什么。 

- 接口中所有方法自动都是public方法，不必提供关键字。
- 接口没有实例。不会有实例字段，java8之前没有具体方法。

~~~java
public class Employee implements Comparable<Employee>{		// 高层提供compareTo 编译器容易检查到
    @Override
    public int compareTo(Employee o) {
        return Double.compare(this.salary, o.salary);		// double比较的good方法
    }
}
~~~

- 对于继承，如果子类和超类的比较具有不同的含义，就应该将不同类间的对象比较视作非法的。反之，可以在超类提供一个final的compareTo方法。（类似于equals）。

~~~java
Comparable X = new Employee("Kai", 10000);	//	接口变量必须引用实现了这个接口的类对象
if(anObject instanceof Comparable)	{}		// 检查对象是否实现了特定的接口
// 接口可以被扩展
public interface Moveable{
    void move(double x, double y);
}
public interface Powered extends Moveable{
    double ilesPerGallon();
    double SPEED_LIMIT = 95;		// 接口可以包含常量，默认是 public static final 的
}
~~~

- 每个类只能有一个直接超类，但可以实现多个接口。
- java8允许在接口中增加静态方法，但目前为止，通常的做法是将静态方法放在伴随类中。如：Collection/Collections 和 Path/Paths。

~~~java
Paths.get("jdk-11", "conf", "security");
// 等价于 接口静态方法	自定义接口时，可以采用下述方法
public interface Path{
    public static Path of(URI uri){ ... }
    public static Path of(String first, String... more){ ... }
}
/** URL除了标识一个资源，还会为资源提供一个特定的网络位置，客户端可以用来获取这个资源的一个表示。
	而URI，只能告诉你一个资源是什么，但是无法告诉你它在那里，以及如何得到这个资源。
*/
~~~

- java9中，接口方法可以是private的，private方法可以是静态方法或实例方法。用作接口其他方法的辅助方法。

- 题外：
  - 非静态方法既可以访问静态数据成员又可以访问非静态数据成员，而静态方法只能访问静态数据成员；
  - 非静态方法既可以访问静态方法又可以访问非静态方法，而静态方法只能访问静态数据方法。
    - 原因：静态方法和数据会随着类的定义而被载入内存，而非静态的代码段此时可能并不存在。
  - 静态方法必须被实现。
- 可以为接口方法提供一个默认实现，且必须用 ***default*** 标识。
  - 至此，接口方法可以是公有抽象的、静态的、私有实例的、默认的。

~~~java
public interface Comparable<T>{
    default int compareTo(T other){	return 0;}
}
~~~

- 解决 **默认方法** 冲突：如果一个接口将一个方法定义为默认方法，然后又在超类或另一个接口中定义同样的方法，如果解决冲突？
  - 超类优先：如果超类提供了一个签名相同且具体的方法，则默认方法会被忽略。
    - 根据类优先规则，默认方法重新定义Object类的某个方法是无意义，且容易造成失误的行为。
  - 接口冲突：如果一个接口提供了一个默认方法，另一个接口提供了签名相同的方法（不管是否默认），必须重写这个方法。
    - 如果两个接口都没有提供默认实现，是没有冲突的也是不会冲突的。

- 接口与回调

~~~java
public class MainTest {
    public static void main(String[] args) {
        TimePrinter listener = new TimePrinter();
        // timer可以每隔确定的时间段，调用方法来实现动作。但调用类的对象会更加灵活。
        // timer调用的对象需要实现ActionListener接口，并在actionPerformed方法中定义动作
        Timer timer = new Timer(1000, listener);
        timer.start();
        JOptionPane.showMessageDialog(null, "Quit program");
        System.exit(0);     // 程序退出码
    }
}
class TimePrinter implements ActionListener{
    @Override
    public void actionPerformed(ActionEvent e) {
        System.out.println("At the tone, the time is " + Instant.ofEpochMilli(e.getWhen()));
        Toolkit.getDefaultToolkit().beep();
    }
}
~~~

- 如果想对String做比较，且比较规则是按字符串长度排序，那就需要新建一个比较器。

~~~java
String[] friends = {"Peter", "Paul", "Mary"};
Arrays.sort(friends, new LengthComparator());
//
class LengthComparator implements Comparator<String>{
    public int compare(String first, String second){
        return first.length() - second.length();
    }
}
~~~

- cloneable接口是用来做深拷贝的，Object默认的克隆操作是浅拷贝的。

- Cloneable接口时标记接口，不包含任何方法，clone方法是从Object继承来的。要使用clone函数需要：
  - 实现Cloneable接口
  - 重新定义clone方法并指定public访问修饰符

- clone实现深浅拷贝的区别在于函数体。

~~~java
class Employee implements Cloneable{
    public Employee clone() throws CloneNotSupportedException{
        Employee cloned = (Employee)super.clone();		// Object的clone
        cloned.hireDay = (Date) hireDay.clone();		// 然后把可变实例字段单独clone一下
        return cloned;
    }
}
~~~

- 因为不能确保子类的实现者一定会修正clone，所以，Object的clone方法被声明为了protected。

### 5.2 Lambda表达式

- lambda表达式 是一个可传递的代码块，可以在以后执行一次或多次。

- 三要素：括号 箭头 表达式
- 无须指定lambda表达式的返回类型，总是从上下文推导可得出。

~~~java
// 括号 箭头 表达式
Arrays.sort(friends, (String first, String second) -> first.length() - second.length());
// 如果一个表达式写不完
Arrays.sort(friends, (String first, String second) -> {
    if(first.length() < second.length())	return -1;
    else if(first.length() > second.length())	return 1;
    return 0;
});
// 即使没有参数,也要提供空括号
() -> { for(int i = 100; i >= 0; i--)	System.out.println(i); }
// 如果参数类型可推断，参数类型可省略
Comparator<String> comp = (f, s) -> f.length() - s.length() ;
// 如果只有一个参数，且类型可推断，小括号可省略
ActionListener listener = event -> System.out.println("The time is " + Instant.ofEpochMilli(event.getWhen()));
~~~

- 函数式接口：只有一个抽象方法的接口。需要这种接口的对象时，可以提供一个lambda表达式。

~~~java
// java.util.function
public interface Predicate<T>{	// 谓词 断言为
    boolean test(T t);
}
public interface Supplier<T>{	// 供应者
    T get();
}
arrayList.removeIf(e -> e == null);
// 对应的方法引用 
arrayList.removeIf(Objects::isNull);
// requireNonNullElseGet java9
LocalDate day = LocalDate.now();
// 懒计算，只有day为null时，才调用供应者
LocalDate hireDay = Objects.requireNonNullElseGet(day, () -> LocalDate.of(1970, 1, 1));
~~~

- 方法引用：指示编译器生成一个函数式接口的实例，覆盖这个接口的抽象方法来调用给定的方法。
  - 只有当lambda表达式的方法体只调用一个方法而不做其他操作时，才能被重写为方法引用。
  - 方法引用不能独立存在，总是会转换为函数式接口的实例。


~~~java
Timer timer = new Timer(1000, e -> System.out.println(e));
// 生成ActionListener对象，actionPerformed方法调用System.out.println(e);
// System.out是PrintStream类的一个实例，有10个重载的println方法。 根据(ActionEvent e),确定了(Object x)这个签名。
Timer timer = new Timer(1000, System.out::println);
//
Arrays.sort(strings, String::compareToIgnoreCase);
~~~

~~~java
// 主要有三种情况
// 等价于向方法传递参数的lambda表达式 System.out::println == x -> System.out.println(x)
object::instanceMethod 
// 第一个参数会成为方法的隐式参数 String::compareToIgnoreCase == (x, y) -> x.compareToIgnoreCase(y)
Class::instanceMethod
// 所有参数都传递到静态方法 Math::pow == (x, y) -> Math.pow(x, y)
Class::staticMethod
// this和super关键字在方法引用中都是被允许的
this::instanceMethod
super::instanceMethod
~~~

| 方法引用          | 等价lambda表达式                       | 说明                                                 |
| ----------------- | -------------------------------------- | ---------------------------------------------------- |
| separator::equals | x -> separator.equals(x)  ~（分隔符）~ | 包含一个对象和一个实例方法的方法表达式，显式参数传入 |
| String::trim      | x -> x.trim()                          | 包含一个类和一个实例方法的方法表达式，隐式参数       |
| String::concat    | (x, y) -> x.concat(y)                  | 类+方法，一个隐式参数，其余的是显式参数              |
| Integer::valueof  | x -> Intege::valueof(x)                | 静态方法，显示参数                                   |
| Integer::sum      | (x, y) -> Integer::sum(x, y)           |                                                      |
| Integer::new      | x -> new Integer(x)                    | 构造器引用，参数传递到构造器                         |
| Integer[]::new    | n -> new Integer[n]                    | 数组构造器引用，lambda参数是数组长度                 |

- **再来一次**：**当需要 只有一个抽象方法的接口 的对象时，可以提供一个lambda表达式**。当lambda表达式的方法体只调用一个方法而不做其他操作时，可以提供一个方法引用。无论是lambda还是方法引用，总是会转换为函数式接口的实例。

- 构造器引用的方法名是new

~~~java
ArrayList<String> names = new ArrayList<>();	// Person(String) 构造器
Stream<Person> stream = names.stream().map(Person::new);
List<Person> people = stream.collect(Collectors.toList());
// java无法构造泛型数组， 所以：Object[] people = stream.toArray();	 但如果要T[]，就需要用到构造器引用
Person[] people = stream.toArray(Person[]::new);
~~~

- 如果希望lambda表达式能访问外围方法或类的变量，就要求该变量是事实最终变量。
  - 事实最终变量：变量初始化后将不会被重新赋值。

~~~java
public static void repeatMessage(String next, int delay){
    ActionListener listener = event -> {
       	System.out.println(text);
        Toolkit.getDefaultToolkit().beep();
    };
    new Timer(delay, listener).start();
}
// lambda表达式可能在repeatMessage调用返回很久之后才开始运行，而此时text变量已经不存在了。那如何保留该值呢？
// lambda表达式的3个部分 参数、代码块、自由变量的值（非参数且不在代码中定义的变量）
// 自由变量的值需要被lambda捕获，并存储在lambda的数据结构中
// 这就要求只能引用值不会改变的变量，无论是lambda改变该值，还是外部改变该值都是不合法的。
// 因为text是String的，所以捕获它是合法的。
~~~

- lambda表达式中的this关键字指的是创建lambda表达式的方法的this，不是函数式接口对象的方法。

~~~java
public class Applicatoin{
    public void init(){
        // this指的是Application的对象，不是ActionListener的对象
        ActionListener listener = e -> System.out.println(this.toString())
    }
}
~~~

- lambda表达式的重点在于延迟执行。（其实不是很懂这句话）
  - 多次运行代码，在单独线程中运行代码，在算法的适当位置运行代码，发生某种情况时执行代码，只在必要时运行代码。
- 有很多专用的函数式接口可以用来简化表达
  - Runnable、Supplier<T>、Consumer<T>、Predicate<T>等。

- 这些通用接口还有基本类型int、long、double的特殊化接口，效率更高
  - BooleanSupplier、*P*Function<T>、*P*Predicate、*P*Consumer等。(P for Int、Long、Double)


~~~java
public interface Runnable {
    void run();
}
// 
public static void repeat(int n, Runnable action){
    for(int i = 0; i < n; i++)	action.run();
}
repeat(10, () -> System.out.println("Hello, World!"));
//
public interface IntConsumer{
    void accept(int value);
}
public static void repeat(int n, IntConsumer action){
    for(int i = 0; i < n; i++)	action.accept(i);
}
repeat(10, i -> System.out.println("Consumer: " + (9 - i)));
~~~

- 大部分标准函数式接口都提供了非抽象方法来生成或合并函数。

~~~java
Predicate.isEqual(a) == a::equals		// a为null也能正常工作
// and、or、negate 用来合并谓词
Predicate.isEqual(a).or(Predicate.isEqual(b) == x -> a.equals(x) || b.equals(y)
~~~

- 自定义的函数式接口可以加上@FunctionalInterface注解。
- Comparator接口包含很多方便的静态方法来创建比较器，如：静态comparing方法取一个“键提取器”函数，它将类型T映射为一个可比较的类型。
- 还可以为提取的键指定比较器。

~~~java
Arrays.sort(people, Comparator.comparing(Person::getName).thenComparing(Person::getAge));
// 自定义比较器
Arrays.sort(people, Comparator.comparing(Person::getName), (s, t) -> Integer.compare(s.length(), t.length()));
// 避免int、long、double的装箱
Arrays.sort(people, Comparator.comparingInt(p -> p.getName().length()));
~~~

- nullsFirst和nullsLast适配器用来处理键函数返回null的情况。

### 5.3 内部类

- 内部类：定义在类中的另一个类。
  - 内部类可以对同一个包中的其他类隐藏
  - 内部类可以访问定义在外部类作用域中的数据，包括原本私有的数据。
- 内部类的对象有一个隐式引用，指向实例化这个对象的外部类的对象。通过该指针，可以访问外部类对象的全部状态。
  - 如：Iterator类不需要一个显式指针指向它的LinkedList。
- 静态内部类没有这个指针。

~~~java
public class TalkingClock {
    private int interval;
    private boolean beep;

    public TalkingClock(int interval, boolean beep){
        this.interval = interval;
        this.beep = beep;
    }

    public void start(){
         ActionListener listener = new TimePrinter();
         // ActionListener listener = this.new TimePrinter();
        Timer timer = new Timer(1000, listener);
        timer.start();
    }

    public class TimePrinter implements ActionListener{
        // automatically generated code
        /** public TimePrinter(TalkingClock clock){
            outer = clock;
        }*/

        public void actionPerformed(ActionEvent e){
            System.out.println(Instant.ofEpochMilli(e.getWhen()));
            if(beep) Toolkit.getDefaultToolkit().beep();
            // if(TalkingClock.this.beep) Toolkit.getDefaultToolkit().beep();
            // if(outer.beep) Toolkit.getDefaultToolkit().beep();
        }
    }
}
// main
public class TalkingClkTest {
    public static void main(String[] args) {
        TalkingClock talkingClock = new TalkingClock(1000, true);
        // 内部类的创建
        // TalkingClock.TimePrinter listener = talkingClock.new TimePrinter();
        talkingClock.start();
        JOptionPane.showMessageDialog(null, "Quit program");
        System.exit(0);
    }
}
~~~

- 内部类中声明的所有静态字段必须是final，并初始化为一个编译时常量。内部类不能有static方法。

- 内部类是编译器现象，与虚拟机无关。编译器会把内部类转换为常规类文件，虚拟机对此一无所知。
  - TalkingClock类内部的TimePrinter类将被转换为类文件TalkingClock$TimerPrinter。
- 内部类的编译机制

~~~java
// this$0 是编译器生成的额外实例字段，对应外围类的引用
Enter class name(e.g. java.util.Date): 
extendsLearn.InnerClass.TalkingClock$TimePrinter
public class extendsLearn.InnerClass.TalkingClock$TimePrinter
{
   public extendsLearn.InnerClass.TalkingClock$TimePrinter(extendsLearn.InnerClass.TalkingClock);

   public void actionPerformed(java.awt.event.ActionEvent);

   final extendsLearn.InnerClass.TalkingClock this$0;
}
// 编译器在外围类添加的静态方法access$000 通过这个字段可以访问参数：TalkingClock对象的beep
Enter class name(e.g. java.util.Date): 
extendsLearn.InnerClass.TalkingClock
public class extendsLearn.InnerClass.TalkingClock
{
   public extendsLearn.InnerClass.TalkingClock(int, boolean);

   static boolean access$000(extendsLearn.InnerClass.TalkingClock);
   public void start();

   private int interval;
   private boolean beep;
}
// if(beep) == if(TalkingClock.access$000(outer))
~~~

- 局部内部类：在一个方法中局部定义类。
  - 局部类不能有访问说明符，作用域被限定在声明这个类的块中。

~~~java
public void start(int interval, boolean beep){
    // 局部内部类
    class TimePrinter implements ActionListener{		// 不能有访问声明符
        public void actionPerformed(ActionEvent e){
            System.out.println(Instant.ofEpochMilli(e.getWhen()));
            // 局部类不仅能访问外部类的字段，还能访问事实最终局部变量。
            // beep字段将被复制为start方法的局部变量
            if(beep) Toolkit.getDefaultToolkit().beep(); 	
        }
    }
    TimePrinter printer = new TimePrinter();
    Timer timer = new Timer(interval, printer);
    timer.start();
}
~~~

- 匿名内部类：甚至不需要为类指定名字，只想要类的一个对象。
  - 匿名内部类不能有构造器，因为它没有名字。
  - 实际上构造参数要传递给超类构造器。
  - 如果内部类实现了一个接口，就不能有构造参数。

~~~java
public void start(int interval, boolean beep){
    ActionListener listener = new ActionListener(){
        public void actionPerformed(ActionEvent e){
            System.out.println(Instant.ofEpochMilli(e.getWhen()));
            if(beep) Toolkit.getDefaultToolkit().beep(); 	
        }
    }
    Timer timer = new Timer(interval, printer);
    timer.start();
}
// SuperType可以是接口和类 内部类需要实现或者扩展该接口或类。
new SuperType(construction parameters){
    inner class methods and data
}
~~~

- 定义类对象和定义匿名内部类对象的直观区别：

~~~java
Person queen = new Person("Marry");				// 定义类对象
Person count = new Person("Dracula"){ ... };	// 定义匿名内部类对象
~~~

~~~java
// lambda表达式实现的事件监听器
public void start(int interval, boolean beep){
    Timer timer = new Timer(interval, e -> {
        System.out.println(Instant.ofEpochMilli(e.getWhen()));
        if(beep) Toolkit.getDefaultToolkit().beep(); 
    });
   	timer.start();
}
~~~

- 静态内部类：将内部类声明为static，此时内部类将不会有指向外围类对象的引用。
  - 静态内部类可以有静态字段和方法。
  - 接口中声明的内部类自动是 static 和 public的。

~~~java
// 在大型项目中想定义一个pair，可能会命名冲突。于是，把pair定义为内部类，同时如果pair不访问外部类的字段，应该设置为static
public class StaticInnerClsTest {
    public static void main(String[] args) {
        double[] doubles = new double[20];
        for(int i = 0; i < doubles.length; i++){
            doubles[i] = 100 * Math.random();
        }
        ArrayAlg.Pair p = ArrayAlg.minmax(doubles);
        System.out.println(p.getFirst());
        System.out.println(p.getSecond());
    }
}
class ArrayAlg{
    public static class Pair{
        private double minv;
        private double maxv;
        public Pair(double minv, double maxv){
            this.minv = minv;
            this.maxv = maxv;
        }
        public double getFirst(){
            return minv;
        }

        public double getSecond(){
            return maxv;
        }
    }
    public static Pair minmax(double[] doubles){
        double minv = doubles[0];
        double maxv = doubles[1];
        for (double aDouble : doubles) {
            minv = Math.min(minv, aDouble);
            maxv = Math.max(maxv, aDouble);
        }
        return new Pair(minv, maxv);
    }
}
~~~

- 外部类以外的其他类需要通过完整的类名访问静态内部类中的静态成员，如果要访问静态内部类中的实例成员，则需要通过静态内部类的实例。
- 静态内部类可以直接访问外部类的静态成员，如果要访问外部类的实例成员，则需要通过外部类的实例去访问。

### 5.4 代理

- 利用代理可以 在运行时 创建 实现了一组给定接口的 新类。
- 代理类包含以下方法：
  - 指定接口所需要的全部方法。
  - Object类中的全部方法。
- 需要提供一个调用处理器（实现了InvocationHandler接口的类对象）。InvocationHandler接口只有一个方法：

~~~java
// 无论何时调用代理对象的方法，invoke方法都会被调用，并传递Method对象和原调用的参数。
// 之后调用处理器必须确定如何处理这个调用。
Object invoke(Object proxy, Method method, Object[] args)
~~~

- 想要创建一个代理对象，需要用到proxy类的newProxyInstance方法。三个参数如下：
  - 一个类加载器
  - 一个Class对象数组，指定需要实现的各个接口
  - 一个调用处理器
- 使用代理可能的目的：
  - 将方法调用路由到远程服务器。
  - 在运行的程序中将用户界面时间和动作关联起来。
  - 为了调试，跟踪方法调用。

~~~java
public class ProxyTest {
    public static void main(String[] args) {
        Object[] elements = new Object[1000];
        for(int i = 0; i < elements.length; i++){
            Integer value = i + 1;
            TraceHandler handler = new TraceHandler(value);
            // 代理对象 系统类加载器加载应用类 Class对象数组：元素对应要实现的接口 调用处理器
            Object proxy = Proxy.newProxyInstance(ClassLoader.getSystemClassLoader(),
                    new Class[]{Comparable.class}, handler);
            elements[i] = proxy;
        }
        Integer key = new Random().nextInt(elements.length) + 1;
        // 代理对象在执行binarySearch时，会执行 element[i].compareTo(key) < 0
        // compareTo调用了TraceHandler类中的invoke方法
        int result = Arrays.binarySearch(elements, key);
        // toString方法也会被代理
        if(result >= 0) System.out.println(elements[result]);
    }
}
// 调用处理器的类 需要实现InvocationHandler
class TraceHandler implements InvocationHandler {
    // 要包装的对象
    private Object target;

    public TraceHandler(Object t){
        target = t;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.print(target);
        System.out.print("." + method.getName() + "(");
        if(args != null){
            for(int i = 0; i < args.length; i++){
                System.out.print(args[i]);
                if(i < args.length - 1)     System.out.println(", ");
            }
        }
        System.out.println(")");
        return method.invoke(target, args);
    }
}
//
500.compareTo(510)
750.compareTo(510)
625.compareTo(510)
562.compareTo(510)
531.compareTo(510)
515.compareTo(510)
507.compareTo(510)
511.compareTo(510)
509.compareTo(510)
510.compareTo(510)
510.toString()
510
~~~

- 所有代理类都扩展了Proxy类，一个代理类只有一个实例字段——调用处理器，它在Proxy类中定义。完成代理对象任务所需要的额外数据要存储在调用处理器中。
- 所有的代理类都要覆盖（重写）Object类的toString、equals、hashCode方法。这些方法只是在调用处理器上调用invoke。
- 对于一个特定的类加载器和预设的一组接口来说，只能有一个代理类。两次newProxyInstance调用会生成同一个类的两个对象。
  - 代理类总是public和final的。

## 六、异常、断言和日志

### 6.1 异常

- 异常处理的任务是将控制权从产生错误的地方转移到能处理这种情况的错误处理器。
- java所有的异常都派生自Throwable类。分为Error和Exception。
  - Error描述了java运行时系统的内部错误和资源耗尽错误。
  - Exception分为 **RuntimeException **和 **其他异常**(如：IOException)。一般编程错误异常属于前者，I/O类问题导致的异常属于后者。
- RuntimeException：错误的强制类型转换、数组访问越界、访问null指针。
- 不派生于RuntimeException的异常：读文件尾后数据、打开不存在文件、Class.forName参数表示的类不存在。
- 如果出现RuntimeException异常，那么一定是我的问题。
- **非检查型异常**：派生自Error类和RuntimeException类的所有异常。
- **检查型异常**：所有其他异常。
  - 需要为此类异常提供异常处理器。编译器会检查作业。

- 4种会抛出异常的情况：
  - 调用了一个抛出检查型异常的方法。FileInputStream构造器。
  - 检测到一个错误，并利用throw语句抛出一个检查型异常。
  - 程序出现错误，如a[ -1] = 0 会抛出一个非检查型异常。
  - JVM或运行时库出现内部错误。

- 一个方法必须声明所有可能抛出的检查型异常。如上述的前两条。
- 子类重写超类的方法时，子类声明的检查性异常不能比超类的更通用。
  - 如果超类没有抛出检查型异常，则子类也不能抛出。
- 自定义异常的习惯性做法

~~~java
public class FileFormatException extends IOException {
    public FileFormatException(){}

    public FileFormatException(String gripe){
        super(gripe);
    }
}
~~~

- 捕获异常：
  - 如果try语句块中的任何代码抛出了catch字句中指定的一个异常类，那么
    - 程序将跳过try语句块的其他代码
    - 执行catch字句中的处理器代码
  - 如果没有抛出任何异常，将跳过catch子句。
  - 如果抛出了catch子句中没有声明的一个异常类型，方法将立即退出。

- 一般经验：要捕获知道如何处理的异常，继续传播不知道怎么处理的异常。
  - 例外：子类重写超类方法，超类没抛出任何异常，那子类只能自己捕获处理。
- 捕获多个异常：

~~~java
try{
    //
}
catch(FileNotFoundException | UnknownHostException e){
    // 当两个异常需要的处理语句是一样的，可以合并
    // 异常变量隐含为final的 不能为e赋不同的值
}
catch(IOException e){
    e.getMessage();
    e.getClass().getName();
}
catch(SQLException e){
    // 可以在catch子句中再次抛出异常，可以改变异常类型，隐藏底层异常细节
    Exception exc = new ServletException("database error" + e.getMessage());	
    exc.initCause(e);		// 设置原始异常为新异常的原因
    throw exc;
}
// 抛出高层异常而不丢失原始异常原因
Trowable original = caughtException.getCause();
~~~

- 不管是否有异常被捕获，**finally子句中的代码都会执行**。主要用于资源清理。
- 但是，如下情况finally块中的close()，也需要抛出异常。

~~~java
public static void main(String[] args) {
    InputStream in = new InputStream() {
        @Override
        public int read() throws IOException {
            return 0;
        }
    };
    try{

    }
    catch(Exception e){
        e.printStackTrace();
    }
    finally{
        in.close();
    }
}
~~~

- 就需要try嵌套，外层try报告错误，内层try关闭输入流。

- try-with-Resources：当资源属于一个实现了AutoCloseable接口的类时，应该使用此方法关闭资源。

~~~java
try(Scanner in = new Scanner(new FileInputStream("/usr/words"),
                             String.valueOf(StandardCharsets.UTF_8));
    PrintWriter out = new PrintWriter("out.txt",
                                      String.valueOf(StandardCharsets.UTF_8)))
{
    while(in.hasNext()){
        out.println(in.next().toUpperCase());
    }
}
catch(Exception e){
    e.printStackTrace();
}
~~~

- 分析堆栈轨迹元素

~~~java
public static void main(String[] args) {
    Throwable throwable = new Throwable();
    StringWriter stringWriter = new StringWriter();
    throwable.printStackTrace(new PrintWriter(stringWriter));
    System.out.println(stringWriter.toString());
}
// StackWalker类会生成StackWalker.StackFrame实力流，其中每个实例分别描述一个栈帧。
// 迭代处理栈帧
StackWalker walker = StackWalker.getInstance();
walker.forEach(frame -> analyze frame)
// 懒处理 Stream<StackWalker.StackFrame>
walker.walk(stream -> process stream)
~~~

~~~java
public static void main(String[] args) {
    try(var in = new Scanner(System.in)){
        System.out.print("Enter n:");
        int n = in.nextInt();
        factorial(n);
    } 
}
public static int factorial(int n){
    System.out.println("factorial(" + n + "):");
    StackWalker walker = StackWalker.getInstance();
    walker.forEach(System.out::println);
    int r;
    if (n <= 1) r = 1;
    else r = n * factorial(n - 1);
    System.out.println("return: " + r);
    return r;
}
// stack：2层->3层->4层
Enter n:3
factorial(3):
Main.factorial(Main.java:15)
Main.main(Main.java:8)
factorial(2):
Main.factorial(Main.java:15)
Main.factorial(Main.java:18)
Main.main(Main.java:8)
factorial(1):
Main.factorial(Main.java:15)
Main.factorial(Main.java:18)
Main.factorial(Main.java:18)
Main.main(Main.java:8)
return: 1
return: 2
return: 6
~~~

### 6.2 断言

- 断言机制允许在测试期间向代码插入一些检查，而生产代码中会自动删除这些检查。

~~~java
assert condition;
assert condition : expression;
// 如果条件为false，会抛出一个AssertionError异常。expression是该异常的构造器参数，会转换为一个消息字符串。
~~~

- 默认情况下，断言是禁用的。可以在运行程序是用 -enableassertions 或 -ea 选项启用断言。
- 启用或禁用断言是类加载器的功能，不必重新编译程序。禁用断言时，类加载器会去除断言代码。
- 断言使用：
  - 断言失败是致命的、不可恢复的错误。
  - 断言检查只是在开发和测试阶段打开。

### 6.3 日志

- 待补充。。。



## 七、泛型程序设计

- 匿名子类

~~~java
ArrayList<String> passwords = new ArrayList<>(){
    public String get(int n){
        return super.get(n).replaceAll(".", "*");		// 匿名子类
    }
};
passwords.add("123456");
String s = passwords.get(0);	// ******
//
public static void main(String[] args) {
    Thread th=new Thread(new Runnable()
                         {
                             @Override
                             public void run()
                             {
                                 System.out.println("runnable");
                             }
                         }) //这里是完成参数的传递，即完成Thread构造器
    {
        @Override
        public void run()
        {
            System.out.println("Thread");
        }
    };//这里是重写父类run()的方法
    th.start();			// Thread
}
~~~

- 泛型类：有一个或多个类型变量的类。
- 泛型方法可以在普通类中定义，也可以在泛型类中定义。

~~~java
public class Pair<T, U> {
    private T first;
    private U second;

    public Pair(){
        first = null;
        second = null;
    }

    public Pair(T f, U s){
        first = f;
        second = s;
    }

    public T getFirst(){
        return first;
    }

    public U getSecond(){
        return second;
    }

    public void setFirst(T newValue){
        first = newValue;
    }

    public void setSecond(U newValue){
        second = newValue;
    }
    // 类型变量放在修饰符后面，返回类型前面
    public static <T> T getMiddle(T... a){
        return a[a.length/2];
    }
}
~~~

~~~java
public class ArrayAlg {
    public static <T> T getMiddle(T... a){
        return a[a.length/2];
    }
}
public static void main(String[] args) {
    String middle = ArrayAlg.<String>getMiddle("Jone", "July");
    // String middle = ArrayAlg.getMiddle("Jone", "July");
    double mid = ArrayAlg.getMiddle(3.14, 1729, 0);		// ERROR
}
~~~

- 类型变量T的限定

~~~java
public static <T extends Comparable> T min(T[] a){		// Raw use of parameterized class 'Comparable'
    if(a == null || a.length == 0)  return null;
    T smallest = a[0];
    for (T ele : a) {
        if (smallest.compareTo(ele) > 0) {
            smallest = ele;
        }
    }
    return smallest;
}
// 这里Comparable虽然是接口 但也用extends表示子类型的意思
// 一个类型变量可以有多个限定 用&隔开；多个类型变量可以用逗号隔开
<T extends Comparable & Serializable, U extends Comparable>
~~~

- 虚拟机没有泛型类型变量，所有对象都属于普通类。
- 类型擦除：无论何时定义一个泛型类型，都会自动提供一个相应的原始类型。类型变量会被擦除并替换为其限定类型。
  - 用第一个限定来替换类型变量
  - 编译器会在必要时进行强制类型转换，来切换限定类型
    - 因此，应该将没有方法的接口（标签接口）放在限定列表末尾
  - 无限定的变量则替换为Object
- 编写一个泛型方法调用时，如果擦除了返回类型，编译器会插入强制类型转换。
- 同理，访问一个泛型字段时，也要插入强制类型转换。

~~~java
Pair<Employee> buddies = ...;
// getFirst在类型擦除后，会返回Object。编译器自动插入了Object到Employee的强制类型转换。
Employee buddy = buddies.getFirst();
Employee buddy = buddies.first;
~~~

- 通过合成桥方法来保持多态

~~~java
public class DateInterval extends Pair<LocalDate>{
    public DateInterval(LocalDate f, LocalDate s){
        super(f, s);
    }
    // 类型擦除后 Pair<LocalDate>变成了Pair。于是，DateInterval应该继承一个 setSecond(Object)的方法
    @Override
    public void setSecond(LocalDate second){
        super.setSecond(second);
    }
    
    @Override
    public LocalDate getSecond(){
        return super.getSecond();
    }
}
// 
public static void main(String[] args) {
    Pair<LocalDate> pair = new DateInterval(LocalDate.now(), LocalDate.now());
    // 在Pair<LocalDate>类型的pair上调用setSecond，会调用DateInterval.setSecond(Object)方法
    // 但显然和多态冲突了，aDate更适合LocalDate
    pair.setSecond(aDate);
}
// 于是编译器为DateInterval合成桥方法setSecond(Object object)
public void setSecond(Object second){
    setSecond((LocalDate)second);
}
// getSecond同理
public LocalDate getSecond(){
    return (LocalDate)super.getSecond();
}
~~~

- 基本类型不能作为类型参数。因为基本类型不能转换为Object。
- 所有类型查询只产生原始类型。

~~~java
Pair<String> stringPair = ...;
Pair<Employee> employeePair = ...;
stringPair.getClass() == employeePair.getClass() == Pair.class	// true 都是Pair.class
~~~

- 不允许 *创建* 参数化类型的数组。即：java不支持泛型类型的数组

~~~java
var table = new Pair<String>[10];	// Error
Pair<String> ps[];					// OK
 ArrayList<Pair<String>> arrayList = new ArrayList<>();				// 好的实践
~~~

- 不能实例化类型变量
  - 用反射和Supplier

~~~java
public Pair(){
    first = new T();		// ERROR
}
// java8之后 最好的办法就是让调用者提供一个构造器表达式。
// makePair接收一个Supplier<T>
Pair<String> p = Pair.makePair(String::new);
//
public static <T> Pair<T> makePair(Supplier<T> constr){
    return new Pair<>(constr.get(), constr.get());
}
// 用反射调用Constructor.newInstance方法来构造对象
// T.class.getConstructor().newInstance()是错误的，T.class()将被类型擦除
public static <T> Pair<T> makerPair(Class<T> cl){
    try{
        return new Pair<>(cl.getConstructor().newInstance(), cl.getConstructor().newInstance());
    }
    catch(Exception e){	return null;}
}
// new T() 和 T.class 都不行
// Class<T> cl: cl.getConstructor().newInstance() 和 Supplier<T> constr: constr.get() 可以
~~~

- 不能构造泛型数组，因为类型擦除总会实例化一个限制类型数组。

~~~java
// ArrayList的实际实现 ？
public class ArrayList<E>{
    private E[] elements;
    
    public ArrayList(){
        elements = (E[]) new Object[10];
    }
}
~~~

- 不能在静态字段和方法中引用类型变量。

~~~java
public class Singleton<T>{
    private static T singleInstance;
    public static T getSingleInstance(){
        if(singleInstance == null)	construct new instance of T;
        return singleInstance;
    }
}
// 如果static可行 那么Singleton<A>.getSingleInstance和Singleton<B>.getSingleInstance 都能得到 Object singleInstance
~~~

- 泛型类扩展Throwable是不合法的，不能抛出或捕获泛型类的对象。
- 不过在异常规范中使用类型变量是合法的。

~~~java
// ERROR
public class Problem<T> extends Exception{};
// ERROR
try{}
catch(T e){}
// OK
public static <T extends Throwable> void doWork(Class<T> t){}
~~~

- 可以取消对检查型异常的检查

~~~java
// java中，必须为检查型异常提供处理器，可以利用泛型取消该机制
@SuppressWarnings("unchecked")
static <T extends Throwable> void throwAs(Throwable t) throws T{
    // 异常类型强制转化
    throw (T) t;
}
// Task接口中包含上述方法，编译器会认为e是一个非检查型异常
Task.<RuntimeException> throwAs(e);
// 下述代码会将所有异常转换为非检查型异常：
try{}
catch(Throwable t){
    Task.<RuntimeException> throwAs(t);
}
// 下为完整实现
public interface Task {
    void run() throws Exception;

    @SuppressWarnings("unchecked")
    static <T extends Throwable> void throwAs(Throwable t) throws T{
        throw (T) t;
    }
    // 从Task到Runnable的适配器
    static Runnable anRunnable(Task task){
        return () -> {
            try{
                task.run();
            }
            catch(Exception e){
                Task.<RuntimeException> throwAs(e);
            }
        };
    }
}
public static void main(String[] args) {
    Thread thread = new Thread(Task.anRunnable( () -> {
        Thread.sleep(1000);
        System.out.println("Hello World!");
        throw new Exception("Check this out!");
    }));
    thread.start();
}
~~~

- 注意擦除后的冲突
  - 倘若两个接口类型是同一接口的不同参数化，一个类或类型变量就不能同时作为这两个接口类型的子类。


~~~java
class Employee implements Comparable<Employee>{}
class Manager extends Employee implements Comparable<Manager>{}
// 这是错误的，因为Manager同时实现了Comparable<Employee>和Comparable<Manager>
// 原因是合成的桥方法可能会冲突
// 不能对不同类型的X有两个这样的方法
public int compareTo(Object other){
    return compareTo((X) other);
}
~~~

- 无论S和T有什么关系，Pair<S>和Pair<T>都没有任何关系。

- 总是可以将参数化类型转换为一个原始类型。
- 泛型类可以扩展或实现其他的泛型类。
  - ArrayList<T>类实现了List<T>接口。
- 通配符类型

~~~java
Pair<Manager>是Pair<? extends Employee>的子类型;
// 限定为Manager的所有超类型
Pair<? super Manager>;
~~~

- 带有超类型限定的通配符允许写入一个泛型对象，带有子类型限定的通配符允许读取一个泛型对象。

~~~java
// 带有子类型限定的通配符的set方法拒绝任何特定类型的参数，因为？不能匹配。get的结果赋给Employee引用是可以的
? extends Employee getFirst();
void setFirst(? extends Employee);
// 带有超类型限定的通配符的get方法不能确定返回对象的类型，只能用Object接收。set方法只能传递Manager类型的对象或者某个子类型对象。
void setFirst(? super Manager);
? super Manager getFirst();
~~~

- 超类型

~~~java
public static <T extends Comparable<? super T>> T min(T[] a){}
// 传入T类型对象肯定是安全的
// 可以同时满足 String与Comparable<String> 和 LocalDate与Comparable<ChronoLocalDate>
~~~

- 无限定通配符

~~~java
Pair<?>;
? getFirst();
void setFirst(?);
// 与原始类型相比，setFirst不可用，getFirst返回值只能赋给Object
// 用途
public static boolean hasNulls(Pair<?> p){
    return p.getFirst() == null || p.getSecond() == null;
}
// 也能用泛型做
~~~

- 通配符捕获
  - 必须保证通配符表示单个确定的类型时，通配符才能被捕获。

~~~java
public static void minmaxBonus(Manager[] a, Pair<? super Manager> result){
    if(a.length == 0)	return;
    Manager min = a[0];
    Manager max = a[0];
    for(int i = 1; i < a.length; i++){
        if(min.getBonus() > a[i].getBonus())	min = a[i];
        if(max.getBonus() < a[i].getBonus())	max = a[i];
    }
    result.setFirst(min);
    result.setSecond(max);
}
public static void maxminBonus(Manager[] a, Pair<? super Manager> result){
	minmaxBonus(a, result);
    PairAlg.swap(result);	// 这里通配符捕获机制就不可避免了
}
class PairAlg{
    public static boolean hasNulls(Pair<?> p){
        return p.getFirst() == null || p.getSecond() == null;
    }
    public static void swap(Pair<?> p){
        swapHelper(p);
    }
    public static <T> void swapHelper(Pair<T> p){
        T t = p.getFirst();
        p.setFirst(p.getSecond());
        p.setSecond(t);
    }
}
~~~

- 反射与泛型

~~~java
// 使用Class<T>参数进行类型匹配
public static <T> Pair<T> makePair(Class<T> c)throws InstantiationException, IllegalAccessException{
    return new Pair<>(c.newInstance(), c.newInstance());
}
// Employee.class是Class<Employee>的一个对象，所以很容易推断出T同Employee匹配
makePair(Employee.class)
~~~

- java泛型在虚拟机中擦除泛型类型，原始Pair类知道它源于泛型类Pair<T>。

~~~java
public static <T extends Comparable<? super T>> T min(T[] a);
// 类型擦除后得到
public static Comparable min(Comparable[] a);
// 可以通过反射API重新构造实现者声明的泛型类和方法的所有内容
~~~

- java.lang.reflect包的接口Type包含以下子类型：
  - Class类，描述具体类型
  - TypeVariable接口，描述类型变量
  - WildcardType接口，描述通配符
  - ParameterizedType接口，描述泛型类或接口类型
  - GenericArrayType接口，描述泛型数组

## 八、集合

- java类库中，集合类的基本接口是Collection接口。
- "for each"循环可以处理任何实现了Iterable接口的对象。Collection接口扩展了Iterable接口，所以标准类库的任何集合都可以使用"for each"。

~~~java
public interface Iterator<E>{
    E next();
    boolean hasNext();
    void remove();		// 删除上次调用next方法是返回的元素
    default void forEachRemaining(Consumer<? super E> action);
}
//
iterator.forEachRemaining(ele -> do something with ele);
// 访问顺序取决于集合类型，ArrayList会从索引0开始顺序访问，HashSet则基本随机。
~~~

- 集合框架中的接口

<img src="C:\Users\Lenovo\Desktop\鱼姬玄的东西\Typora图片\集合框架中的接口.png" style="zoom: 67%;" />

- 集合有两个基本接口：Collection和Map。以Map结尾的类实现了Map接口，其他都实现了Collection接口。

> 链表 LinkedList

- java中，所有链表都是双向链表。
- 多个迭代器并发操作链表时，可能出现错误。链表迭代器能检测到这种修改，并抛出ConcurrentModificationException异常。

> 数组列表 ArrayList

- Vector是同步的，不需要同步就用ArrayList。

> 散列表 hash table

- 散列表可以用于实现很多重要的数据结构。HashSet就是基于散列表的集。

> 树集 TreeSet

- TreeSet是有序集合，排序是红黑树实现的。
- 使用树集，要求元素可比较，必须实现Comparable接口。

> 队列和双端队列

- ArrayDeque和LinkedList都实现了Deque接口，可以提供双端队列，大小可扩展。

> 优先队列

- 同TreeSet一样，优先队列既可以保存实现了Comparable接口的类对象，也能保存构造器中提供的Comparator对象。
  - Comparable 自然排序。（实体类实现compareTo方法）
  - Comparator 是定制排序。（无法修改实体类时，直接在调用方创建）


> 映射 map

- HashMap和TreeMap。
- 键是唯一的。对一个键调用两次put方法，第二个值会取代第一个，并返回第一个值。

~~~java
counts.put(word, counts.getOrDefault(word, 0) + 1);
// 或者用下两句替代
counts.putIfAbsent(word, 0);		// 如果没有word，就put 0
counts.put(word, counts.get(word) + 1);
// 或者
counts.merge(word, 1, Integer::sum);	// word和1组合，否则Integer::sum组合原value和1
~~~

- 映射本身不是集合，但是可以得到映射的视图——实现了Collection接口或某个子接口的对象。
  - 键集、值集合 以及 键值对集。

~~~java
Set<K> keySet();		// 既不是HashSet，也不是TreeSet
Collection<V> values();
Set<Map.Entry<K, V> entrySet()
~~~

- 在键集视图中调用迭代器的remove方法会删除该键以及与它关联的值。不能在键集视图上添加元素。
- 映射条目集视图有同样的限制

~~~java
for(Map.Entry<String, Employee> entry : staff.entrySet()){
    String k = entry.getKey();
    Employee v = entry.getValue();
}
~~~

> 弱散列映射 WeakHashMap

- 在垃圾回收机制看来，如果映射对象是活动的，那其中的所有桶就都是活动的。如果某个key的最后一个引用已经消失，那将没有任何途径可以访问其对应的值。前述两条机制叠加，HashMap将造成空间浪费，而WeakHashMap可以解决这个问题。
- WeakHashMap使用弱引用保存键。WeakReference对象将包含散列表键的引用，当某个对象只能由WeakReference引用时，将该弱引用放入一个队列中。WeakHashMap将删除相关联的映射条目。

> 链接散列表和映射 LinkedHashMap和LinkedHashSet

- 双向链表存储元素项。这样 LinkedHashMap和LinkedHashSet 的项就是有序的。
- 结构是散列桶+桶内链表。链表的前驱后继关系不受桶影响。
- 用访问顺序而不是插入顺序来迭代处理映射条目：

~~~java
LinkedHashMap<K, V>(initialCapacity, loadFactor, true);
// 每次调用get或put时，受影响的项从当前位置删除，并放到桶链表的尾部。
// 访问顺序对于实现缓存的“最近最少使用”原则十分重要
// 例如：如果在表中找不到要的项而表已经相当满时，可以删除表的前几个元素，即LRU的元素
var cache = new LinkedList<K, V>(128, 0.75F, true){
    protected boolean removeEldestEntry(Map.Entry<K, V> eldest){
        return size() > 100;
    }
}
~~~

> EnumSet

- 枚举元素集，值在集中，则对应位置1。

~~~java
enum Weekday{ MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY };
EnumSet<Weekday> always = EnumSet.allof(Weekday.class);
var never = EnumSet.noneof(Weekday.class);
var workday = EnumSet.range(Weekday.MONDAY, Weekday.FRIDAY);
var mwf = EnumSet.of(Weekday.MONDAY, Weekday.FRIDAY);
~~~

> EnumMap

- 键为枚举类型的映射。

> 标识散列映射 IdentityHashMap

- 键的散列值通过System.identityHashCode而不是hashCode计算，根据对象的内存地址计算散列码。两个对象在比较时，IdentityHashMap类使用==，而不是equals
  - 键对象即使内容相同，也被视为不同对象。

> 视图和包装器

- 视图：keySet方法返回一个实现了Set接口的类对象，由这个类的方法操作原映射。这种集合叫做视图。

~~~java
List<String> names = List.of("Peter", "Paul", "Mary");
Set<Integer> numbers = Set.of(2, 3, 5);
Map<String, Integer> scores = Map.of("Peter", 2, "Paul", 3, "Mary", 5);
// 元素、键、值不能为null 这些集合对象是不可修改的。
Map<String, Integer> scores = ofEntries(entry("Peter", 2), entry("Paul", 3), entry("Mary", 5));
// 如果想要一个可变更的集合：
var names = new ArrayList<>(List.of(...));
~~~

- Collections类包含很多实用方法，参数和返回值都是集合。
  - Collections.nCopies、Collections.emptySet、Collections.singleton等。
- java没有Pair类，java9之后可以用Map.Entry(first, second)来替代。

- 子范围视图：

~~~java
// 对子范围的任何操作都会反映到整个列表
List<Employee> group2 = staff.subList(10, 20);
// 对于有序集和映射，可以使用排序顺序而不是元素位置建立子范围。
SortedSet<E> subSet(E from, E to);
SortedSet<E> headSet(E to);
SortedSet<E> tailSet(E from);
// 
SortedMap<E> subMap(E from, E to);
SortedMap<E> headMap(E to);
SortedMap<E> tailMap(E from);
// NavigableSet接口允许指定是否包含边界
~~~

- 不可修改的视图

~~~java
Collections.unmodifiable*;
// 返回一个实现了List接口的类对象，不能通过视图更改该集合
lookAt(Collections.unmodifiableList(staff));
// 视图只包装了接口而不是具体的集合对象，所以只能访问接口定义的方法。
// 如：addFirst和addLast是LinkedList类的方法，不是List接口的方法，就不行。
~~~

- 同步视图：类库设计者使用视图机制来确保常规集合是线程安全的，如Collections类的静态synchronizedMap方法。

~~~java
var map = Collections.synchronizedMap(new HashMap<String, Employee>());		// put和get等方法是同步的
~~~

- 检查型视图：用来对泛型类型可能出现的问题提供调试支持。

~~~java
var strings = new ArrayList<String>();
ArrayList rawList = strings;
rawList.add(new Date());
// 这个错误在add时检查不到，只有在get方法强制类型转换时，才会抛出异常
// 检查型视图可以探测这类问题
List<String> safeStrings = Collections.checkedList(strings, String.class);
safeStrings.add(new Date());	// ClassCastException
// 检查型视图受限于虚拟机可以完成的运行时检查，对于ArrayList<Pair<String>>，由于虚拟机类型擦除后有原始Pair，所以无法阻止Pair<Date>插入。
~~~

> 算法

- 泛型最大值算法

~~~java
public static <T extends Comparable> T max(Collection<T> c){
	if(c.isEmpty()) throw new NoSuchElementException();
    Iterator<T> iter = c.iterator();
    T largest = iter.next();
    while(iter.hasNext()){
        T next = iter.next();
        if(largest.compareTo(next) < 0){
            largest = next;
        }
    }
    return largest;
}
~~~

- 排序与混排

~~~java
var staff = new LinkedList<String>();
Collections.sort(staff);	// 假定staff实现了Comparable接口
staff.sort(Comparator.comparingDouble(Employee::getSalary));
// 降序排序 Collections.reverseOrder()
staff.sort(Comparator.reverseOrder());
staff.sort(Comparator.comparingDouble(Employee::getSalary).reversed());
//
var cards = new ArrayList<Card>();
Collections.shuffle(cards);
~~~

- 二分查找

~~~java
int idx = Collections.binarySearch(c, element);
int idx = Collections.binarySearch(c, ele, comparator);
~~~

- 一些API

~~~java
Collections.replaceAll(words, "C++", "Java");
words.removeIf(w -> w.length() <= 3);
words.replaceAll(String::toLowerCase);
coll1.removeAll(coll2);		// 从coll1中删除coll2中出现的所有元素
coll1.retainAll(coll2);		// 从coll1中删除所有未在coll2中出现的元素	保持
// 交集
var result = new HashSet<String>(firstSet);
result.retainAll(secondSet);
// 数组转集合
String[] values = ...;
var staff = new HashSet<>(List.of(values));
// 集合转数组，但Object[]且不能强转
Object[] values = staff.toArray();
String[] values = staff.toArray(new String[0]);
~~~

> 遗留集合

- Hashtable类：同步的HashMap。但如果想要并发访问，应该用ConcurrentHashMap，否则，应该用HashMap。
- Enumeration接口。
- 属性映射：
  - 键和值都是字符串。
  - 可以很容易的保存到文件并从文件加载。
  - 有一个二级表存放默认值。


~~~java
var settings = new Properties();
settings.setProperty("width", "600.0");
settings.setProperty("filename", "/home/raven.html");
// 
var out = new FileOutputStream("program.properties");
settings.store(out, "Program Properties");		// 第二个参数是注解
//
var in = new FileInputStream("program.properties");
settings.load(in);
// 可以通过System.getProperties生成Properties对象描述系统信息，如：
var usrDir = System.getProperties("usr.home");
// 通过二级属性映射来指定默认值
var defaultSettings = new Properties();
defaultSettings.setProperty("width", "200.0");
defaultSettings.setProperty("height", "400.0");
var settings = new Properties(defaultSettings);
// 如果想要有层次结构的配置信息，应该改用Preferences类
~~~

- 栈：扩展了Vector类。
- 位向量(BitSet)：提供了便于读取、设置或重置各个位的接口。可以避免掩码或者其他调整位的操作。

## 九、并发

### 9.1 线程状态

- Thread.start方法将创建一个执行run方法的新线程。
- java线程的六种状态：新建、可运行、阻塞、等待、计时等待、终止。
  - 新建：new Thread(r);
  - 可运行：调用start方法。可能正在运行也可能没有运行。
  - 阻塞：当一个线程试图获得一个内部的对象锁，而该锁被其他线程占用时，线程阻塞。
    - 内部的对象锁不是java.util.concurrent库的Lock。

  - 等待：当线程等待另一个线程 通知调度器出现一个条件时，线程等待。
    - Object.wait、Thread.join、java.util.concurrent库的Lock、Condition。
    - 等待和阻塞没有太大区别。

  - 计时等待：方法有超时参数。


- 线程终止：run方法正常退出、因为一个没有捕获的异常终止了run方法，线程意外终止。
  - 除了废弃的stop方法，没有办法可以强制线程终止。如果希望一个线程停止，应该中断它。

### 9.2 线程属性

> 中断线程

- interrupt方法可以请求终止一个线程。这会设置线程的中断状态。

~~~java
Thread.currentThread().isInterrupted();	// 检查中断状态
~~~

- 如果线程被阻塞，就无法检查中断状态。需要引入InterruptedException异常。当在一个被sleep或wait调用阻塞的线程上调用interrupt方法时，该阻塞调用（sleep或wait）将被一个InterruptedException异常中断。
- 如果设置了中断状态，此时调用sleep方法，线程不会睡眠，而会清除中断状态并抛出InterruptedException。
  - 因此，如果循环里有sleep，面对可能的interrupt，就该捕获InterruptedException，不要检测中断状态。
-  不要抑制（捕获并忽略）InterruptedException，应该抛出它。

> 守护线程 Daemon~(ˈdiːmən)~

- 通过调用 t.setDaemon(true) 可以将一个线程转化为守护线程。
- 守护线程的唯一作用是为其他线程提供服务。
  - 发送时钟信号、清空过时缓存项等。

> 线程名

~~~java
var t = new Thread(runnable);
t.setName("Web crawler");
~~~

> 未捕获异常处理器

- run方法不能抛出检查型异常，抛出非检查型异常可能导致线程终止。
- 在线程死亡之前，异常会传递到一个用于处理未捕获异常的处理器。
- 该处理器是一个实现了Thread.UncaughtExceptionHandler接口的类。可以通过setUncaughtExceptionHandler和setDefaultUncaughtExceptionHandler方法为线程安装处理器。对于没有安装处理器的线程，该线程的ThreadGroup对象就是处理器。

~~~java
// Thread.UncaughtExceptionHandler的唯一方法
void uncaughtException(Thread t, Throwable e)
// 该方法会处理未捕获异常
~~~

> 线程优先级

- 每个线程都有优先级。新建线程默认继承构造它的线程的优先级。setPriority方法可以设置优先级。

- 现在不应使用线程优先级

### 9.3 同步

- 有两种机制可以防止并发访问代码块：synchronized关键字 和 重入锁/递归锁ReentrantLock~(riːˈentrənt)~。

> 锁对象

- ReentrantLock保护代码块的基本结构

~~~java
myLock.lock();
try{
    critical section
}
finally{
    myLock.unlock();
}
// 不能用try-with-resources语句，一个是解锁方法不是close，另一个是锁是共享的，作用域应该高
~~~

~~~java
public class ThreadTest {
    public static final int DELAY = 10;
    public static final int NACCOUNTS = 100;
    public static final double MAX_AMOUNT = 1000;
    public static final double INITIAL_BALANCE = 1000;

    public static void main(String[] args) {
        Bank bank = new Bank(NACCOUNTS, INITIAL_BALANCE);
        for(int i = 0; i < NACCOUNTS; i++){
            int fromAccount = i;
            Runnable r = () -> {
                try{
                    while(true){
                        double amount = MAX_AMOUNT * Math.random();
                        int toAccount = (int)(bank.size() * Math.random());
                        bank.transfer(fromAccount, toAccount, amount);
                        Thread.sleep((int) (DELAY * Math.random()));
                    }
                }
                catch(InterruptedException e){
                    e.printStackTrace();
                }
            };
            Thread t = new Thread(r);
            t.start();
        }
    }
}
//
public class Bank {
    private final double[] accounts;
    private ReentrantLock bankLock = new ReentrantLock();

    public Bank(int n, double initialBalance){
        accounts = new double[n];
        Arrays.fill(accounts, initialBalance);
    }

    public void transfer(int from, int to, double amount){
        bankLock.lock();
        try {
            // if(accounts[from] < amount) return;
            System.out.print(Thread.currentThread());
            accounts[from] -= amount;
            System.out.printf(" %10.2f from %d to %d", amount, from, to);
            accounts[to] += amount;
            System.out.printf(" Total Balance: %10.2f%n", getTotalBalance());
        }
        finally {
            bankLock.unlock();
        }
    }

    public double getTotalBalance(){
        bankLock.lock();
        try{
            double total = 0;
            for(double ele : accounts){
                total += ele;
            }
            return total;  
        }
        finally {
            bankLock.unlock();
        }
    }

    public int size(){
        return accounts.length;
    }
}
~~~

- 重入锁：线程可以反复获得已拥有的锁。锁通过持有计数来跟踪对lock方法的嵌套调用，每次调用lock都必须unlock来释放锁。
  - 被一个锁保护的代码可以调用另一个使用相同锁的方法。
  - 在上例中：transfer方法调用getTotalBalance方法也会封锁bankLock对象，此时bankLock对象的持有计数为2。

- 重入锁可以通过参数构造一个公平策略的锁，但这可能影响性能，所以默认情况下，不要求锁公平
  - 公平锁倾向于等待时间最长的线程，但比常规锁慢很多。而且公平锁无法确保线程调度器是公平的。
- 一个锁对象可以有一个或多个相关联的条件对象。 每个条件对象管理那些已经进入被保护代码段但还不能运行的线程。
- conditionObj.await：当前线程暂停，放弃锁，进入条件的等待集(wait set)，进入阻塞状态。
- conditionObj.signalAll：解除该条件等待集中所有线程的阻塞状态。

~~~java
public void transfer(int from, int to, double amount){
    bankLock.lock();
    try {
        // 这里用while而不是if，await被唤醒并得到锁后，会接着执行，因为不一定满足条件，所以需要while再次进行条件判断。
        while(accounts[from] < amount) {
            sufficientFunds.await();
        }
        System.out.print(Thread.currentThread());
        accounts[from] -= amount;
        System.out.printf(" %10.2f from %d to %d", amount, from, to);
        accounts[to] += amount;
        System.out.printf(" Total Balance: %10.2f%n", getTotalBalance());
        // 这里并不是说满足转账条件了，只是可能满足了，值得唤醒并开始竞争。
        // 如果用signal，会随机选择一个线程唤醒，但如果该线程还是不满足条件，或另被阻塞，可能不满足signal的原始意图。
        sufficientFunds.signalAll();
    } catch (InterruptedException e) {
        e.printStackTrace();
    } finally {
        bankLock.unlock();
    }
}
~~~

> synchronized

- java中的每个对象都有一个内部锁，如果一个方法声明有synchronized关键字，那么对象的锁将保护整个方法。
  - 要调用这个方法，线程必须获得内部对象锁。
- 内部对象锁只有一个关联条件。wait方法将一个线程增加到等待集中，notifyAll/notify方法可以解除等待线程的阻塞

~~~java
class Bank{
    private double[] accounts;
    public synchronized void transfer(int from, int to, int amount) throws InterruptedException{
        while(accounts[from] < amount){
            await();
        }
        accouts[from] -= amount;
        accounts[to] += amount;
        notifyAll();
    }
    public synchronized double getTotalBalance(){...}
}
~~~

- 静态方法也可以声明为同步的。如果Bank类有一个同步静态方法，那么调用该方法时，Bank.class对象的内部锁会锁定。没有线程可以调用该类的任何静态同步方法。
- 同步策略选择优先级建议：java.util.concurrent包中的机制 > synchronized > Lock/Condition。

- 同步块：获得java对象锁的另一种机制。

~~~java
public class Bank{
    private double[] accounts;
    private var lock = new Object();	// 创建lock对象只是为了使用每个java对象拥有的锁
    public void transfer(int from, int to, int amount){
        synchronized(lock){
    		accounts.set(from, accounts.get(from)-amount);
        	accounts.set(to, accounts.get(to)+amount);
		}
    }
}
// 客户端锁定：使用一个对象的锁来实现额外的原子操作。
// 这依赖于Vector类会对自己的所有更改方法使用内部锁，这并不是肯定的，所以不推荐使用
public void transfer(Vector<Double> accounts, int from, int to, int amount){
    synchronized(accounts){
        accounts.set(from, accounts.get(from)-amount);
        accounts.set(to, accounts.get(to)+amount);
    }
    System.out.println(...);
}
~~~

- 监视器：为了做到面向对象式非显式锁来保证多线程安全，而提出的一种解决方案。特性如下：
  - 监视器是只含有私有字段的类
  - 每个对象关联一个锁
  - 所有方法由该锁锁定
  - 锁可以有任意多个相关联的条件

- **volatile** 关键字为实例字段的同步访问机制提供了一种免锁机制。对声明该关键字的变量的修改，对读取该变量的其他线程都可见。
  - volatile保证了数据的一致性，但没有保障原子性。
- java.util.concurrent.atomic包的很多类使用高效的机器级指令来保证操作的原子性。

~~~java
public static AtomicLong nextNumber = new AtomicLong();
long id = nextNumber.incrementAndGet();	// 原子的完成 自增并返回自增后的值
// 如果想完成更加复杂的更新，就必须使用compareAndSet方法。
// 跟踪不同线程观察到的最大值
public static AtomicLong largest = new AtomicLong();
largest.updateAndGet(x -> Math.max(x, observed));
largest.accumulateAndGet(observed, Math::max);		// 二元操作符 合并 参数和原子值
// getAndAccumulate和getAndUpdate 会返回原值
~~~

- 如果大量线程要访问相同的原子值，乐观更新就需要大量重试。可以改用LongAdder和LongAccumulator类来解决这个问题。

- 线程局部变量：ThreadLocal辅助类能为各个线程提供自己的实例。

~~~java
// SimpleDateFormat不是线程安全的
public static final SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
// 如果两个线程都执行以下操作，dateFormat使用的内部数据结构可能被并发访问破坏掉，造成结果混乱。
String dateStamp = dateFormat.format(new Date());
// 可以为每个线程构造一个实例
public static final ThreadLocal<SimpleDateFormat> dateFormat = 
    ThreadLocal.withInitial(()-> new SimpleDateFormat("yyyy-MM-dd"));
// 在一个给定线程中首次调用get()，会调用构造器中的lambda表达式。此后get方法会返回属于当前线程的实例。
String dateStamp = dateFormat.get().format(new Date());
//
int random = ThreadLocalRandom.current().nextInt(upperBound);
~~~

### 9.4 线程安全集合

- 多线程并发修改一个数据结构可能导致该数据结构被破坏。如一个线程正在调整散列表各个桶间的链接关系，然后被另一个线程抢占，并遍历该散列表，这可能会造成混乱。
- 很多线程问题可以使用队列来描述。生产者将元素插入队列，消费者从队列获取元素。使用队列可以安全地从一个线程向另一个线程传递数据。
  - 如：允许某一个特定线程访问银行内部，转账线程将转账指令对象插入一个队列，特定线程取指令并完成操作。因为只有单一线程访问银行，所以不需要同步。
- 阻塞队列(blocking queue) 在队列满+插入操作 和 队列空+移出操作的时候导致线程阻塞。
  - put和take方法在队列空或满时会阻塞队列。
  - offer、poll、peek方法替代add、remove、element操作来完成多线程任务，给出错误提示而不是抛出异常

~~~C++
// 带超时参数的offer和poll方法
boolean success = q.offer(x, 100, TimeUnit.MILLISECONDS);
Object head = q.poll(100, TimeUnit.MILLISECONDS);
// offer和poll put和take是等效的
~~~

- 阻塞队列变体：LinkedBlockingQueue、ArrayBlockingQueue、PriorityBlockingQueue、LinkedBlockingDeque、DelayQueue、TransferQueue。

- 利用线程安全数据结构完成同步操作，可以不需要显式的线程同步。

~~~java
public class BlockingQueueTest {
    private static final int FILE_QUEUE_SIZE = 10;
    private static final int SEARCH_THREADS = 100;
    private static final Path DUMMY = Path.of("");
    private static BlockingQueue<Path> queue = new ArrayBlockingQueue<>(FILE_QUEUE_SIZE);

    public static void main(String[] args) {
        String currentPath = System.getProperty("user.dir");
        String directory = currentPath + File.separator + "blockingQueueFile";
        Runnable enumerator = () -> {
            try{
                enumerate(Path.of(directory));
                queue.put(DUMMY);
            }catch (Exception e){
                e.printStackTrace();
            }
        };
        new Thread(enumerator).start();
        for(int i = 0; i < SEARCH_THREADS; i++){
            Runnable searcher = () -> {
                var done = false;
                while(!done){
                    try{
                        Path path = queue.take();
                        if(path == DUMMY){
                            queue.put(path);
                            done = true;
                        }
                        else    search(path);
                    }
                    catch(Exception e){
                        e.printStackTrace();
                    }
                }

            };
            new Thread(searcher).start();
        }
    }

    public static void enumerate(Path directory) throws IOException, InterruptedException{
        try(Stream<Path> children =  Files.list(directory)){
            var list = children.collect(Collectors.toList());
            for(Path ele : list){
                if(Files.isDirectory(ele))  enumerate(ele);
                else    queue.put(ele);
            }
        }
    }

    public static void search(Path file){
        try(var in = new Scanner(file, StandardCharsets.UTF_8)) {
            String line = in.nextLine();
            System.out.println(file + ":" + line);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
~~~

- java.util.concurrent包提供了映射、有序集和队列的高效实现：ConcurrentHashMap、ConcurrentSkipListMap、ConcurrentSkipListSet和ConcurrentLinkedQueue。运行并发访问数据结构的不同部分，size操作不一定在常量时间内完成，可能需要遍历集合。
- 这些集合会返回弱一致性的迭代器。但不会将同一个值返回两次，也不会抛出ConcurrentModificationException。
- 原子更新：

~~~java
 map.compute(word, (k, v) -> v == null ? 1 : v + 1);		// ConcurrentHashMap中不允许有null，null用来指示键不存在
 map.computeIfAbsent(word, k -> new LongAdder()).increment();		// 
// merge
map.merge(word, 1L, Long::sum);
map.merge(word, 1L, (existValue, newValue) -> existValue + newValue);
~~~

~~~java
public class CHMDemo {
    private static ConcurrentHashMap<String, Long> map = new ConcurrentHashMap<String, Long>();

    public static void process(Path file){
        try(var in = new Scanner(file)) {
            while(in.hasNext()){
                String word = in.next();
                map.merge(word, 1L, Long::sum);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Set<Path> descendants(Path rootDir) throws IOException {
        try(Stream<Path> paths = Files.walk(rootDir)){		// dfs所有文件
            return paths.collect(Collectors.toSet());
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        int processors = Runtime.getRuntime().availableProcessors();		// 可用线程数
        ExecutorService executor = Executors.newFixedThreadPool(processors);		// 固定大小线程池
        String directory = System.getProperty("user.dir");
        Path current = Path.of(directory).getParent();
        for(var p : descendants(current)){
            if(p.getFileName().toString().endsWith(".java")){
                executor.execute(() -> process(p));	// Runnable
            }
        }
        executor.shutdown();		// 停止接收新任务，原来的任务继续执行
        executor.awaitTermination(10, TimeUnit.MINUTES);		// 阻塞 10分钟或者到任务执行完
        map.forEach((k, v) -> {
            if(v >= 10){
                System.out.println("occurs " + v + " times:" + k);
            }
        });
    }
}
~~~

- 对并发散列映射(ConcurrentHashMap)的批操作：
  - java为ConcurrentHashMap提供了批操作的API，可以和别的处理该映射的线程一起执行。
  - 三种操作：search、reduce、forEach。
  - 四个版本：operationsKeys、operationsValues、operations、operationsEntries。
  - 参数化阈值：元素多于阈值则并行完成批操作，根据需要可以设置为Long.MAX_VALUE和1。

~~~java
map.search(threshold, (k, v) -> v > 1000 ? : k : null);		// 直到找到第一个非null
map.forEach(threshold, (k, v) -> System.out.println(k + " " + v));
map.forEach(threshold, (k, v) -> v > 100 ? : k + " " + v : null, System.out::println);		// 额外的转换器参数
Long count = map.reduceValues(threshold, v -> v > 1000 ? 1L : null, Long::sum);
~~~

- 并发集视图：想要一个”ConcurrentHashSet“，但类库没有提供

~~~java
Set<String> set = ConcurrentHashMap.<String>newKeySet();
//
Set<String> words = map.keySet(1L);
words.add("java");		// 新增元素时默认value是1
~~~

- CopyOnWriteArrayList是CopyOnWriteArraySet的线程安全集合。基于写时复制。
- 并行数组算法： parallel~(ˈperəˌlel)~

~~~java
Arrays.parallelSort(words, Comparator.comparing(String::length));		// 
Arrays.parallelSetAll(values, i -> i % 10); 	// i是索引
Arrays.parallelPrefix(values, (x, y) -> x * y);		// 给定结合操作的相应前缀的累加结果替换各个数组元素
~~~

- 同步包装器：将任何集合类变成线程安全的。使用锁来保护集合方法。

~~~java
List<E> syncArrayList = Collections.synchronizedList(new ArrayList<E>());
Map<K, V> syncHashMap = Collections.synchronizedMap(new HashMap<K, V>());	
~~~

- 如果希望迭代访问一个集合，同时可能存在另一个线程访问这个集合，就需要使用“客户端”锁定

~~~java
synchronized(syncHashMap){
    Iterator<K> iter = syncHashMap.keySet().iterator();
    while(iter.hasNext())	...;
}
// 因为for each的本质是一个迭代器，所以使用的时候，也得同上述代码一样。
~~~

- 除了同步包装的ArrayList性能好于CopyOnWriteArrayList之外，最好应该使用同步包中的集合。	

### 9.5 任务和线程池

- 如果有大量生命周期很短的线程需求，应该使用线程池。

- Runnable封装了一个异步运行的任务，没有参数和返回值。Callable接口是一个参数化的类型，有返回值， 可以抛出异常。
  - Thread只接收Runnable类，Executor接受Runnable和Callable。

~~~java
public interface Callable<V>{
    V call() throws Exception;
}
~~~

- Future保存异步计算的结果。Future对象的所有者在计算完成后可以获得结果。Future<V>接口方法如下

~~~java
V get();		// 阻塞调用
V get(long timeout, TimeUnit unit);		// 阻塞调用+超时抛异常
boolean isDone();
void cancel(boolean mayInterrupt);		// 取消计算，如果计算正在进行则参数为true且被中断
boolean isCancelled();
~~~

- 执行Callable的一种方法：

~~~java
Callable<Integer> task = new Callable<Integer>() {
    @Override
    public Integer call() throws Exception {
        System.out.println("call方法");
        return 1;
    }
};
FutureTask<Integer> integerFutureTask = new FutureTask<>(task);		// FutureTask同时实现了Runnable和Future接口
var t = new Thread(integerFutureTask);
t.start();
...
Integer res = integerFutureTask.get();
~~~

- 执行器(Executors)：通过很多静态工厂方法来构造线程池。返回的线程池实现了ExecutorService接口。

~~~java
// 将任务交给ExecutorService
Future<T> submit(Callable<T> task);
Future<?> submit(Runnable task);
Future<?> submit(Runnable task, T result);
// 
shutdown();
shutdownNow();
~~~

- 控制任务组：控制一组相关任务。
  - invokeAny：提交一组Callable对象，并返回某个已完成任务的结果。如：搜索+任意符合条件的结果==最快完成的结果。
  - invokeAll：阻塞并返回所有答案的Future列表。

~~~java
public class ExecutorDemo {
    public static long occurences(String word, Path file){
        try (var in = new Scanner(file)){
            int count = 0;
            while (in.hasNext()){
                if(in.next().equals(word))  count++;
            }
            return count;
        }
        catch (IOException e) {
            e.printStackTrace();
            return 0;
        }
    }

    public static Set<Path> descendants(Path rootDir) throws IOException {
        try(Stream<Path> entries = Files.walk(rootDir)){
            return entries.filter(Files::isRegularFile).collect(Collectors.toSet());
        }
    }

    public static Callable<Path> searchForTask(String word, Path path){
        return () -> {
            try (var in = new Scanner(path)){
                while(in.hasNext()){
                    if(in.next().equals(word))  return path;
                    if(Thread.currentThread().isInterrupted()){
                        System.out.println("Search in " + path + " canceled.");
                        return null;
                    }
                }
                throw new NoSuchElementException();
            }
        };
    }

    public static void main(String[] args)
            throws IOException, InterruptedException, ExecutionException {
        try{
            String start = System.getProperty("user.dir");
            String word = "final";
            Set<Path> files = descendants(Path.of(start));
            ArrayList<Callable<Long>> tasks = new ArrayList<>();
            for(Path file :files){
                Callable<Long> task = () -> occurences(word, file);
                tasks.add(task);
            }
            ExecutorService executor = Executors.newCachedThreadPool();
            Instant startTime = Instant.now();
            List<Future<Long>> futures = executor.invokeAll(tasks);
            long total = 0;
            for(var result : futures){
                total += result.get();
            }
            Instant endTime = Instant.now();
            System.out.println("Occurrences of " + word + ": " + total);
            System.out.println("Time elapsed: " +
                    Duration.between(startTime, endTime).toMillis() + " ms");

            ArrayList<Callable<Path>> searchTasks = new ArrayList<>();
            for(Path file : files){
                searchTasks.add(searchForTask(word, file));
            }
            Path found = executor.invokeAny(searchTasks);
            System.out.println(word + " occurs in: " + found);
            executor.shutdown();
        }
        finally {
            ;
        }
    }
}
~~~

- Executor、Executors和ExecutorService：

  - Executor 和 ExecutorService 接口主要的区别是：ExecutorService 接口继承了 Executor 接口，是 Executor 的子接口。

  - Executor 和 ExecutorService 接口第二个区别是：Executor 接口定义了 execute()方法用来接收一个Runnable接口的对象，而 ExecutorService 接口中的 submit()方法可以接受Runnable和Callable接口的对象。

  - Executor 和 ExecutorService 接口第三个区别是：Executor 中的 execute() 方法不返回任何结果，而 ExecutorService 中的 submit()方法可以通过一个 Future 对象返回运算结果。

  - Executor 和 ExecutorService 接口第四个区别是：除了允许客户端提交一个任务，ExecutorService 还提供用来控制线程池的方法。比如：调用 shutDown() 方法终止线程池。

  - Executors 类提供工厂方法用来创建不同类型的线程池。


- fork-join框架：适用于计算密集型任务，将任务分解并做负载均衡优化，最后join结果。
  - 工作密取：每个工作线程都用一个双端队列来完成任务，一个工作线程闲时，会从另一个双端队列队尾密取一个任务。

~~~java
public class ForkJoinTest {
    public static void main(String[] args){
        final int SIZE = 10000000;
        double[] numbers = new double[SIZE];
        for(int i = 0; i < SIZE; i++){
            numbers[i] = Math.random();
        }
        var counter = new Counter(numbers, 0, numbers.length, x -> x > 0.5);
        ForkJoinPool pool = new ForkJoinPool();
        pool.invoke(counter);
        System.out.println(counter.join());
    }
}

class Counter extends RecursiveTask<Integer> {
    public static final int THRESHOLD = 1000;
    private double[] values;
    private int from;
    private int to;
    private DoublePredicate filter;

    public Counter(double[] values, int from, int to, DoublePredicate filter){
        this.values = values;
        this.from = from;
        this.to = to;
        this.filter = filter;
    }

    @Override
    protected Integer compute() {
        if(to - from < THRESHOLD){
            int count = 0;
            for(int i = from; i < to; i++){
                if(filter.test(values[i])){
                    count++;
                }
            }
            return count;
        }
        else{
            int mid = (from + to) / 2;
            var first = new Counter(values, from, mid, filter);
            var second = new Counter(values, mid, to, filter);
            invokeAll(first, second);
            return first.join() + second.join();
        }
    }
}
~~~

### 9.6 异步计算

- Future对象的get方法会阻塞线程直至获得结果。CompletableFuture类实现了Future接口，它通过回调函数 无阻塞的在结果可用时对结果进行处理。

~~~java
var client = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder(URI.create(urlString)).GET().build();
CompletableFuture<HttpResponse<String>> f = client.sendAsync(request, BodyHandler.asString());
~~~

- 多数情况下需要自定义CompletableFuture。运行异步任务应该调用静态方法CompletableFuture.supplyAsync。

~~~java
public CompletableFuture<String> readPage(URL url){
    return CompletableFuture.supplyAsync(() -> {
        try{
            return new String(url.openStream().readAllBytes(), "UTF-8");
        }
        catch(IOException e){
            throw new UncheckedIOException(e);
        }
    }, executor);		// 默认执行器是ForkJoinPool.commonPool()返回的执行器
}
~~~

- CompletableFuture可用采用两种方式完成：得到一个结果 或者 有一个未捕获的异常。可用使用whenComplete方法处理这两种情况。

~~~java
f.whenComplete((s, t) -> {
    if(t == null) {;}
    else {;}
}); 
~~~

- CompletableFuture可以手动设置一个完成值（承诺）。

~~~java
var f = new CompletableFuture<Integer>();
// 显式设置完成值，两个任务同时计算一个答案
executor.execute(() -> {
    int n = workHard(arg);
    f.complete(n);
});
executor.execute(() -> {
    int n = workSmart(arg);
    f.complete(n);
});
// 对一个异常完成future
Throwable t = ...;
f.completeExceptionally(t);
~~~

- 组合可完成Future：将异步任务组合成一个处理管线。

~~~java
CompletableFuture<String> contents = readPage(url);
CompletableFuture<List<URL>> imageURLs = contents.thenApply(this::getImageURLs);
// thenApply
CompletableFuture<U> future.thenApply(f);
CompletableFuture<U> future.thenApplyAsync(f);
// thenCompose 将T -> CompletableFuture<U> 和 U -> CompletableFuture<V> 组合为一个 T -> CompletableFuture<V>
~~~

- handle方法可以在CompletableFuture完成时，计算一个新结果。exceptionally和completeOnTimeout也能计算一个假值。

~~~java
public class CompletableFutureTest {
    private URL urlToProcess;
    private ExecutorService executor = Executors.newCachedThreadPool();
    // 大约是 <img src = " ">
    private static final Pattern IMG_PATTERN = Pattern.compile(
            "[<]\\s*[iI][mM][gG]\\s*[^>]*[sS][rR][cC]\\s*[=]\\s*['\"]([^'\"]*)['\"][^>]*[>]");

    public CompletableFuture<String> readPage(URL url){
        return CompletableFuture.supplyAsync(() -> {
            try{
                var contents = new String(url.openStream().readAllBytes(), StandardCharsets.UTF_8);
                System.out.println("Read page from " +url);
                return contents;
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }, executor);
    }

    public List<URL> getImageURLs(String webpage){
        try{
            ArrayList<URL> urls = new ArrayList<>();
            Matcher matcher = IMG_PATTERN.matcher(webpage);
            while(matcher.find()){
                var url = new URL(urlToProcess, matcher.group(1));
                urls.add(url);
            }
            System.out.println("Found URLs: " + urls);
            return urls;
        }
        catch(IOException e){
            throw new UncheckedIOException(e);
        }
    }

    public CompletableFuture<List<BufferedImage>> getImages(List<URL> urls){
        return CompletableFuture.supplyAsync(() -> {
            try{
                var result = new ArrayList<BufferedImage>();
                for(URL url : urls){
                    result.add(ImageIO.read(url));
                    System.out.println("Loaded " + url);
                }
                return result;
            }
            catch(IOException e){
                throw new UncheckedIOException(e);
            }
        }, executor);
    }

    public void saveImages(List<BufferedImage> images){
        System.out.println("Saving " + images.size() + " images");
        try{
            for(int i = 0; i < images.size(); i++){
                // 写入路径
                String filename = "/tmp/image" + (i + 1) + ".png";
                ImageIO.write(images.get(i), "PNG", new File(filename));
            }
        }
        catch(IOException e){
            throw new UncheckedIOException(e);
        }
        executor.shutdown();
    }

    public void run(URL url){
        urlToProcess = url;
        CompletableFuture.completedFuture(url).thenComposeAsync(this::readPage, executor)
                .thenApply(this::getImageURLs).thenCompose(this::getImages).thenAccept(this::saveImages);
    }

    public static void main(String[] args) throws MalformedURLException {
        new CompletableFutureTest().run(new URL("http://horstmann.com/index.html"));
    }
}

~~~

### 9.7 进程

- 进程：Process类和ProcessBuilder类。
  - Process：在单独一个OS进程中执行一个命令，允许与标准输入、输出和错误流交互。
  - ProcessBuilder：配置Process对象。

~~~java
var builder = new ProcessBuilder("gcc", "myapp.c");			// 第一个字符串必须是可执行的命令
var builder = new ProcessBuilder("cmd.exe", "/C", "dir");		// 不是预期结果
// 更改工作目录
builder = builder.directory(path.toFile());
// 处理进程的标准输入、输出、错误流 	分别是一个管道
Process p = builder.start();
OutputStream processIn = p.getOutputStream();
InputStream processOut = p.getInputStream();
InputStream processErr = p.getErrorStream();
// 进程的输入输出与控制台关联
builder.redirectIO();
// 重定向
builder.redirectInput(...);
// 合并输出和错误流
builder.redirectErrorStream(true);
// 修改进程环境
Map<String, String> env = builder.environment();
env.put("LANG", "fr_FR");
env.remove("JAVA_HOME");
// 管道
ProcessBuilder.startPipeline(List.of(new ProcessBuilder(), new ProcessBuilder(), ...));
~~~

- 运行一个进程

~~~java
Process process = new ProcessBuilder("/bin/ls", "-l").directory(Path.of("/tmp").toFile()).start();
// 进程流的缓冲空间有限，不能写太多，要及时输出。
try(var in = new Scanner(process.getInputStream())){
    while(in.hasNextLine()){
        System.out.println(in.nextLine());
    }
}
// 等待完成
int exitCode = process.waitFor();
if(process.waitfor(delay, TimeUnit.SECONDS)){
    int res = process.exitValue();
}
else{
    process.destroyForcibly();
}
// 进程完成时会收到一个异步通知
process.onExit().thenAccept(p -> System.out.println("Exit value: " + p.exitValue()));
~~~

- 进程句柄：通过ProcessHandle接口可以获得进程的更多信息。4种获得方式：
  - Process对象p：p.toHandle()
  - long类型OS进程ID：ProcessHandle.of(id)
  - Process.current()：运行当前java虚拟机的进程的句柄
  - ProcessHandle.allProcesses()：对当前进程可见的所有OS进程流

~~~java
long pid = handle.pid();
Optional<ProcessHandle> parent = handle.parent();
Stream<ProcessHandle> children = handle.children();
Stream<ProcessHandle> descendants = handle.descendants();
// 
ProcessHandle.Info info = ProcessHandle.current().info();
// Optional<String>
System.out.println(info.command());
System.out.println(info.totalCpuDuration());
System.out.println(info.user());
// Optional<String[]>
System.out.println(info.arguments());
// ProcessHandle接口也有 isAlive、destroy等方法，可以监视或强制进程终止。
~~~



## 十、注解和反射补充

> 文自：遇见狂神说

- **注解(Annotation) **是JDK5.0引入的新技术；（编译器等程序）通过反射读取注解，实现对元数据的访问(package, class, method, field)。

> 内置注解

- @Override：标识方法重写了父类的对应方法。
- @Deprecated~(ˈdeprɪkeɪt)~：类、属性、方法、接口等废弃。
- @SuppressWarnings：抑制警告，需要参数。
  - @SuppressWarnings(value = {"unchecked", "deprecation"})    @SuppressWarnings("all")

> 元注解

- 负责注解其他注解，共4个。
- @Target：描述注解的使用范围，能用在哪。
- @Retention：描述注解的生命周期，需要在什么级别保存该注解信息（SOURCE < CLASS < **RUNTIME**）。
- @Documented：说明该注解将被包含在javadoc中。
- @Inherited~(ɪnˈherɪtɪd)~：说明子类可以继承父类中的该注解。

~~~java
@Inherited
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(value = {ElementType.FIELD, ElementType.METHOD, ElementType.CONSTRUCTOR, ElementType.TYPE})
public @interface MyAnnotation{

}
// -_-!
@MyAnnotation
public class AnnotationTest {
    @MyAnnotation
    private String name;
    @MyAnnotation
    public AnnotationTest(String name){
        this.name = name;
    }
    @MyAnnotation
    public static void test(){
        System.out.println("哎，我啥也不干，就是玩！");
    }
}
~~~

> 自定义注解

- @interface
- 如果自定义的注解里有参数，那使用的时候就得有参数。或者有默认值也行。

~~~java
@Inherited
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(value = {ElementType.FIELD, ElementType.METHOD, ElementType.CONSTRUCTOR, ElementType.TYPE})
public @interface MyAnnotation{
    // 注解参数：参数类型 + 参数名() [default xxx];
    // 如果只有一个参数，建议用value命名
    int value() default 10086;

    String name() default "";
    int age() default 18;

    String[] schools() default {"汾阳中学", "xdu"};
}
//
@MyAnnotation(value = 188, name = "type", age = 18)
public class AnnotationTest {
    @MyAnnotation(value = 199)
    private String name;
    @MyAnnotation(name = "constructor")
    public AnnotationTest(String name){
        this.name = name;
    }
    @MyAnnotation(age = 17, schools = {"a校", "b校"})
    public static void test(){
        System.out.println("哎，我啥也不干，就是玩！");
    }
}
~~~

- 动态语言：运行时才确定数据类型的语言。
- 静态语言：编译时变量的数据类型就可以确定的语言。
- 哪些类型有Class对象：各种类（外部类、局部内部类、匿名内部类、成员内部类、静态内部类）、接口、数组、枚举、annotation、基本数据类型、void。

~~~java
Class cls = Integer.TYPE;		// 通过内置类型的TYPE属性获得Class对象 		int 
~~~

- 类的主动引用：一定会发生类的初始化
  - 虚拟机启动时，先初始化main所在的类
  - new一个对象
  - 调用类的静态成员和方法
  - 对类进行反射调用
  - 子类初始化会触发父类初始化
- 类的被动引用：不会发生类的初始化
  - 当访问一个静态域时，只有真正声明这个域的类才会被初始化。通过子类引用父类的静态变量，不会导致子类初始化。
  - 通过数组定义类引用，不会触发类初始化
  - 引用常量不会触发类初始化（常量在链接阶段就存入常量池了）

- 类加载器：

~~~java
public static void main(String[] args) throws ClassNotFoundException {
    // 应用类加载器 jdk.internal.loader.ClassLoaders$AppClassLoader@3fee733d
    ClassLoader systemClassLoader = ClassLoader.getSystemClassLoader();
    System.out.println(systemClassLoader);
    // 平台类加载器 jdk.internal.loader.ClassLoaders$PlatformClassLoader@723279cf
    ClassLoader parent = systemClassLoader.getParent();
    System.out.println(parent);
    // 引导类加载器，负责java平台核心库，无法直接获得 null
    ClassLoader ancestor = parent.getParent();
    System.out.println(ancestor);
    // 自定义类 使用应用类加载器加载
    Class<?> current = Class.forName("extendsLearn.more.KaiTest01");
    ClassLoader classLoader = current.getClassLoader();
    System.out.println(classLoader);
    // 类库 使用引导类加载器
    Class<?> aClass = Class.forName("java.lang.Double");
    ClassLoader classLoader1 = aClass.getClassLoader();
    System.out.println(classLoader1);
}
~~~

> 普通方法、反射方法和关闭安全监测的反射方法性能对比

- 反射方法明显慢于普通方法，关闭安全监测后能减小性能差距。

~~~java
public class MethodTest {
    public static void normalMethod(){
        Employee employee = new Employee();
        Instant startTime = Instant.now();
        for(int i = 0; i < 1000000000; i++){
            employee.getName();
        }
        Instant endTime = Instant.now();
        System.out.println(Duration.between(startTime, endTime).toMillis() + " ms");
    }

    public static void reflectMethod()
            throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        Class<?> empClass = Class.forName("extendsLearn.Employee");
        Constructor<?> constructor = empClass.getConstructor(null);
        Employee employee = (Employee)constructor.newInstance(null);
        Method method = empClass.getMethod("getName", null);
        Instant startTime = Instant.now();
        for(int i = 0; i < 1000000000; i++){
            method.invoke(employee, null);
        }
        Instant endTime = Instant.now();
        System.out.println(Duration.between(startTime, endTime).toMillis() + " ms");
    }

    public static void refMetWithAccess()
            throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException, InstantiationException, IllegalAccessException {
        Class<?> empClass = Class.forName("extendsLearn.Employee");
        Constructor<?> constructor = empClass.getConstructor(null);
        Employee employee = (Employee)constructor.newInstance(null);
        Method method = empClass.getMethod("getName", null);
        method.setAccessible(true);
        Instant startTime = Instant.now();
        for(int i = 0; i < 1000000000; i++){
            method.invoke(employee, null);
        }
        Instant endTime = Instant.now();
        System.out.println(Duration.between(startTime, endTime).toMillis() + " ms");
    }

    public static void main(String[] args) throws ReflectiveOperationException{
        normalMethod();
        reflectMethod();
        refMetWithAccess();
    }
}
// 3 ms 	3861 ms 	1753 ms
~~~

> 泛型与反射

~~~java
public static void test(Map<String, Employee> map, List<Double> list){}

public static void main(String[] args) {
    try{
        Class<?> cls = Class.forName("extendsLearn.reflect.reflectWithGeneric");
        Method method = cls.getMethod("test", Map.class, List.class);
        for (Type paramType : method.getGenericParameterTypes()) {
            System.out.println(paramType);
        }
    }
    catch(Exception e){
        e.printStackTrace();
    }
}
// java.util.Map<java.lang.String, extendsLearn.Employee>
// java.util.List<java.lang.Double>
~~~

> 注解与反射

~~~java
public class ReflectWithAnno {
    public static void main(String[] args) {
        try {
            Class<?> cls = Class.forName("extendsLearn.reflect.Student");
            Field field = cls.getDeclaredField("name");
            kaiField annotation = field.getAnnotation(kaiField.class);
            System.out.println(annotation.colName() + ": " +  field.getName());
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }
}

@kaiTable("stu_table")
class Student{
    @kaiField(colName = "db_name", type = "varchar", length = 3)
    String name;
    @kaiField(colName = "db_age", type = "int", length = 10)
    int age;
    public Student(){

    }
    public Student(String name, int age){
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public String toString() {
        return "Student{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@interface kaiTable{
    String value();
}

@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
@interface kaiField{
    String colName();
    String type();
    int length();
}
~~~

