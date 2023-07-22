# C++并发编程





## 1.线程管理

### 1.1 基础

- 使用C++线程库启动线程，可以归结为构造`std::thread`对象

~~~C++
void do_some_work();
std::thread my_thread(do_some_work);

// std::thread可以用可调用类型构造，将带有函数调用符类型的实例传入std::thread类中，替换默认的构造函数。
class background_task
{
public:
  void operator()() const
  {
    do_something();
    do_something_else();
  }
};
background_task f;

std::thread my_thread(f);						// 1
std::thread my_thread((background_task()));  	// 2
std::thread my_thread{background_task()};   	// 3
std::thread my_thread([]{						// 4
  do_something();
  do_something_else();
});
~~~

~~~C++
// 这是一个 带一个参数 并返回std::thread对象的函数，该函数指针指向 没有参数并返回background_task对象的函数 
// 因为参数是临时变量而不是命名变量
std::thread my_thread(background_task());
~~~

- 启动了线程，需要明确是要等待线程结束(*加入式*)，还是让其自主运行(*分离式*)。如果`std::thread`对象销毁之前还没有做出决定，程序就会终止(`std::thread`的析构函数会调用`std::terminate()`)。因此，即便是有异常存在，也需要确保线程能够正确的*加入*(joined)或*分离*(detached)。

- 如果不等待线程，就必须保证线程结束之前，可访问的数据得有效性。
  - 这种情况很可能发生在线程还没结束，函数已经退出的时候，这时线程函数还持有函数局部变量的指针或引用。

~~~C++
struct func
{
  int& i;
  func(int& i_) : i(i_) {}
  void operator() ()
  {
    for (unsigned j=0 ; j<1000000 ; ++j)
      do_something(i);           // 1. 潜在访问隐患：悬空引用
  }
};
void oops()
{
  int some_local_state=0;
  func my_func(some_local_state);
  std::thread my_thread(my_func);
  my_thread.detach();          // 2. 不等待线程结束
}                              // 3. 新线程可能还在运行
~~~

- join()是简单粗暴的等待线程完成或不等待。调用join()的行为清理了线程相关的存储部分，这样`std::thread`**对象**将不再与已经完成的**线程**有任何关联。

- 在线程运行之后，在join()调用之前产生并抛出异常，意味着join()调用会被跳过。

~~~C++
class thread_guard
{
  std::thread& t;
public:
  explicit thread_guard(std::thread& t_):
    t(t_)
  {}
  ~thread_guard()
  {
    if(t.joinable()) // 1
    {
      t.join();      // 2
    }
  }
  thread_guard(thread_guard const&)=delete;   // 3
  thread_guard& operator=(thread_guard const&)=delete;
};
struct func;
void f()
{
  int some_local_state=0;
  func my_func(some_local_state);
  std::thread t(my_func);
  thread_guard g(t);
  do_something_in_current_thread();
}    // 4
~~~

- detach()会让线程在后台运行，这就意味着主线程不能与之产生直接交互。也就是说，不会等待这个线程结束；如果线程分离，那么就不可能有`std::thread`对象能引用它，分离线程的确在后台运行，所以分离线程不能被加入。调用`std::thread`成员函数detach()来分离一个线程之后，相应的`std::thread`**对象**就与实际执行的**线程**无关了，并且这个线程也无法加入。

- 分离线程又叫*守护线程*(daemon threads)，在Unix中指没有任何显式的用户接口，并在后台运行的线程。

### 1.2 向线程函数传参

~~~C++
void f(int i, std::string const& s);
std::thread t(f, 3, "hello");
~~~

- 参数的可访问性问题

~~~C++
// error: buffer指向本地变量，会导致引用出错 Oops
void f(int i,std::string const& s);
void oops(int some_param)
{
  char buffer[1024]; // 1
  sprintf(buffer, "%i",some_param);
  std::thread t(f,3,buffer); // 2
  t.detach();
}
// 解决 
std::thread t(f,3,std::string(buffer));
~~~

- 参数的意外复制

~~~C++
// error: 2处 本来期待一个引用data，却因为std::thread的构造函数不知道，进行了拷贝。
void update_data_for_widget(widget_id w,widget_data& data); // 1
void oops_again(widget_id w)
{
  widget_data data;
  std::thread t(update_data_for_widget,w,data); // 2
  display_status();
  t.join();
  process_widget_data(data); // 3
}
// 解决
std::thread t(update_data_for_widget,w,std::ref(data));
~~~

- 可以传递一个成员函数指针作为线程函数

~~~C++
class X
{
public:
  void do_lengthy_work(int);
};
X my_x;
int num(0);
// 需要提供一个合适的对象指针作为第一个参数
std::thread t(&X::do_lengthy_work, &my_x, num);

// 提供的函数参数可以移动，但不能拷贝。
void process_big_object(std::unique_ptr<big_object>);
std::unique_ptr<big_object> p(new big_object);
p->prepare_data(42);
std::thread t(process_big_object,std::move(p));
~~~

### 1.3 转移线程所有权

- C++标准库中有很多*资源占有*(resource-owning)类型，比如`std::ifstream`,`std::unique_ptr`还有`std::thread`都是可移动，但不可拷贝。

~~~C++
void some_function();
void some_other_function();
std::thread t1(some_function);            // 1 
std::thread t2=std::move(t1);            // 2 t2与some_function线程关联 t1现在没有关联执行线程
t1=std::thread(some_other_function);     // 3 临时对象会隐式调用移动操作
std::thread t3;                          // 4 t3现在没有关联执行线程
t3=std::move(t2);                        // 5
t1=std::move(t3);                        // 6 t1已经有了一个关联的线程，系统直接调用std::terminate()终止程序继续运行
// 线程对象被析构前，显式的等待线程完成或者分离它；进行赋值时也需要满足这些条件(不能通过赋一个新值给std::thread对象的方式来”丢弃”一个线程)。
~~~

- `std::thread`支持移动，所以其对象可以做参数和返回值

~~~C++
// std::thread支持移动的好处是可以创建thread_guard类的实例(定义见 清单2.3)，并且拥有其线程的所有权。
class scoped_thread
{
  std::thread t;
public:
  explicit scoped_thread(std::thread t_):                 // 1
    t(std::move(t_))
  {
    if(!t.joinable())                                     // 2
      throw std::logic_error(“No thread”);
  }
  ~scoped_thread()
  {
    t.join();                                            // 3
  }
  scoped_thread(scoped_thread const&)=delete;
  scoped_thread& operator=(scoped_thread const&)=delete;
};
struct func; // 定义在清单2.1中
void f()
{
  int some_local_state;
  scoped_thread t(std::thread(func(some_local_state)));    // 4
  do_something_in_current_thread();
}                                                        // 5
~~~

- 如果容器是移动敏感的(比如，标准中的`std::vector<>`)，那么移动操作同样适用于这些容器。

~~~C++
void do_work(unsigned id);
void f()
{
  std::vector<std::thread> threads;
  for(unsigned i=0; i < 20; ++i)
  {
    threads.push_back(std::thread(do_work,i)); // 产生线程
  } 
  std::for_each(threads.begin(),threads.end(),
                  std::mem_fn(&std::thread::join)); // 对每个线程调用join()
}
~~~

### 1.4 运行时决定线程数量

- ` std::thread::hardware_concurrency()`函数将返回能同时并发在一个程序中的线程数量。

### 1.5 识别线程

- 线程标识类型是`std::thread::id`。获取方式：线程对象调用get_id()或者在当前线程中调用`std::this_thread::get_id()`



## 2.线程间共享数据

### 2.1 互斥量保护共享数据

- 互斥量的使用

~~~C++
// 这两个函数中对数据的访问是互斥的：list_contains()不可能看到正在被add_to_list()修改的列表
#include <mutex>
#include <list>
#include <algorithm>
std::list<int> someList;
std::mutex myMutex;
void addToList(int newValue){
    std::unique_lock<std::mutex> mtx(myMutex);
    someList.push_back(newValue);
}
bool list_contains(int valueToFind){
    std::lock_guard<std::mutex> mtx(myMutex);
    return std::find(someList.begin(), someList.end(), value_to_find) != someList.end();
}
~~~

- **切勿将受保护数据的指针或引用传递到互斥锁作用域之外**，无论是函数返回值，还是存储在外部可见内存，亦或是以参数的形式传递到用户提供的函数中去。

~~~C++
class some_data
{
  int a;
  std::string b;
public:
  void do_something();
};
class data_wrapper
{
private:
  some_data data;
  std::mutex m;
public:
  template<typename Function>
  void process_data(Function func)
  {
    std::lock_guard<std::mutex> l(m);
    func(data);    // 1 传递“保护”数据给用户函数
  }
};
some_data* unprotected;
void malicious_function(some_data& protected_data)
{
  unprotected=&protected_data;
}
data_wrapper x;
void foo()
{
  x.process_data(malicious_function);    	// 2 传递一个恶意函数
  unprotected->do_something();    			// 3 在无保护的情况下访问保护数据
}
~~~

- 线程安全堆栈

~~~C++
// 如果遇到stack<vector<int>>pop的多线程场景，top()+拷贝 可能因为等号左边的vector空间申请失败异常而使pop()出的元素未被成功获取，丢失。
// 解决方案有三：传入一个引用/返回指向弹出值的指针/无异常抛出的拷贝构造函数或移动构造函数
#include <exception>
#include <memory>
#include <stack>
#include <mutex>
template <typename T>
class threadsafe_stack{
public:
    threadsafe_stack(){}
    ~threadsafe_stack();
    threadsafe_stack & operator = (const threadsafe_stack &)=delete;
    threadsafe_stack(const threadsafe_stack & other){
        std::lock_guard<std::mutex> lock(other.m);
        data = other.data;
    }
    void push(T val){
        std::lock_guard<std::mutex> lock(m);
        data.push(val);
    }
    std::shared_ptr<T> pop(){
        std::lock_guard<std::mutex> lock(m);
        if(data.empty()) throw empty_stack();
        std::shared_ptr<T> const res(std::make_shared<T>(data.top()));
        data.pop();
        return res;
    }
    void pop(T & val){
        std::lock_guard<std::mutex> lock(m);
        if(data.empty()) throw empty_stack();
        val = data.top();
        data.pop();
    }
    bool empty() const{
        std::lock_guard<std::mutex> lock(m);
        return data.empty();
    }
private:
    mutable std::mutex mtx;
    std::stack<T> data;
}
~~~

- 为了避免多个互斥量的死锁可能(一个线程先P(A)再P(B)，另一个先P(B)再P(A) 或者说环路等待)，我们采用std::lock()一次性锁住多个互斥量

~~~C++
class some_big_object{};
void swap(some_big_object& lhs,some_big_object& rhs);
class X
{
private:
    some_big_object some_detail;
    std::mutex m;
public:
    X(some_big_object const& sd):some_detail(sd){}
    friend void swap(X& lhs, X& rhs)
    {
        if(&lhs==&rhs)
            return;
        // std::lock要么将两个锁都锁住，要么一个都不锁
        std::lock(lhs.m,rhs.m); // 1
        // 类似于收养了这个锁，lock_guard对象除了不用对mutex参数加锁外，其余行为不变。
        std::lock_guard<std::mutex> lock_a(lhs.m,std::adopt_lock); // 2
        // 提供std::adopt_lock参数除了表示std::lock_guard对象可获取锁之外，还将锁交由std::lock_guard对象管理，
        // 而不需要std::lock_guard对象再去构建新的锁。
        std::lock_guard<std::mutex> lock_b(rhs.m,std::adopt_lock); // 3
        swap(lhs.some_detail,rhs.some_detail);
    }
};
/*
 * // std::unique_lock实例不会总与互斥量的数据类型相关；允许std::unique_lock实例不带互斥量：信息已被存储，且已被更新。
{
    // std::defer_lock 表明互斥量应保持解锁状态
    std::unique_lock<std::mutex> lock_a(lhs.m,std::defer_lock); // 1
    std::unique_lock<std::mutex> lock_b(rhs.m,std::defer_lock); // 1 std::def_lock 留下未上锁的互斥量
    std::lock(lock_a,lock_b); // 2 互斥量在这里上锁
}
{
    std::lock(lhs.m, rhs.m);
    std::lock_guard<std::mutex> lock_a(lhs.m, std::adopt_lock);
    std::lock_guard<std::mutex> lock_b(rhs.m, std::adopt_lock);
}*/
~~~

- 死锁避免：
  - 死锁避免的建议1：要么只获取一个锁（避免占有并等待条件），要么用std::lock()管理多个mutex。
  - 死锁避免的建议2：避免在持有锁时调用别人提供的代码（避免意外的锁多个mutex）
  - 死锁避免的建议3：使用固定顺序获取锁（破坏环路等待条件）
  - 死锁避免的建议4：使用锁的层次结构 当试图对一个互斥量上锁且在该互斥量已被低层锁持有时，上锁是不允许的
- 锁的层次结构：

~~~C++
// 锁的层次结构 当试图对一个互斥量上锁且在该互斥量已被低层锁持有时，上锁是不允许的。
// 虽然是运行时检测，但是它没有时间依赖性——不必去等待那些导致死锁出现的罕见条件。
class hierarchical_mutex
{
    std::mutex internal_mutex;
    unsigned long const hierarchy_value;
    unsigned long previous_hierarchy_value;
    // thread_local表示变量被线程持有 不是static automatic 或 dynamic
    static thread_local unsigned long this_thread_hierarchy_value;  // 1
    // lock失败会抛出异常而不是等待
    void check_for_hierarchy_violation()
    {
        if(this_thread_hierarchy_value <= hierarchy_value)  // 2
        {
            throw std::logic_error("mutex hierarchy violated");
        }
    }
    void update_hierarchy_value()
    {
        previous_hierarchy_value=this_thread_hierarchy_value;  // 3
        this_thread_hierarchy_value=hierarchy_value;
    }
public:
    explicit hierarchical_mutex(unsigned long value):
            hierarchy_value(value),
            previous_hierarchy_value(0)
    {}
    void lock()
    {
        check_for_hierarchy_violation();
        internal_mutex.lock();  // 4
        update_hierarchy_value();  // 5
    }
    void unlock()
    {
        this_thread_hierarchy_value=previous_hierarchy_value;  // 6
        internal_mutex.unlock();
    }
    bool try_lock()
    {
        check_for_hierarchy_violation();
        if(!internal_mutex.try_lock())  // 7
            return false;
        update_hierarchy_value();
        return true;
    }
};
thread_local unsigned long hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);  // 8


hierarchical_mutex high_level_mutex(10000);
hierarchical_mutex mid_level_mutex(5000);
hierarchical_mutex low_level_mutex(100);
void high_level_stuff(int some_param);
void low_level_stuff(int some_param);
int do_mid_level_stuff();
int mid_level_func()
{
    std::lock_guard<hierarchical_mutex> lk_mid(mid_level_mutex);
    return do_mid_level_stuff();
}
void high_level_func()
{
    std::lock_guard<hierarchical_mutex> lk_high(high_level_mutex);
    high_level_stuff(mid_level_func());
}
void low_level_func(){
    std::lock_guard<hierarchical_mutex> lk(low_level_mutex);
    low_level_stuff(mid_level_func());
}
void thread_a()
{
    high_level_func();
}
void thread_b()
{
    low_level_func();
}
~~~

- unique_lock

~~~C++
// std::unique_lock是可移动，但不可赋值的类型
// std::unique_lock可以提前释放
std::unique_lock<std::mutex> get_lock()
{
    extern std::mutex some_mutex;
    std::unique_lock<std::mutex> lk(some_mutex);
    prepare_data();
    return lk;  // 1
}
void process_data()
{
    std::unique_lock<std::mutex> lk(get_lock());  // 2
    do_something();
}

// 减少锁持有时间
class Y
{
private:
    int some_detail;
    mutable std::mutex m;
    int get_detail() const
    {
        std::lock_guard<std::mutex> lock_a(m);  // 1
        return some_detail;
    }
public:
    Y(int sd):some_detail(sd){}
    friend bool operator==(Y const& lhs, Y const& rhs)
    {
        if(&lhs==&rhs)
            return true;
        // 把内存访问和比较操作分开 只锁内存访问
        // 但是 分开就意味着 访存到比较之间的数据修改并不会被锁
        int const lhs_value=lhs.get_detail();  // 2
        int const rhs_value=rhs.get_detail();  // 3
        return lhs_value==rhs_value;  // 4
    }
};
~~~

### 2.2 保护共享数据的替代设施

- 保护共享数据的初始化过程

~~~C++
// std::once_flag和std::call_once
std::once_flag resource_flag;  // 1
void init_resource()
{
    resource_ptr.reset(new some_resource);
}
void bar()
{
    std::call_once(resource_flag,init_resource);  // 可以完整的进行一次初始化
    resource_ptr->do_something();
}
// 初始化及定义完全在一个线程中发生，并且没有其他线程可在初始化完成前对其进行处理，条件竞争终止于初始化阶段
// std::call_once的替代方案
class my_class{};

my_class& get_my_class_instance()
{
  static my_class instance;  // 线程安全的初始化过程
  return instance;
}
~~~

- 共享锁

~~~C++
#include <boost/thread/shared_mutex.hpp>
// 读者写者锁 写时独占 读时共享
class dns_entry{};
class dns_cache{
    std::map<std::string, dns_entry> entries;
    mutable boost::shared_mutex smtx;
public:
    dns_entry find_entry(std::string const& domain) const {
        std::lock_guard<boost::shared_mutex> shared_lock(smtx);
        std::map<std::string,dns_entry>::const_iterator const it= entries.find(domain);
        return (it==entries.end())?dns_entry():it->second;
    }

    void update_or_add_entry(std::string const& domain,
                             dns_entry const& dns_details)
    {
        std::lock_guard<boost::shared_mutex> lk(smtx);  // 2
        entries[domain]=dns_details;
    }
};
~~~

- 嵌套锁 std::recursive_mutex



## 3.同步并发操作

### 3.1 等待一个事件或其他条件

- 条件变量

~~~C++
// std::condition_variable和std::condition_variable_any
// 区别：前者仅限于与std::mutex一起工作，而后者可以和任何满足最低标准的互斥量一起工作
class Data{};
std::mutex mtx;
std::queue<Data> data_queue;
std::condition_variable data_cond;
bool more_data_to_prepare();
Data produce_data();
void consume_data(Data data);
bool is_last_data(Data data);
void producer(){
    while(more_data_to_prepare()){
        Data data = produce_data();
        std::lock_guard<std::mutex> lk(mtx);
        data_queue.push(data);
        data_cond.notify_one();
    }
}
void consumer(){
    while(true){
        std::unique_lock<std::mutex> lk(mtx);
        // 条件不满足  则解锁互斥量 线程阻塞或等待
        // 当准备数据的线程调用notify_one()通知条件变量时，处理数据的线程从睡眠状态中苏醒，重新获取互斥锁，并且对条件再次检查
        // 条件满足   从wait()返回并继续持有锁。
        data_cond.wait(lk, [](){return !data_queue.empty();});
        Data data = data_queue.front();
        data_queue.pop();
        lk.unlock();
        consume_data(data);
        if(is_last_data(data))     break;
    }
}
~~~

- 条件变量构造线程安全队列

~~~C++
template<typename T>
class threadsafe_queue
{
private:
  mutable std::mutex mut;  // 1 互斥量必须是可变的 
  std::queue<T> data_queue;
  std::condition_variable data_cond;
public:
  threadsafe_queue()
  {}
  threadsafe_queue(threadsafe_queue const& other)
  {
    std::lock_guard<std::mutex> lk(other.mut);
    data_queue=other.data_queue;
  }
  void push(T new_value)
  {
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(new_value);
    data_cond.notify_one();
  }
  void wait_and_pop(T& value)
  {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk,[this]{return !data_queue.empty();});
    value=data_queue.front();
    data_queue.pop();
  }
  std::shared_ptr<T> wait_and_pop()
  {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk,[this]{return !data_queue.empty();});
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }
  bool try_pop(T& value)
  {
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty())
      return false;
    value=data_queue.front();
    data_queue.pop();
    return true;
  }
  std::shared_ptr<T> try_pop()
  {
    std::lock_guard<std::mutex> lk(mut);
    if(data_queue.empty())
      return std::shared_ptr<T>();
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }
  bool empty() const
  {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }
};
~~~

### 3.2 使用期望等待一次性事件

#### 3.2.1 std::async

- 使用`std::async`启动一个异步任务，返回一个`std::future`对象，这个对象持有最终计算出来的结果。当你需要这个值时，你只需要调用这个对象的get()成员函数；并且会阻塞线程直到“期望”状态为就绪为止；

~~~C++
#include<future>
int getAnswer();
void do_other_stuff();
int main(){
   std::future<int> theAnswer = std::async(getAnswer); 
   do_other_stuff();
   // 在这里阻塞至线程就绪
   std::cout<<"The answer is "<<theAnswer.get()<<std::endl;
}
// 同std::bind/std::thread
#include <string>
struct X
{
  void foo(int,std::string const&);
  std::string bar(std::string const&);
};
X x;
auto f1=std::async(&X::foo,&x,42,"hello");  // 调用p->foo(42, "hello")，p是指向x的指针
auto f2=std::async(&X::bar,x,"goodbye");  // 调用tmpx.bar("goodbye")， tmpx是x的拷贝副本

X baz(X&);
std::async(baz,std::ref(x));  // 调用baz(x)

struct Y
{
  double operator()(double);
};
Y y;
auto f3=std::async(Y(),3.141);  // 调用tmpy(3.141)，tmpy通过Y的移动构造函数得到
auto f4=std::async(std::ref(y),2.718);  // 调用y(2.718)
~~~

- `std::async`可选的额外参数` std::launch::async`/`std::launch::deferred`。默认前者。
  - `std::launch::deferred`表示函数调用被延迟到`wait()`或`get()`函数调用时才执行。
  - `std::launch::async` 表明函数必须在其所在的独立线程上执行

~~~C++
auto f6 = std::async(std::launch::async, Y(), 1.2);	// 在新线程上执行
auto f7 = std::async(std::launch::deferred, baz, std::ref(x)); // 在wait()或get()调用时执行
auto f8=std::async(
              std::launch::deferred | std::launch::async,
              baz,std::ref(x));  // 实现选择执行方式
f7.wait();  //  调用延迟函数
~~~

#### 3.2.2 std::packaged_task

- `std::packaged_task<>`对一个函数或可调用对象，绑定一个期望。当`std::packaged_task<>` 对象被调用，它就会调用相关函数或可调用对象，将期望状态置为就绪，返回值也会被存储为相关数据。

- `std::packaged_task<>`的模板参数是一个函数签名，类型可以不完全匹配。

~~~C++
// std::packaged_task<>的偏特化
template<>
class packaged_task<std::string(std::vector<char>*,int)>
{
public:
  template<typename Callable>
  explicit packaged_task(Callable&& f);
  std::future<std::string> get_future();
  void operator()(std::vector<char>*,int);
};
// 这里的std::packaged_task对象是一个可调用对象，并且它可以包含在一个std::function对象中。
// 传递到std::thread对象中，就可作为线程函数；
// 传递另一个函数中，就作为可调用对象，或可以直接进行调用。
// 当std::packaged_task作为一个函数调用时，可为函数调用操作符提供所需的参数，并且返回值作为异步结果存储在std::future，可通过get_future()获取。

std::mutex m;
std::deque<std::packaged_task<void()> > tasks;
bool gui_shutdown_message_received();
void get_and_process_gui_message();
void gui_thread()  // 1
{
  while(!gui_shutdown_message_received())  // 2
  {
    get_and_process_gui_message();  // 3
    std::packaged_task<void()> task;
    {
      std::lock_guard<std::mutex> lk(m);
      if(tasks.empty())  // 4
        continue;
      task=std::move(tasks.front());  // 5
      tasks.pop_front();
    }
    task();  // 6
  }
}
std::thread gui_bg_thread(gui_thread);
template<typename Func>
std::future<void> post_task_for_gui_thread(Func f)
{
  std::packaged_task<void()> task(f);  // 7
  std::future<void> res=task.get_future();  // 8
  std::lock_guard<std::mutex> lk(m);  // 9
  tasks.push_back(std::move(task));  // 10
  return res;
}
~~~

#### 3.2.3 std::promises

- `std::promise<T>`提供设定值的方式(类型为T)，这个类型会和`std::future<T>` 对象相关联。一对`std::promise/std::future`会为这种方式提供一个可行的机制；在期望上可以阻塞等待线程，同时，提供数据的线程可以使用组合中的“承诺”来对相关值进行设置，以及将“期望”的状态置为“就绪”。

~~~C++
// 当“承诺”的值已经设置完毕(使用set_value()成员函数)，对应“期望”的状态变为“就绪”，并且可用于检索已存储的值。
void process_connections(connection_set& connections)
{
  while(!done(connections))  // 1
  {
    for(connection_iterator  // 2
            connection=connections.begin(),end=connections.end();
          connection!=end;
          ++connection)
    {
      if(connection->has_incoming_data())  // 3
      {
        data_packet data=connection->incoming();
        std::promise<payload_type>& p = connection->get_promise(data.id);  // 4
        p.set_value(data.payload);
      }
      if(connection->has_outgoing_data())  // 5
      {
        outgoing_packet data = connection->top_of_outgoing_queue();
        connection->send(data.payload);
        data.promise.set_value(true);  // 6
      }
    }
  }
}
~~~

- 为期望存储异常

~~~C++ 
double square_root(double x){
  if(x < 0){
    throw std::out_of_range("x<0");
  }
  return sqrt(x);
}
void learnFutureException(){
  // 异常在函数被调用时存储到期望中
  std::future<double> f = std::async(square_root, -1);
  // get()时抛出
  double y = f.get();
  // 主动存储异常
  extern std::promise<double> some_promise;
  try{
    some_promise.set_value(1.14514);
  }
  catch(...){
    // some_promise.set_exception(std::current_exception());
    some_promise.set_exception(std::make_exception_ptr(std::logic_error("foo ")));
  }
}
~~~

- 多个线程的等待

~~~C++ 
// std::future独享同步结果的所有权且一次性的获得数据。（只有一个线程能拿到结果）
void learnSharedFuture(){
  std::promise<int> p;
  std::future<int> f(p.get_future());
  assert(f.valid());
  std::shared_future<int> sf(std::move(f));
  assert(!f.valid());
  assert(sf.valid());
  // 直接share() 自动转移所有权
  std::promise<double> p2;
  auto sf2 = p2.get_future().share();
}
~~~

### 3.3 限定等待时间

- 

~~~C++
#include <chrono>
int some_task();
void do_something_with(int);
// 时延超时函数以_for为后缀 绝对超时函数以_until为后缀
void learnTime(){
    // 当前时间点
    auto time_point_now = std::chrono::system_clock::now();
    // 显示转换
    std::chrono::milliseconds ms(54802);
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(ms);
    // 时延
    std::future<int> f = std::async(some_task);
    if(f.wait_for(std::chrono::milliseconds(35)) == std::future_status::ready)
        do_something_with(f.get());
    std::this_thread::sleep_for(ms);
    // 时间节点
    auto start = std::chrono::high_resolution_clock::now();
    do_something_with(1);
    auto stop = std::chrono::high_resolution_clock::now();
//    std::cout << "do_something_with took " << std::chrono::duration<double, std::chrono::seconds>(stop-start).count()
//    << "seconds" << std::endl;
}

//
bool done = false;
std::mutex m;
std::condition_variable cv;
bool wait_loop(){
    auto const timeout = std::chrono::steady_clock::now()+std::chrono::milliseconds (500);
    std::unique_lock<std::mutex> lk(m);
    while(!done){
        if(cv.wait_until(lk, timeout) == std::cv_status::timeout){
            break;
        }
    }
    return done;
}
~~~

### 3.4 使用同步操作简化代码

- std::async启动多个线程，并行完成任务

~~~C++
template<typename T>
std::list<T> parallel_quick_sort(std::list<T> input){
    if(input.empty())   return input;
    std::list<T> result;
    result.splice(result.begin(), input, input.begin());
    T const & pivot = *result.begin();
    auto pivotIdx = std::partition(input.begin(), input.end(), [&](T const & t){return t<pivot;});
    std::list<T> lower_part;
    lower_part.splice(lower_part.end(), input, input.begin(), pivotIdx);

    std::future<std::list<T>> future = std::async(&parallel_quick_sort<T>, std::move(lower_part));
    std::list<T> lp(future.get());
    std::list<T> hp(parallel_quick_sort(std::move(input)));

    result.splice(result.begin(), lp);
    result.splice(result.end(), hp);
    return result;
}
~~~

- spawn_task

~~~C++
template<typename F, typename A>
std::future< typename std::result_of<F(A&&)>::type > spawn_task(F&&f, A&& a){
    typedef typename std::result_of<F(A&&)>::type result_type;
    std::packaged_task<result_type(A&&)> task(std::forward<F>(f));
    std::future<result_type> res(task.get_future());
    std::thread t(std::move(task), std::forward<A>(a));
    t.detach();
    return res;
}
~~~

- C++实现 消息传递式 同步



## 4.C++内存序和原子操作

- 原子操作

~~~C++
void learnAtomic(){
    // 设置和清除两种状态 总是被初始化为清除
    std::atomic_flag f = ATOMIC_FLAG_INIT;
    // 默认是std::memory_order_seq_cst;
    f.clear(std::memory_order_release);
    bool x = f.test_and_set();

    //
    std::atomic<bool> b(true);
    bool y = b.load(std::memory_order_acquire);
    b.store(true, std::memory_order_release);
    y = b.exchange(false, std::memory_order_acq_rel);

    if(b.is_lock_free())
        std::cout << "b.is_lock_free()" << std::endl;
}

// 自旋互斥锁
class spinlock_mutex{
private:
    std::atomic_flag f;
public:
    spinlock_mutex(): f(ATOMIC_FLAG_INIT){}

    void lock(){
        while(f.test_and_set(std::memory_order_acquire));
    }

    void unlock(){
        f.clear(std::memory_order_release);
    }
};

void learnCompareAndExchange(){
    bool expected = false;
    std::atomic<bool> b{};
    // 当前值和预期值一样的时候 也可能会失败（伪失败 spurious failure），所以用循环来完成比较并交换操作
    while(!b.compare_exchange_weak(expected, true) && !expected);
}

// std::atomic<T*> 指针运算
class Foo{};
void learnAtomicTStar(){
    Foo some_array[5];
    std::atomic<Foo*> p(some_array);
    Foo* x=p.fetch_add(2);  // p加2，并返回原始值
    assert(x==some_array);
    assert(p.load()==&some_array[2]);
    x=(p-=1);  // p减1，并返回原始值
    assert(x==&some_array[1]);
    assert(p.load()==&some_array[1]);
}
~~~

- 六种内存序：详见C_Cpp随笔。

- 栅栏
  - std::atomic_thread_fence：同步线程间的内存访问
  - std::atomic_signal_fence：线程内信号间的同步


~~~C++
std::atomic<bool> x,y;
std::atomic<int> z;
// 栅栏就是同步点
void write_x_then_y()
{
  x.store(true,std::memory_order_relaxed);  // 1
  std::atomic_thread_fence(std::memory_order_release);  // 2
  y.store(true,std::memory_order_relaxed);  // 3
}
void read_y_then_x()
{
  while(!y.load(std::memory_order_relaxed));  // 4
  std::atomic_thread_fence(std::memory_order_acquire);  // 5
  if(x.load(std::memory_order_relaxed))  // 6
    ++z;
}
~~~



## 5.基于锁的并发数据结构设计

- 线程安全：多个线程可以**并发**的访问这个数据结构，线程可对这个数据结构做**相同或不同的操作**，并且每一个线程都能在自己的自治域中看到该数据结构。且在多线程环境下，**无数据丢失和损毁**，所有的数据需要维持原样，且**无条件竞争**。这样的数据结构，称之为“线程安全”的数据结构。
- serialization：线程轮流访问被保护的数据，即串行访问。
- 提高并发能力的总体思路：减少保护区域、减少序列化操作。
- 提高并发的思路：
  - 锁的范围中的操作，是否允许在所外执行？
  - 数据结构中不同的区域是否能被不同的互斥量所保护？
  - 所有操作都需要同级互斥量保护吗？
  - 能否对数据结构进行简单的修改，以增加并发访问的概率，且不影响操作语义？

- 是否安全，考虑如下几点：
  - 对数据结构的修改操作是否进行了加锁保护
  - 操作之间的竞争是否会造成数据异常
  - 操作是否抛出异常，以及异常抛出是否破坏数据结构
    - 如上锁异常、解锁异常
    - 操作系统原因导致的操作异常
  - 死锁
  - 
