

# C++ 基础知识

1. 32位编译器：**32位系统下指针占用4字节**
       `char`：1个字节
       `char*`（即指针变量）: 4个字节（32位的寻址空间是2^32, 即32个bit，也就是4个字节。同理64位编译器）
       `short int` : 2个字节
       `int`： 4个字节
       `unsigned int`: 4个字节
       `float`: 4个字节
       `double`:  8个字节
       `long`:  4个字节
       `long long`: 8个字节
       `unsigned long`: 4个字节
   64位编译器：**64位系统下指针占用8字节**
      `char` ：1个字节
      `char`*(即指针变量): 8个字节
      `short int`: 2个字节
      `int`： 4个字节
      `unsigned int`: 4个字节
      `float`: 4个字节
      `double`:  8个字节
      `long`:  8个字节
      `long long`: 8个字节
      `unsigned long`: 8个字节

2. 宏定义：
```c++
   #include <stdio.h>
   #define a 10 
   
   void foo();  
   int main(){ 
      printf("%d..", a); 
      foo(); 
      printf("%d", a); 
   } 
   void foo(){ 
      #undef a 
      #define a 50 
   }
```

输出：10..10

宏定义是**在编译器预处理阶段中就进行替换**了，替换成什么只与define和undefine的**位置有关系**，与它们**在哪个函数中无关**。

以本题为例：#define a 10 到 #undef a之间的代码在预处理阶段就将a全部换为10，#define a 50后面的代码会将a替换为50。

如果没有#define a 50后面再使用a，编译器就会报错了。   



3. 以32位C++程序，请计算sizeof的值______

   ```c++
   void Func ( char str[100] ) { sizeof( str ) = ? }
   void*p = malloc( 100 ); sizeof( p ) = ?;
   ```

   数组做函数参数，会退化成指针 故4;

   void*p = malloc( 100 ); p是 空指针类型 ，32位机下占4个字节，64位机下占8个字节；sizeof(p)=4

4. 箭头运算符**->**用法：

   1. 相当于把解引用和成员访问符两个操作符结合在一起，换句话说，`p->func()`和`(*p).func()`所表示的意思一样。

      ```c++
      class A{
      	public:
        func();
      }
      int main(){
        A *p = new A();
        *p.a(); //或者使用p->a，二者等价，且更加简洁
      }
      ```
      
   2. 指向结构体成员运算符
     ```c++
       #include<stdio.h>
       // 结构体的声明与定义
       struct{
        char name[10];
        int age;
        char sex;
       }person;
       void main(){
        int i;
        // 此处就是指向结构体成员运算符（->）的用法
        i = person->age; // 提取结构体成员变量age的值，并赋值给变量i
       }
     ```
   
5. i++与++i的区别:

   1. 首先，单独拿出来说++i和i++，意思都是一样的，就是i=i+1。

   2. 如果当做运算符来说，就是a=i++或者a=++i这样的形式。情况就不一样了。

      先说a=i++，这个运算的意思是先把i的值赋予a，然后在执行i=i+1；

      而a=++i，这个的意思是先执行i=i+1，然后在把i的值赋予a；

      举个例子来说，如果一开始i=4。

      那么执行a=i++这条语句之后，a=4，i=5；

      那么执行a=++i这条语句之后，i=5，a=5；

      同理，i--和--i的用法也是一样的。

      - 对于内置数据类型，两者差别不大；
      - 对于自定义数据类型（如类），++i 返回对象的引用，而 i++ 返回对象的值，导致较大的复制开销，因此效率低。

      

6. **值传递，指针传递，引用传递**

   1. 值传递：在堆栈中开辟了内存空间以存放由主调函数放进来的实参的值，从而成为了实参的一个副本。值传递的特点是被调函数对形式参数的任何操作都是作为局部变量进行，不会影响主调函数的实参变量的值。

      ```cpp
      void change1 (int n){
          n++;
      }
      
      int main()
      {
          int num = 1;
          change1(num);
          cout << num << endl;       // 1
      }
      ```

   2. 引用传递: 被调函数的形式参数虽然也作为局部变量在堆栈中开辟了内存空间，但是这时存放的是由主调函数放进来的实参变量的地址。被调函数对形参的任何操作都被处理成间接寻址，即通过堆栈中存放的地址访问主调函数中的实参变量。正因为如此，被调函数对形参做的任何操作都影响了主调函数中的实参变量。

      ```cpp
      void change3 (int& n){
          n++;
      }
      
      int main()
      {
          int num = 1;
          change3(num);
          cout << num << endl;       // 2
      }
      ```

   3. 指针传递：形参为指向实参地址的指针，当对形参的指向操作时，就相当于对实参本身进行的操作。指针传递本质上是值传递，只不过传递的值是实参的地址，所以可以据此找到实参并对其操作。

      ```cpp
      void change2 (int* n){
          *n = *n + 1;
      }
      
      int main()
      {
          int num = 1;
          change2(&num);
          cout << num << endl;       // 2
      }
      ```

4. 左移和右移运算符 (> > 和 < <)
   1. 左移：代表把1的二进制表示左移30位，左移一位相当于乘以2
   2. 右移：相当于除以2

8. Struct与class的区别：
   1. struct默认权限为public
   2. class默认权限为private
		
		```cpp
		struct C1{
		  int m1;  // 在这里没有指定public,private,protected，则默认为public
		}
		
		class C1{
		  int m2;	 // 在这里没有指定public,private,protected，则默认为private
		}

9. 构造函数与析构函数

   ```cpp
   class Person
   {
   public:
   	//构造函数
   	Person()
   	{
   		cout << "Person的构造函数调用" << endl;
   	}
   	//析构函数
   	~Person()
   	{
   		cout << "Person的析构函数调用" << endl;
   	}
   
   };
   
   void test01()
   {
   	Person p;
   }
   
   int main() {
   	
   	test01();
   }
   ```

   10. 构造函数类型和调用：

       两种分类方式：

       ​	按参数分为： 有参构造和无参构造

       ​	按类型分为： 普通构造和拷贝构造

       三种调用方式：

       ​	括号法

       ​	显示法

       ​	隐式转换法

       ```cpp
       class Person {
       public:
       	//无参（默认）构造函数
       	Person() {
       		cout << "无参构造函数!" << endl;
       	}
       	//有参构造函数
       	Person(int a) {
       		age = a;
       		cout << "有参构造函数!" << endl;
       	}
       	//拷贝构造函数
       	Person(const Person& p) {
       		age = p.age;
       		cout << "拷贝构造函数!" << endl;
       	}
       	//析构函数
       	~Person() {
       		cout << "析构函数!" << endl;
       	}
       public:
       	int age;
       };
       
       //2、构造函数的调用
       //调用无参构造函数
       void test01() {
       	Person p; //调用无参构造函数
       }
       
       //调用有参的构造函数
       int main() {
         
       	//2.1  括号法，常用
       	Person p1(10);
       	//注意1：调用无参构造函数不能加括号，如果加了编译器认为这是一个函数声明
       	Person p2();
       
       	//2.2 显式法
       	Person p2 = Person(10); 
       	Person p3 = Person(p2);
       	Person(10)单独写就是匿名对象  当前行结束之后，马上析构
       
       	//2.3 隐式转换法
       	Person p4 = 10; // Person p4 = Person(10); 
       	Person p5 = p4; // Person p5 = Person(p4); 
       
       	//注意2：不能利用 拷贝构造函数 初始化匿名对象 编译器认为是对象声明
       	//Person p5(p4);
       }
       ```

       11. 构造函数调用规则

           默认情况下，c++编译器至少给一个类添加3个函数

           1．默认构造函数(无参，函数体为空)

           2．默认析构函数(无参，函数体为空)

           3．默认拷贝构造函数，对属性进行值拷贝

           **构造函数调用规则如下**：

           * 如果用户定义有参构造函数，c++不在提供默认无参构造，但是会提供默认拷贝构造


           * 如果用户定义拷贝构造函数，c++不会再提供其他构造函数