# -*- coding: utf-8 -*-

#导入tensorflow包
import tensorflow as tf
#定义变量
a=tf.constant([1.0,2.0],name="a")
b=tf.constant([2.0,3.0],name="b")
#定义计算方法
result=a+b
#开启会话，运行
sess=tf.Session()
a=sess.run(result)
#打印结果
print(a)