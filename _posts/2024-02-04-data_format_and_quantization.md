---
layout: post
title:  "Numeric Data Types and Quantization"
date:   2024-02-04 18:18:42 +0000
categories: quantization
---

## Introduction
What will be coverted in this post are
1. Intro to different data types
2. How to convert among different types
3. How do they accelerate LLM

## Section 1: Data Types
# Bits and Bytes
First let's start from the very beginning: bits. Computers are machines that operate with 0's and 1's. Each 0 or 1 is one `bit`. These are the very basic building blocks of every computer program.

1 higher level of abstraction is `byte`, which is 8 `bit`s.

`bits` and `bytes` are not easily readable by human beings - our brains have a very small cache that only holds 4-7 pieces of concepts at a time. So we invent primitiv types, e.g. boolean, short, int, long, float, double, etc.
{::comment}
Modern computer architecture usually supports 64-bits, which means the chips move 64 `bits` at a time with their registers.
{:/comment}

# Integers
Let's start with integers that are simpler to understand.

1 `int` = 32 `bits` or 4 `bytes`. `01` and `10` in base 2 convert to `1` and `2` in base 10.

| Integer    | n-bit Range | Note |
| -------- | ------- | ----|
| Unsigned  | [0, $$ 2^n $$ − 1] | |
| Sign-Magnitude  | [−$2^{(n-1)}$ − 1, $2^{(n-1)}$ − 1] | Both 000…00 and 100…00 represent 0|
| Two's Complement  | [−$2^{(n-1)}$, $2^{(n-1)}$ − 1] | 000…00 represents 0, 100…00 represents -$2^{(n-1)}$|

{::comment}
{% highlight c++ %}
int8_t
int16_t
int32_t
uint8_t
uint16_t
uint32_t
{% endhighlight %}
{:/comment}

Base on the table, here are the range of common integer types supported in C++

| Integer Type | Range |
| -------- | ------- |
| uint8_t | [0, 255] |
| uint16_t | [0, 65,535] |
| uint32_t | [0, 4,294,967,295] |

Check out the [Standard C++ data types][cpp-types].

[cpp-types]: https://en.cppreference.com/w/cpp/header/cstdint
