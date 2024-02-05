---
layout: post
title:  "Numeric Data Types and Quantization"
date:   2024-02-04 18:18:42 +0000
categories: quantization
---

# Introduction
What will be covered in this post are
1. Intro to different data types
2. How to convert among different types
3. How do they accelerate LLM

# Numeric Data Types
## Bits and Bytes
First let's start from the very beginning: bits. Computers are machines that operate with 0's and 1's. Each 0 or 1 is one `bit`. These are the very basic building blocks of every computer program.

1 higher level of abstraction is `byte`, which is 8 `bit`s.

`bits` and `bytes` are not easily readable by human beings - our brains have a very small cache that only holds 4-7 pieces of concepts at a time. So we invent primitiv types, e.g. boolean, short, int, long, float, double, etc.
{::comment}
Modern computer architecture usually supports 64-bits, which means the chips move 64 `bits` at a time with their registers.
{:/comment}

## Integers
Let's start with integers that are simpler to understand.

1 `int` = 32 `bits` or 4 `bytes`. `01` and `10` in base 2 convert to `1` and `2` in base 10.

| Integer    | n-bit Range | Note |
| -------- | ------- | ----|
| Unsigned  | [0, $2^n$ − 1] | |
| Sign-Magnitude  | [−$2^{(n-1)}$ − 1, $2^{(n-1)}$ − 1] | Both 000…00 and 100…00 represent 0|
| Two's Complement  | [−$2^{(n-1)}$, $2^{(n-1)}$ − 1] | 000…00 represents 0, 100…00 represents -$2^{(n-1)}$|

{::comment}
{% highlight cpp %}
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

## Floats
Floats are much more interesting (and complicated) than integers. How do you represend 0.12345 with `0`'s and `1`'s?

[IEEE 754](https://standards.ieee.org/ieee/754/6210/) has defined how to represent `float`s, using 16 bits, 32 bits... 256 bits. For the interest of this post, we will start with the single precision - 32 bits or `fp32`, and then go through half precision - 16 bits or `fp16`, Google Brain 16-bit float `bf16`, 2 formats of 8-bit floats `fp8 e4m3` and `fp8 e5m2`.

A floating number is equal to:

$Floating\ Point = Mantissa * Exponent$
{: style="text-align: center;"}

{::comment}
Example style
{: style="color:gray; font-size: 80%; text-align: center;"}
{:/comment}

In IEEE 754, $mantissa = (1 + fraction)$, and $exponent = 2^{exponent}$. 

Let's take a deeper look into different precisions.

### FP32
[IEEE 754](https://standards.ieee.org/ieee/754/6210/) single precision floating point number with 32 bits.


{% include numerics/float32.svg %}


<br>
- 1 bit sign
- 8 bits exponent: 30th bit is $2^8 = 128$, 29th bit = $64$ ...
- 23 bits fraction/mantissa: 22th bit = $1/2 = 0.5$, 21th bit = $1/2^2 = 0.25$, ... 
- The final number in base 10:

${(-1)}^{sign} * (1 + fraction) * 2^{exponent-127}$
{: style="text-align: center;"}

\
Example
{% include numerics/float32_example.svg %}


\
In this case
- sign = $ 0 $
- exponent = $$ (0111\ 1100)_2 = (64 + 32 + 16 + 8 + 4)_{10} = 124_{10} $$
- fraction = $$ (0100\ 0000\ ...)_2 = {1/4}_{10} = 0.25_{10} $$
- total = $ 0.15625 = {(-1)}^0 * {(1 + 0.25)} * 2^{124-127} = 1 * 1.25 * 2^{-3} $

### FP16 
[IEEE 754](https://standards.ieee.org/ieee/754/6210/) Half Precision 16-bit Float (FP16)


${(-1)}^{sign} * (1 + fraction) * 2^{exponent-15}$
{: style="text-align: center;"}

{% include numerics/float16_example.svg %}

\
$ 3.0 = {(-1)}^0 * {(1 + 0.5)} * 2^{16-15} = 1 * 1.5 * 2^1 $

### BF16
[Google](https://arxiv.org/pdf/1905.12322.pdf) Brain Float (BF16).


${(-1)}^{sign} * (1 + fraction) * 2^{exponent-127}$
{: style="text-align: center;"}

{% include numerics/bf16_example.svg %}

\
$ 40.5 = {(-1)}^0 * {(1 + 0.25 + 0.015625)} * 2^{132-127} = 1 * 1.265625 * 2^5 $

### FP8 E4M3
[Nvidia](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) FP8 E4M3
{% include numerics/fp8_e4m3.svg %}

\
$ -1.75 = {(-1)}^1 * {(1 + 0.5 + 0.25)} * 2^{7-7} = {(-1)} * 1.75 * 2^0 $

### FP8 E5M2
[Nvidia](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) FP8 E5M2 is for gradient in the backward propagation
{% include numerics/fp8_e5m2.svg %}

\
$ 0.09375 = {(-1)}^0 * {(1 + 0.5)} * 2^{11-15} = 1 * 1.5 * 2^{-4} $

### Comparison

| FP Precision | Exponent (bits) | Fraction (bits) | Total (bits) | Dynamic Range | Precision |
| -------- | ------- | ---- | ---- | ---- | ---- |
| FP32 | 8 | 23 | 32 | | |
| FP16 | 5 | 10 | 16 | | |
| BF16 | 8 | 7 | 16 | | |
| FP8 E4M3 | 4 | 3 | 8 | +/-448 | |
| FP8 E5M2 | 5 | 2 | 8 | +/-57344 | |

### Why should I care?
It turns out that as we have much more complex models, the size and computation time significantly increase. For LLMs we are talking about trillions of parameters - the data type of these parameters would significantly affect the model size and computations speed.


# Quantization
The process of converting higher precision numbers to lower precision, or changing from continuous or otherwise large set of values to a discrete set, is called __Quantization__