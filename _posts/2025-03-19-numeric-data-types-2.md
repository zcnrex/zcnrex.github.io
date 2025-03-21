---
layout: post
title:  "AWQ Dequantization Kernel Details"
date:   2025-03-19 18:18:42 +0000
tags: quantization
---

Let's first take a look at a core function in [AWQ repository](https://github.com/mit-han-lab/llm-awq/tree/main).

{% highlight c linenos %}
__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    /********************************** Initialization **********************************/
    uint4 result;

    uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);
    /********************************** End of Initialization **********************************/

    /********************************** Masking **********************************/
    static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
    static constexpr uint32_t TOP_MASK              = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    const uint32_t top_i4s = i4s >> 8;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[0])
                    : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[1])
                    : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[2])
                    : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                    : "=r"(h[3])
                    : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    /********************************** End of Masking **********************************/

    /********************************** Conversion **********************************/
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    static constexpr uint32_t NEG_64 = 0xd400d400;

    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    /********************************** End of Conversion **********************************/

    return result;
}
{% endhighlight %}

\
This short function (only ~40 lines with good formatting and comments) converts 8 `uint4_t` to 8 `fp16`, both represented using `uint32_t`. (There's no `uint4_t`, but here we use this notation to distinguish it from `int4`) The speed of this function has non-trivial impact to the overall speed of inferencing LLMs quantized with AWQ.


There are many details in this short program. Let's break them down piece by piece.

### Section 1: Initialization
This section initializes and prepares all the variables used in this function.

{% highlight c %}
uint4 result;
uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);
{% endhighlight %}

- `int4`: [a cuda built-in type](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#char-short-int-long-longlong-float-double) that represents 4 32-bit integers. It's size is 16 bytes (4 * 4 bytes).
- `reinterpret_cast` here converts the original data into `uint32_t` so it's easy to manipulate in assembly.

### Section 2: Masking & Magic Number
This section prepares the `uint4_t` to an intermediate form before finally converting to the right value in `fp16`.
#### Assembly 😨
{% highlight c %}
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
{% endhighlight %}

Let's look at the keywords:
- `asm`: a keyword used to include assembly language instructions directly within C++ code.
- `volatile`: indicate to the compiler that a variable may be changed by external factors that are beyond the compiler's control
- `lop3`: a [PTX (Parallel Thread Execution)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) instruction. "PTX exposes the GPU as a data-parallel computing device.". Specifically, `lop3` performs bit manipulation and has a very interesting design.
- `.b32`: a data type specifier that indicates a 32-bit (4-byte) basic data type.

#### Constants
{% highlight c %}
// (%1 & %2) | %3
static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
static constexpr uint32_t TOP_MASK              = 0x00f000f0;
// 0110 0100 0000 0000 = 2 ^ (25 - 15) = 1024
static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;
{% endhighlight %}

#### An intelligent hardware design
```
immLut = (0xf0 & 0xcc) | 0xaa;
```
This makes `lop3` very versatile. It can support up to 256 different operations (`0x00 - 0xFF`) using this simple mechanism.

```
ta = 0xF0;
tb = 0xCC;
tc = 0xAA;

immLut = F(ta, tb, tc);
```
Examples:
```
If F = (a & b & c);
immLut = 0xF0 & 0xCC & 0xAA = 0x80

If F = (a | b | c);
immLut = 0xF0 | 0xCC | 0xAA = 0xFE

If F = (a & b & ~c);
immLut = 0xF0 & 0xCC & (~0xAA) = 0x40

If F = ((a & b | c) ^ a);
immLut = (0xF0 & 0xCC | 0xAA) ^ 0xF0 = 0x1A
```

More details see the [official doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3)

#### A trick
`|` the magic number is equivalent to multiply it. Using `lop3` packs 2 instructions into 1: 1) mask 2) multiply.

### Section 3: Conversion
#### Constants
{% highlight c %}
// 0110 0100 0000 0000 = 2 ^ (25 - 15) = 1024
static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
// 0010 1100 0000 0000 = 2 ^ (11 - 15) = 1 / 16
static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
// 1101 0100 0000 0000 = (-1) * 2 ^ (20 - 15) = 64
static constexpr uint32_t NEG_64 = 0xd400d400;
{% endhighlight %}

#### Assembly 😨
{% highlight c %}
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
{% endhighlight %}

- `.f16x2`: 2 `fp16` numbers. It's only supported by [sm_53](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) above - [Pascal GPU Architecture](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or later.
- `fma`: fused multiply & add instruction. It can apply these 2 operations in 1 cycle because of the hardware support.

#### Floating Point Arithmetic 😵

<span style="color: red; font-size: 12.0pt;">⚠️Dreadful Mathmatics Alert⚠️</span>
{: style="text-align: center;"}

**Step 0**

Assume we have an uint4_t number $$ 5_{\text{10}} = 0101_2 $$ that's packed with other 4bit ints.

We have an empty fp16 `result` $$ 0000\ 0000\ 0000\ 0000_2 $$

**Step 1**
```cpp
asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
```
Apply the mask `0x000f` = $$ 0000\ 0000\ 0000\ 1111_2 $$ using (`0xf0 & 0xcc`) as defined by `immLut`

The fp16 `result` = $$ 0000\ 0000\ 0000\ 0101_2 = 1 + 1/256 + 1/1024 $$

Apply the `I4s_TO_F16s_MAGIC_NUM` $$ 0x6400 = 0110\ 0100\ 0000\ 0000_2 $$ using the `(| 0xaa)` part of `immLut`

The fp16 `result` = $$ 0110\ 0100\ 0000\ 0101_2 = 2 ^ {\text{10}} * (1 + 1/256 + 1/1024)_{\text{10}} = 1031_{\text{10}} $$

**Step 2**
```cpp
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
```
Substract `FP16_TOP_MAGIC_NUM` = `0x6400` from `result`

The final fp16 `result` becomes $$ 5_{\text{10}} $$

(We don't write step-by-step base 2 representation of fp substraction because it requires multiple steps).

**Summary**

Here's how the FP result changes step by step to produce FP equivalent of $$ 0101_2 = 5_{\text{10}} $$

```
0000 0000 0000 0000 = 0
0000 0000 0000 0101 = 1031/1024 // after & mask
0110 0100 0000 0101 = 1031      // after | magic number
0100 0101 0000 0000 = 5         // after - magic number
```

#### The magic number - why it works

`I4s_TO_F16s_MAGIC_NUM = 0x64006400`
If we only look at the first 16 bits

```
0110 0100 0000 0000
SEEE EEMM MMMM MMMM
```

The exponent part is $$ 110\ 01_2 = 25_{\text{10}} $$

Reminder that FP16 = $$ sign * 2 ^ {\text{exponent - 15}} * (1 + Fraction)_{\text{10}} $$

This magic number is $$ 1 * 2 ^ {\text{25 - 15}} * 1_{\text{10}} = 1024_{\text{10}} $$

Applying `(| 0xaa)` essentially means the $$ original\ number * 1024 $$

#### The magic number - why there are 2
Why does it define `0x64006400` twice with different contant names? It turns out that this program was adapted from [Nvidia FasterTransformer](https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h) library.

Originally the 2nd magic number in the conversion section is different from the masking section:
{% highlight c %}
static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
{% endhighlight %}
This would extract another 8 from the original int8, which will result in $[-8, 7]$ range. However AWQ doesn't need to support a negative integers.

### Practice Time!!
This [PR](https://github.com/sgl-project/sglang/pull/4537) implements the converstion to BF16. Are you able to understand it? Do you know why it doesn't have a top mask, but shift 3 times instead?
