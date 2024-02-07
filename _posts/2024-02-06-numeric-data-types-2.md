---
layout: post
title:  "Numeric Data Types Part 2: Conversion"
date:   2024-02-06 18:18:42 +0000
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

### Section 2: Masking - preparation
This section prepares the `uint4_t` to an intermediate form before finally converting to `fp16`.
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
static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
static constexpr uint32_t TOP_MASK              = 0x00f000f0;
static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;
{% endhighlight %}

#### An intelligent hardware design
#### A trick
### Section 3: Conversion
#### Constants
{% highlight c %}
static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
static constexpr uint32_t NEG_64 = 0xd400d400;
{% endhighlight %}

#### Assembly 😨
{% highlight c %}
asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
{% endhighlight %}
- `.f16x2`: 2 `fp16` numbers. It's only supported by [sm_53](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) above - [Pascal GPU Architecture](https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/) or later.

#### Floating Point Arithmetic 😵

<span style="color: red; font-size: 12.0pt;">⚠️Dreadful Mathmatics Alert⚠️</span>
{: style="text-align: center;"}

#### The magic number - why it works
#### The magic number - why there are 2
Why does it define `0x64006400` twice with different contant names? It turns out that this program was adapted from [Nvidia FasterTransformer](https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h) library.

Originally the 2nd magic number in the conversion section is different from the masking section:
{% highlight c %}
static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
{% endhighlight %}
This would extract another 8 from the original int8, which will result in $[-8, 7]$ range. However AWQ doesn't need to support a negative integers.

### The Elephant in the Room
This function is awesome, but **Why do we need this function in the first place**?
