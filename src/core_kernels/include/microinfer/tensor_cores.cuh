#pragma once

// A warp's matrix multiply on the tensor cores, hand-written over the two PTX
// instructions it takes: mma.sync, 16 x 8 outputs accumulated in fp32 from 16
// x 16 and 16 x 8 fp16 operands, and ldmatrix.sync, which reads 8 x 8 fp16
// matrices from shared memory into the registers mma.sync reads them in. Both
// need sm_80. Within the from-scratch boundary: instructions, not a library
// (ADR-0001).
//
// Which thread holds what, `lane` being the thread in its warp:
// - C, 16 x 8 fp32, four registers: rows lane/4 and lane/4 + 8 in (0, 1) and
//   (2, 3), columns 2 (lane % 4) and the one after in each pair;
// - A, 16 x 16 fp16, four registers of two: rows lane/4 and lane/4 + 8 at
//   columns 2 (lane % 4) in registers 0 and 1, and at 8 more in 2 and 3;
// - B, 16 x 8 fp16, two registers of two: column lane/4, at rows 2 (lane % 4)
//   in register 0 and 8 more in 1.
// So two C's side by side, rounded to fp16, are the registers of an A: what
// a warp holds of P = softmax(QK^T) it can multiply by V with no exchange.

#include <cuda_fp16.h>

#include <cstdint>

namespace microinfer
{
  namespace tensor_cores
  {

    // C += A B.
    __device__ __forceinline__ void
    multiply_add(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
    {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                     "r"(b[1]));
    }

    // C += A B with C in fp16, two registers of two, columns as above, rows
    // lane/4 in the first and lane/4 + 8 in the second. Half the cost of an
    // fp32 C on a consumer GPU, for what is small enough to be summed in it.
    __device__ __forceinline__ void multiply_add_fp16(uint32_t (&c)[2],
                                                      const uint32_t (&a)[4],
                                                      const uint32_t (&b)[2])
    {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                   "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                   : "+r"(c[0]), "+r"(c[1])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                     "r"(b[1]));
    }

    // Where a thread reads its part of an operand that begins at a row-major
    // matrix's first element, as an address in shared memory: the matrix
    // begins `matrix`, has `stride` elements a row, and it and its rows begin
    // on 16 bytes. An operand that begins at another element of the matrix is
    // read as many bytes further, which an instruction's own offset adds.
    __device__ __forceinline__ uint32_t shared_address(const __half *at)
    {
      return static_cast<uint32_t>(__cvta_generic_to_shared(at));
    }
    __device__ __forceinline__ uint32_t address_a(const __half *matrix,
                                                  int stride)
    {
      const int lane = threadIdx.x % 32;
      return shared_address(matrix + (lane % 16) * stride + (lane / 16) * 8);
    }
    __device__ __forceinline__ uint32_t address_b_rows(const __half *matrix,
                                                       int stride)
    {
      const int lane = threadIdx.x % 32;
      return shared_address(matrix + (lane % 8) * stride + (lane / 8 % 2) * 8);
    }
    __device__ __forceinline__ uint32_t address_b_columns(const __half *matrix,
                                                          int stride)
    {
      return shared_address(matrix + (threadIdx.x % 16) * stride);
    }

    // The A that rows [0, 16) and columns [0, 16) make, from an address_a.
    __device__ __forceinline__ void load_a(uint32_t (&a)[4], uint32_t address)
    {
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                   "{%0,%1,%2,%3}, [%4];\n"
                   : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                   : "r"(address));
    }

    // The B that is the transpose of rows [0, 8) and columns [0, 16), eight
    // keys' sixteen channels, the reduction along a row of memory, from an
    // address_b_rows.
    __device__ __forceinline__ void load_b_rows(uint32_t (&b)[2],
                                                uint32_t address)
    {
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                   : "=r"(b[0]), "=r"(b[1])
                   : "r"(address));
    }

    // The B that rows [0, 16) and columns [0, 8) make as they are, sixteen
    // values' eight channels, the reduction down a column of memory, from an
    // address_b_columns.
    __device__ __forceinline__ void load_b_columns(uint32_t (&b)[2],
                                                   uint32_t address)
    {
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                   "{%0,%1}, [%2];\n"
                   : "=r"(b[0]), "=r"(b[1])
                   : "r"(address));
    }

  } // namespace tensor_cores
} // namespace microinfer
