# GGML Backend Architecture - Critical Concepts for LNS Implementation

This document explains the ggml backend architecture from the HuggingFace blog post
and connects it to your GSoC LNS backend implementation using xlnscpp.

## Reference: HuggingFace GGML Introduction
https://huggingface.co/blog/introduction-to-ggml

---

## 1. GGML_BACKEND: The Computation Engine

### **What it is:**
`ggml_backend` is an **abstraction layer for where computation happens** (CPU, GPU, etc.).

### **Key responsibilities:**
- Execute tensor operations (matmul, add, etc.)
- Manage memory allocation on the device
- Provide device-specific implementations

### **Example backends in ggml:**
- `ggml_backend_cpu_init()` - CPU computation
- `ggml_backend_cuda_init()` - NVIDIA GPU
- `ggml_backend_metal_init()` - Apple Metal
- **YOUR TARGET:** `ggml_backend_lns_init()` - LNS using xlnscpp on CPU

### **Critical for your project:**
Your LNS backend will:
1. Receive FP32 tensors from llama.cpp
2. Convert to LNS using `fp2xlns32()`
3. Perform operations using `xlns32_mul`, `xlns32_add`
4. Convert back to FP32 using `xlns322fp()`
5. Return results to llama.cpp

```c
// Conceptual pseudocode for your LNS backend
ggml_backend_lns {
    // Convert input tensors
    for each element:
        lns_tensor[i] = fp2xlns32(fp_tensor[i]);
    
    // Compute in LNS
    lns_matmul(lns_A, lns_B, lns_C);  // Uses xlns32_mul internally
    
    // Convert output back
    for each element:
        fp_result[i] = xlns322fp(lns_tensor[i]);
}
```

---

## 2. GGML_CONTEXT: The Tensor Container

### **What it is:**
A memory arena that holds multiple tensors and their metadata.

### **Purpose:**
- Batch allocate memory for multiple tensors efficiently
- Track tensor shapes, strides, and data types
- Enable automatic memory cleanup

### **Think of it as:**
A "workspace" that holds all the tensors for one forward pass through a neural network layer.

### **For your LNS backend:**
The context will **still hold FP32 metadata** but your backend will:
- Temporarily create LNS copies during computation
- Or tag tensors to indicate they should be computed in LNS

```c
// ggml context example
ggml_context* ctx = ggml_init(...);

// Create tensors (metadata only at this point)
struct ggml_tensor* A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
struct ggml_tensor* B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, K);

// Your backend will convert these when computing
```

---

## 3. GGML_CGRAPH: The Computation Graph

### **What it is:**
A directed acyclic graph (DAG) describing **what operations to perform**.

### **Purpose:**
- Separate **operation definition** from **execution**
- Enable optimization passes (fusion, reordering)
- Support automatic differentiation (for training)

### **Example structure:**
```
Input A (4×3) ──┐
                ├─> MatMul ──> Output C (4×2)
Input B (3×2) ──┘
```

### **For your LNS backend:**
The cgraph tells your backend:
1. Which tensors to multiply
2. In what order
3. Your backend executes each node using LNS operations

```c
// Build computation graph
struct ggml_cgraph* graph = ggml_new_graph(ctx);
struct ggml_tensor* result = ggml_mul_mat(ctx, A, B);  // Defines operation
ggml_build_forward_expand(graph, result);  // Adds to graph

// Execute on your backend
ggml_backend_graph_compute(lns_backend, graph);  // Your code runs here!
```

---

## 4. GGML_BACKEND_BUFFER: Holds Tensor Data

### **What it is:**
The **actual memory** where tensor values live.

### **Key distinction:**
- `ggml_tensor` = metadata (shape, type, name)
- `ggml_backend_buffer` = actual float/int values in memory

### **For your LNS backend:**
You have two options:

**Option A: Convert on-the-fly (RECOMMENDED FOR PROOF-OF-CONCEPT)**
```c
// Buffer stays FP32, convert during compute
float* fp_data = (float*)ggml_backend_buffer_get_data(buffer);
xlns32* lns_temp = malloc(...);
for (i = 0; i < n; i++)
    lns_temp[i] = fp2xlns32(fp_data[i]);
// ... compute ...
for (i = 0; i < n; i++)
    fp_data[i] = xlns322fp(lns_temp[i]);
```

**Option B: Native LNS storage (ADVANCED, NOT REQUIRED)**
```c
// Store data as LNS internally
// More complex, requires custom buffer type
```

**Start with Option A.**

---

## 5. GGML_BACKEND_BUFFER_TYPE: Memory Allocator

### **What it is:**
A factory for creating backend-specific buffers.

### **Purpose:**
- Abstract device memory allocation
- Handle device-specific alignment requirements
- Support pinned memory, unified memory, etc.

### **For your LNS backend:**
Initially, you can just use the CPU buffer type since LNS will run on CPU:

```c
ggml_backend_buffer_type_t lns_buftype = ggml_backend_cpu_buffer_type();
// Or create custom if needed later
```

---

## 6. PUTTING IT ALL TOGETHER: LNS Backend Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  LLAMA.CPP (sees everything as FP32)                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  GGML_CONTEXT (tensor metadata)                         │
│    - Tensor A: shape [4, 3], type F32                   │
│    - Tensor B: shape [3, 2], type F32                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  GGML_CGRAPH (operation: C = A × B)                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  GGML_BACKEND_LNS (YOUR CODE)                           │
│                                                          │
│  1. Read FP32 data from backend_buffer                  │
│     float* A_fp = get_buffer_data(A);                   │
│                                                          │
│  2. Convert to LNS                                      │
│     xlns32* A_lns = malloc(...);                        │
│     for (i...) A_lns[i] = fp2xlns32(A_fp[i]);          │
│                                                          │
│  3. Compute using xlnscpp                               │
│     for (i...) {                                        │
│       xlns32 sum = fp2xlns32(0.0f);                     │
│       for (k...)                                        │
│         sum = xlns32_add(sum,                           │
│                 xlns32_mul(A_lns[...], B_lns[...]));   │
│       C_lns[i] = sum;                                   │
│     }                                                   │
│                                                          │
│  4. Convert back to FP32                                │
│     for (i...) C_fp[i] = xlns322fp(C_lns[i]);          │
│                                                          │
│  5. Write results to output buffer                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  LLAMA.CPP receives FP32 result                         │
│  (Never knows LNS was used internally!)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 7. MINIMAL IMPLEMENTATION STRATEGY FOR GSOC

### **Phase 1: Backend Skeleton (Week 1-2)**
- Register a new backend with ggml
- Implement `ggml_backend_lns_init()`
- Use CPU buffer type initially
- Make it return FP32 results (no LNS yet)

### **Phase 2: Single Operation (Week 3-4)**
- Implement matmul in LNS
- Profile conversion overhead
- Compare numeric accuracy with FP32

### **Phase 3: Validation (Week 5-6)**
- Run a tiny LLM model (tinyllama or similar)
- Measure perplexity difference
- Document precision loss

### **Phase 4: Report (Week 7-8)**
- Design document
- Numeric analysis
- Architectural decisions

---

## 8. CRITICAL WARNINGS

### **DO NOT:**
- ❌ Try to implement all ggml operations (there are 100+)
- ❌ Optimize performance before correctness
- ❌ Store tensors in LNS format (convert on-the-fly)
- ❌ Attempt GPU implementation

### **DO:**
- ✅ Focus on matrix multiply ONLY initially
- ✅ Validate every result against FP32
- ✅ Use xlns32_ideal mode first (accurate but slow)
- ✅ Measure numeric drift explicitly

---

## 9. TEST STRATEGY

### **Unit test:**
```c
// Test single matmul
float A_fp[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
float B_fp[6] = {1,2,3,4,5,6};
float C_fp[8];
float C_lns[8];

// FP32 reference
matmul_fp32(A_fp, B_fp, C_fp, 4, 3, 2);

// LNS backend
ggml_backend_lns_matmul(A_fp, B_fp, C_lns, 4, 3, 2);

// Compare
for (int i = 0; i < 8; i++) {
    float err = fabs(C_fp[i] - C_lns[i]);
    printf("C[%d]: FP32=%.4f LNS=%.4f Error=%.6f\n", 
           i, C_fp[i], C_lns[i], err);
}
```

---

## 10. QUESTIONS TO ANSWER IN YOUR DESIGN PROPOSAL

1. **Where exactly will FP↔LNS conversion happen?**
   (Answer: Inside backend operation implementations)

2. **How will you handle tensor sizes that don't fit in memory?**
   (Answer: Process in chunks, document size limits)

3. **What numeric precision can you guarantee?**
   (Answer: Measure actual error on test matrices)

4. **Can you support quantized models (int8, int4)?**
   (Answer: NO for proof-of-concept - scope creep)

5. **Will you modify ggml or just add a backend?**
   (Answer: Just add backend, minimal ggml changes)

---

## 11. YOUR NEXT CONCRETE STEPS

1. ✅ **DONE:** Understand xlnscpp
2. ✅ **DONE:** Understand FP matmul
3. ⏭️  **NEXT:** Clone ggml repository
4. ⏭️  **NEXT:** Find ggml's backend registration code
5. ⏭️  **NEXT:** Study an existing backend (start with CPU backend)
6. ⏭️  **NEXT:** Create a "hello world" backend that just calls CPU backend
7. ⏭️  **NEXT:** Replace one operation with LNS version

**Do NOT attempt to design the entire system yet.**
Build incrementally. Validate at each step.

---

## 12. RESOURCES

### **Code locations (in ggml repo):**
```
ggml/src/ggml-backend.c          - Backend interface
ggml/src/ggml-backend-impl.h     - Backend implementation helpers
ggml/src/ggml-cpu.c              - CPU backend (your template)
ggml/examples/simple/            - Simple examples
```

### **Key functions to study:**
```c
ggml_backend_cpu_init()          - How to create a backend
ggml_backend_cpu_compute_forward() - Where operations execute
ggml_backend_buffer_get_data()   - Access tensor memory
```

---

## FINAL REALITY CHECK

Your mentor expects:
- A working backend that proves LNS can be integrated
- Honest numeric analysis (including failures)
- Minimal scope (matmul + maybe add)
- Clear documentation of trade-offs

Your mentor does NOT expect:
- Production-ready performance
- All operations implemented
- GPU support
- Automatic conversion handling

**Stay focused. Stay minimal. Prove the concept.**
