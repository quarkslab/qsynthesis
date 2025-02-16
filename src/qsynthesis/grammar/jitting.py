from qsynthesis.grammar.ops import BvOp
import pydffi
import array

# First, declare an FFI context
CODE = '''
#include <stdio.h>
#include <stdint.h> 
uint64_t add(uint64_t a, uint64_t b) { return a+b; }
uint64_t and(uint64_t a, uint64_t b) { return a&b; }
uint64_t or(uint64_t a, uint64_t b) { return a|b; }
uint64_t xor(uint64_t a, uint64_t b) { return a^b; }
uint64_t sub(uint64_t a, uint64_t b) { return a-b; }
uint64_t mul(uint64_t a, uint64_t b) { return a*b; }
uint64_t udiv(uint64_t a, uint64_t b) { return a/b; }
uint64_t usub(uint64_t a) { return -a; }
uint64_t invert(uint64_t a) { return ~a; }
uint64_t to_uint(int64_t a) { return (uint64_t) a; }
uint64_t urem(uint64_t a, uint64_t b) { return  b==0? a: a % b ; }
uint64_t ashr(uint64_t a, uint64_t b) { int bitsz = sizeof(uint64_t)*8; return (b >= bitsz)? ((int64_t) a) >> bitsz-1: ((int64_t) a) >> b; }
uint64_t sle(int64_t a, int64_t b) { return a <= b ; }
uint64_t slt(int64_t a, int64_t b) { return a < b ; }
uint64_t sge(int64_t a, int64_t b) { return a >= b ; }
uint64_t sgt(int64_t a, int64_t b) { return a > b ; }
uint64_t lshift(uint64_t a, uint64_t b) { return (b >= sizeof(uint64_t)*8)? 0: a << b; }
uint64_t rshift(uint64_t a, uint64_t b) { return (b >= sizeof(uint64_t)*8)? 0: a >> b; }
uint64_t rol(uint64_t i, uint64_t n) { int bitsz = 8*sizeof(n); return (n << (i%bitsz)) | (n >> (8*sizeof(n) - (i%bitsz))); }
uint64_t ror(uint64_t i, uint64_t n) { int bitsz = 8*sizeof(n); return (n >> (i%bitsz))|(n << (8*sizeof(n) - (i%bitsz))); }
uint64_t mod(uint64_t a, uint64_t b) { return a % b; }


void add_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] + b[i]; } }
void and_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] & b[i]; } }
void or_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] | b[i]; } }
void xor_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] ^ b[i]; } }
void sub_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] - b[i]; } }
void mul_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] * b[i]; } }
void udiv_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] / b[i]; } }
void usub_arr(uint64_t* dst, uint64_t* a, size_t n) { for(int i=0; i < n; i ++) { dst[i] = -a[i]; } }
void invert_arr(uint64_t* dst, uint64_t* a, size_t n) { for(int i=0; i < n; i ++) { dst[i] = ~a[i]; } }
void urem_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = b[i]==0? a[i]: a[i] % b[i] ; } }
void ashr_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { int bitsz = sizeof(uint64_t)*8; for(int i=0; i < n; i ++) { dst[i] = (b[i] >= bitsz)? ((int64_t) a[i]) >> bitsz-1: ((int64_t) a[i]) >> b[i]; } }
void sle_arr(uint64_t* dst, int64_t* a, int64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] <= b[i] ; } }
void slt_arr(uint64_t* dst, int64_t* a, int64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] < b[i] ; } }
void sge_arr(uint64_t* dst, int64_t* a, int64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] >= b[i] ; } }
void sgt_arr(uint64_t* dst, int64_t* a, int64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] > b[i] ; } }
void lshift_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = (b[i] >= sizeof(uint64_t)*8)? 0: a[i] << b[i]; } }
void rshift_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = (b[i] >= sizeof(uint64_t)*8)? 0: a[i] >> b[i]; } }
void rol_arr(uint64_t* dst, uint64_t* i, uint64_t* n, size_t sz) { for(int j=0; j < sz; j ++) { int bsz = 8*sizeof(n[j]); dst[j] = (n[j] << (i[j]%bsz)) | (n[j] >> (bsz - (i[j]%bsz))); } }
void ror_arr(uint64_t* dst, uint64_t* i, uint64_t* n, size_t sz) { for(int j=0; j < sz; j ++) { int bsz = 8*sizeof(n[j]); dst[j] = (n[j] >> (i[j]%bsz)) | (n[j] << (bsz - (i[j]%bsz))); } }
void mod_arr(uint64_t* dst, uint64_t* a, uint64_t* b, size_t n) { for(int i=0; i < n; i ++) { dst[i] = a[i] % b[i] ; } }
'''

ffi_ctx = pydffi.FFI(lazyJITWrappers=False)


def make_compilation_unit(bitsize: int):
    global ffi_ctx
    good_code = CODE.replace("64", str(bitsize))
    return ffi_ctx.compile(good_code)


def _get_func_name(op: 'Operator') -> str:
    return {
        BvOp.NOT: "invert",
        BvOp.AND: "and",
        BvOp.OR: "or",
        BvOp.XOR: "xor",
        BvOp.NEG: "usub",
        BvOp.ADD: "add",
        BvOp.MUL: "mul",
        BvOp.SUB: "sub",
        BvOp.SHL: "lshift",
        BvOp.LSHR: "rshift",
        BvOp.ROL: "rol",
        BvOp.ROR: "ror",
        BvOp.ASHR: "ashr",
    }[op.id]


def get_op_eval(cu, op):
    return getattr(cu.funcs, _get_func_name(op))


def get_op_eval_array(cu, op):
    return getattr(cu.funcs, f"{_get_func_name(op)}_arr")


def get_native_array_type(size_t, size):
    global ffi_ctx
    typ = {
        8: ffi_ctx.UInt8Ty,
        16: ffi_ctx.UInt16Ty,
        32: ffi_ctx.UInt32Ty,
        64: ffi_ctx.UInt64Ty
    }
    return ffi_ctx.arrayType(typ[size_t], size)


def init_array_cst(ffi_array, val, size, size_t) -> None:
    map = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}
    pydffi.view_as_bytes(ffi_array)[:] = array.array(map[size_t], [val]*size).tobytes()
