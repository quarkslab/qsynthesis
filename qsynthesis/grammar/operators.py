from qsynthesis.grammar.ops import BvOp, Operator
import operator
import pydffi


# First, declare an FFI context
CODE = '''
#include <stdio.h>
#include <stdint.h> 
uint64_t add(uint64_t a, uint64_t b) { return a+b; }
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

ffi_ctx = pydffi.FFI()
CU = ffi_ctx.compile(CODE)


OPERATORS = {               # ID               strop    Trit op         Py op                 Eval Array          arit comm   id_eq  id_zero is_pfx  can_ov bool_ret
    # BoolOp.EQUAL:    Operator(BoolOp.EQUAL,    "==",    operator.eq,    operator.eq,                            2,   True,  True,  False,  False,  False, True),
    # BoolOp.DISTINCT: Operator(BoolOp.DISTINCT, "!=",    operator.ne,    operator.ne,                            2,   True,  False, True,   False,  False, True),
    # BoolOp.IFF:      Operator(BoolOp.IFF,      "iff",   "iff",          None,                                   2,   False, False, False,  False,  False, True),
    # BoolOp.LOR:      Operator(BoolOp.LOR,      "lor",   "lor",          lambda x,y: x or y,                     2,   True,  True,  False,  True,   False, True),
    # BoolOp.AND:      Operator(BoolOp.AND,      "land",  "land",         lambda x,y: x and y,                    2,   True,  True,  False,  True,   False, True),
    # BoolOp.NOT:      Operator(BoolOp.NOT,      "lnot",  "lnot",         lambda x: not x,                        1,   False, False, False,  True,   False, True),
    BvOp.NOT:        Operator(BvOp.NOT,        "~",     operator.invert, CU.funcs.invert,    CU.funcs.invert_arr, 1,   False, False, False,  True,   False, False),
    BvOp.AND:        Operator(BvOp.AND,        "&",     operator.and_,  operator.and_,       CU.funcs.and_arr,    2,   True,  True,  False,  False,  False, False),
    BvOp.OR:         Operator(BvOp.OR,         '|',     operator.or_,   operator.or_,        CU.funcs.or_arr,     2,   True,  True,  False,  False,  False, False),
    BvOp.XOR:        Operator(BvOp.XOR,        '^',     operator.xor,   operator.xor,        CU.funcs.xor_arr,    2,   True,  False, True,   False,  False, False),
    BvOp.NEG:        Operator(BvOp.NEG,        '-',     operator.neg,   CU.funcs.usub,       CU.funcs.usub_arr,   1,   False, False, False,  True,   False, False),
    BvOp.ADD:        Operator(BvOp.ADD,        '+',     operator.add,   CU.funcs.add,        CU.funcs.add_arr,    2,   True,  False, False,  False,  True,  False),
    BvOp.MUL:        Operator(BvOp.MUL,        '*',     operator.mul,   CU.funcs.mul,        CU.funcs.mul_arr,    2,   True,  False, False,  False,  True,  False),
    BvOp.SUB:        Operator(BvOp.SUB,        '-',     operator.sub,   CU.funcs.sub,        CU.funcs.sub_arr,    2,   False, False, True,   False,  False, False),
    BvOp.SHL:        Operator(BvOp.SHL,        "<<",    operator.lshift, CU.funcs.lshift,    CU.funcs.lshift_arr, 2,   False, False, False,  False,  True,  False),
    BvOp.LSHR:       Operator(BvOp.LSHR,       ">>",    operator.rshift, CU.funcs.rshift,    CU.funcs.rshift_arr, 2,   False, False, False,  False,   False, False),

    BvOp.ROL:        Operator(BvOp.ROL,        "bvrol", "bvrol",         CU.funcs.rol,       CU.funcs.rol_arr,    2,   False, False, False,  True,   False, False),
    BvOp.ROR:        Operator(BvOp.ROR,        "bvror", "bvror",         CU.funcs.ror,       CU.funcs.ror_arr,    2,   False, False, False,  True,   False, False),
    # BvOp.UDIV:       Operator(BvOp.UDIV,       "/",     operator.truediv,CU.funcs.udiv,      CU.funcs.udiv_arr,   2,   False, False, False,  False,  False, False), # TODO: Fix /0
    # BvOp.UREM:       Operator(BvOp.UREM,       "%",     operator.mod,    CU.funcs.urem,      CU.funcs.urem_arr,   2,   False, False, False,  False,  False, False), # TODO: Fix /0
    BvOp.ASHR:       Operator(BvOp.ASHR,       "bvashr", "bvashr",       CU.funcs.ashr,      CU.funcs.ashr_arr,   2,   False, False, False,  True,   False, False),
    # BvOp.SDIV:       Operator(BvOp.SDIV,       "bvsdiv","bvsdiv",       operator.truediv,    2,   False, False, False,  False,  False, False),
    # BvOp.SREM:       Operator(BvOp.SREM,       "bvsrem","bvsrem",       None,                2,   False, False, True,   True,   False, False),
    # BvOp.SMOD:       Operator(BvOp.SMOD,       "bvsmod","bvsmod",       CU.funcs.mod,        2,   False, False, True,   False,  False, False),
    # BvOp.XNOR:       Operator(BvOp.XNOR,       "bvxnor","bvxnor",       lambda x,y:~(x ^ y), 2,   True,  False, False,  True,   False, False),
    # BvOp.NOR:        Operator(BvOp.NOR,        "bvnor", "bvnor",        lambda x, y:~(x | y),2,   True,  False, False,  True,   False, False),
    # BvOp.NAND:       Operator(BvOp.NAND,       "bvnand","bvnand",       lambda x, y:~(x & y),2,   True,  False, False,  True,   False, False),
    # BvOp.ZEXT:       Operator(BvOp.ZEXT,       "zx",    "zx",           lambda y, x: x,      2,   False, True,  False,  True,   False, False),
    # BvOp.SEXT:       Operator(BvOp.SEXT,       "sx",    "sx",           sign_ext,            2,   False, False, False,  True,   False, False), # FIXME: operator
    # BvOp.CONCAT:     Operator(BvOp.CONCAT,     "concat","concat",       concat,              2,   False, False, False,  True,   False, False), # FIXME: operator
    # BvOp.EXTRACT:    Operator(BvOp.EXTRACT,    "extract","extract",     extract,             2,   False, False, False,  True,   False, False), # FIXME: operator
    # BvOp.ITE:        Operator(BvOp.ITE,        "If",     "If",          ite,                 3,   False, False, False,  True,   False, False),
    # BvOp.UGE:        Operator(BvOp.UGE,        ">=",     operator.ge,   operator.ge,         2,   False, True,  False,  False,  False, True),
    # BvOp.UGT:        Operator(BvOp.UGT,        ">",      operator.gt,   operator.gt,         2,   False, False, True,   False,  False, True),
    # BvOp.ULE:        Operator(BvOp.ULE,        "<=",     operator.le,   operator.le,         2,   False, True,  False,  False,  False, True),
    # BvOp.ULT:        Operator(BvOp.ULT,        "<",      operator.lt,   operator.lt,         2,   False, False, True,   False,  False, True),
    # BvOp.SLE:        Operator(BvOp.SLE,        "bvsle",  "bvsle",       CU.funcs.sle,        2,   False, True,  False,  True,   False, True),
    # BvOp.SLT:        Operator(BvOp.SLT,        "bvslt",  "bvslt",       CU.funcs.slt,        2,   False, False, True,   True,   False, True),
    # BvOp.SGE:        Operator(BvOp.SGE,        "bvsge",  "bvsge",       CU.funcs.sge,        2,   False, True,  False,  True,   False, True),
    # BvOp.SGT:        Operator(BvOp.SGT,        "bvsgt",  "bvsgt",       CU.funcs.sgt,        2,   False, False, True,   True,   False, True)
}
