import logging
import sys

from triton import ARCH

from qsynthesis import SimpleSymExec, TopDownSynthesizer, InputOutputOracleLevelDB
import qsynthesis
import logging
logging.basicConfig(level=logging.DEBUG)

qsynthesis.enable_logging()


RIP_ADDR = 0x40B160
RSP_ADDR = 0x800000

blob = b'UH\x89\xe5H\x89}\xf8H\x89u\xf0H\x89U\xe8H\x89M\xe0L\x89E\xd8H\x8bE' \
       b'\xe0H\xf7\xd0H\x0bE\xf8H\x89\xc2H\x8bE\xe0H\x01\xd0H\x8dH\x01H\x8b' \
       b'E\xf8H+E\xe8H\x8bU\xe8H\xf7\xd2H\x0bU\xf8H\x01\xd2H)\xd0H\x83\xe8' \
       b'\x02H!\xc1H\x8bE\xe0H\xf7\xd0H\x0bE\xf8H\x89\xc2H\x8bE\xe0H\x01\xd0' \
       b'H\x8dp\x01H\x8bE\xf8H+E\xe8H\x8bU\xe8H\xf7\xd2H\x0bU\xf8H\x01\xd2' \
       b'H)\xd0H\x83\xe8\x02H\t\xf0H)\xc1H\x89\xc8H\x83\xe8\x01]\xc3'


def test(oracle_file):
    # Perform symbolic execution of the instructions
    symexec = SimpleSymExec(ARCH.X86_64)
    symexec.initialize_register('rip', RIP_ADDR)
    symexec.initialize_register('rsp', RSP_ADDR)
    symexec.execute_blob(blob, RIP_ADDR)
    rax = symexec.get_register_ast("rax")

    # Load lookup tables
    ltm = InputOutputOracleLevelDB.load(oracle_file)

    # Perform Synthesis of the expression
    synthesizer = TopDownSynthesizer(ltm)
    synt_rax, simp = synthesizer.synthesize(rax)

    # Print synthesis results
    print(f"simplified: {simp}")
    print(f"synthesized expression: {synt_rax.pp_str}")
    sz, nsz = rax.node_count, synt_rax.node_count
    print(f"size: {rax.node_count} -> {synt_rax.node_count}\nsize reduction:{((sz-nsz)*100)/sz:.2f}%")
    return symexec, rax, synt_rax


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"./{sys.argv[0]} oracle_table")
        sys.exit(1)
    sx, rax, srax = test(sys.argv[1])
'''
simplified: True
synthesized expression: (((~(rcx)) & rdi) ^ (~(rdx)))
size: 51 -> 7
size reduction:86.27%
'''