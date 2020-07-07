import logging
from pathlib import Path

from triton import ARCH

from qsynthesis.utils.symexec import SimpleSymExec
from qsynthesis.algorithms import TopDownSynthesizer
from qsynthesis.tables import LookupTablePickle

#logging.basicConfig(level=logging.DEBUG)

RIP_ADDR = 0x40B160
RSP_ADDR = 0x800000

INSTRUCTIONS = [b'U', b'H\x89\xe5', b'H\x89}\xf8', b'H\x89u\xf0', b'H\x89U\xe8', b'H\x89M\xe0', b'L\x89E\xd8',
                b'H\x8bE\xf0', b'H#E\xe0', b'H\x89\xc2', b'H\x8bE\xf0', b'H\x0bE\xe0', b'H\x0f\xaf\xd0', b'H\x8bE\xe0',
                b'H\xf7\xd0', b'H#E\xf0', b'H\x89\xc1', b'H\x8bE\xf0', b'H\xf7\xd0', b'H#E\xe0', b'H\x0f\xaf\xc1',
                b'H\x01\xc2', b'H\x8bE\xe0', b'H\x0f\xaf\xc0', b'H\x89\xd6', b'H!\xc6', b'H\x8bE\xf0', b'H#E\xe0',
                b'H\x89\xc2', b'H\x8bE\xf0', b'H\x0bE\xe0', b'H\x0f\xaf\xd0', b'H\x8bE\xe0', b'H\xf7\xd0', b'H#E\xf0',
                b'H\x89\xc1', b'H\x8bE\xf0', b'H\xf7\xd0', b'H#E\xe0', b'H\x0f\xaf\xc1', b'H\x01\xc2', b'H\x8bE\xe0',
                b'H\x0f\xaf\xc0', b'H\t\xd0', b'H)\xc6', b'H\x89\xf0', b'H\x83\xe8\x01', b'H3E\xf0', b'H\x89\xc2',
                b'H\x8bE\xf0', b'H#E\xe0', b'H\x89\xc1', b'H\x8bE\xf0', b'H\x0bE\xe0', b'H\x0f\xaf\xc8', b'H\x8bE\xe0',
                b'H\xf7\xd0', b'H#E\xf0', b'H\x89\xc6', b'H\x8bE\xf0', b'H\xf7\xd0', b'H#E\xe0', b'H\x0f\xaf\xc6',
                b'H\x01\xc1', b'H\x8bE\xe0', b'H\x0f\xaf\xc0', b'H1\xc8', b'H#E\xf0', b'H\x01\xc0', b'H)\xc2',
                b'H\x89\xd0', b']', b'\xc3']


def test():
    # Perform symbolic execution of the instructions
    symexec = SimpleSymExec(ARCH.X86_64)
    symexec.initialize_register('rip', RIP_ADDR)
    symexec.initialize_register('rsp', RSP_ADDR)
    for opcode in INSTRUCTIONS:
        symexec.execute(opcode)
    rax = symexec.get_register_ast("rax")

    # Load lookup tables
    current = Path(__file__).parent.absolute()
    ltms = [LookupTablePickle.load(x) for x in Path(current / "../../lts/zlts_15_mirror_str").iterdir()]

    # Perform Synthesis of the expression
    synthesizer = TopDownSynthesizer(ltms)
    synt_rax, simp = synthesizer.synthesize(rax)

    # Print synthesis results
    print(f"simplified: {simp}")
    print(f"synthesized expression: {synt_rax.pp_str}")
    print(f"size: {rax.node_count} -> {synt_rax.node_count} scale reduction:{synt_rax.node_count/rax.node_count:.2f}")
    return symexec, synt_rax


if __name__ == "__main__":
    sx, srax = test()
