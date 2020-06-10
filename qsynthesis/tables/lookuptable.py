import pickle
from pathlib import Path
import logging
from typing import Optional, List, Dict, Union, Generator, Tuple, Any, TypeVar, Iterable

from qsynthesis.grammar import TritonGrammar
from qsynthesis.tables.base import BaseTable, HashType, Hash


class LookupTable(BaseTable):
    def __init__(self, gr: TritonGrammar, inputs: Union[int, List[Dict[str, int]]], hash_mode: HashType=HashType.RAW, f_name: str = ""):
        super(LookupTable, self).__init__(gr, inputs, hash_mode, f_name)
        self.lookup_table = {}

    @property
    def size(self):
        return len(self.lookup_table)

    def _get_item(self, hash: Hash) -> Optional[str]:
        return self.lookup_table.get(hash, None)

    def __iter__(self) -> Iterable[Tuple[Hash, str]]:
        return self.lookup_table.items()

    def add_entry(self, hash: Hash, value: str) -> None:
        self.lookup_table[hash] = value

    def add_entries(self, worklist):
        for s, outs in worklist:
            print("outs", outs, " str:", s)
            h = self.hash(outs)
            self.lookup_table[h] = s

    def save(self, file: Union[Path, str]) -> None:
        logging.info(f"Start dumping lookup table: {len(self.lookup_table)} entries")
        with open(file, 'wb') as f:
            g_dict = self.grammar.to_dict()
            g_dict['hash-mode'] = self.hash_mode.name
            pickle.dump(g_dict, f)
            pickle.dump(self.grammar.dump_inputs(self.inputs), f)
            pickle.dump(self.lookup_table, f)

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTable':
        f = Path(file)
        with open(f, 'rb') as f:
            raw = pickle.load(f)
            hm = HashType[raw['hash-mode']] if "hash-mode" in raw else HashType.RAW
            gr = TritonGrammar.from_dict(raw)
            inp_l = pickle.load(f)
            inputs = TritonGrammar.load_inputs(inp_l)
            lkp = LookupTable(gr, inputs, hm, f.name)
            lkp.lookup_table = pickle.load(f)
            return lkp

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'BaseTable':
        return LookupTable(grammar, inputs, hash_mode, filename)
