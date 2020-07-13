import pickle
from pathlib import Path
import logging
from typing import Optional, List, Dict, Union, Generator, Tuple, Any, TypeVar, Iterable
from binascii import unhexlify, hexlify

from qsynthesis.grammar import TritonGrammar
from qsynthesis.tables.base import LookupTable, HashType, Hash


class LookupTablePickle(LookupTable):
    def __init__(self, gr: TritonGrammar, inputs: Union[int, List[Dict[str, int]]], hash_mode: HashType=HashType.RAW, f_name: str = ""):
        super(LookupTablePickle, self).__init__(gr, inputs, hash_mode, f_name)
        self.lookup_table = {}

    @property
    def size(self):
        return len(self.lookup_table)

    def _get_item(self, hash: Hash) -> Optional[str]:
        return self.lookup_table.get(hash, None)

    def __iter__(self) -> Iterable[Tuple[Hash, str]]:
        return iter(self.lookup_table.items())

    def add_entry(self, hash: Hash, value: str) -> None:
        self.lookup_table[hash] = value

    def add_entries(self, worklist, calc_hash=False):
        for outs, s in worklist:
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
    def load(file: Union[Path, str]) -> 'LookupTablePickle':
        f = Path(file)
        with open(f, 'rb') as f:
            raw = pickle.load(f)
            hm = HashType[raw['hash_mode']] if "hash_mode" in raw else HashType.RAW
            gr = TritonGrammar.from_dict(raw)
            inp_l = pickle.load(f)
            inputs = TritonGrammar.load_inputs(inp_l)
            lkp = LookupTablePickle(gr, inputs, hm, f.name)
            lkp.lookup_table = pickle.load(f)
            return lkp

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'LookupTablePickle':
        return LookupTablePickle(grammar, inputs, hash_mode, filename)


class LookupTableRaw(LookupTable):

    EXPORT_FILE_CHUNK_LIMIT = 40000000

    def __init__(self, gr: TritonGrammar, inputs: Union[int, List[Dict[str, int]]], hash_mode: HashType=HashType.RAW, f_name: str = ""):
        super(LookupTableRaw, self).__init__(gr, inputs, hash_mode, f_name)

    @property
    def size(self):
        raise NotImplementedError()

    def _get_item(self, hash: Hash) -> Optional[str]:
        raise NotImplementedError()

    def __iter__(self) -> Iterable[Tuple[Hash, str]]:
        with open(str(self.name), "rb") as f:
            _ = f.readline()
            _ = f.readline()
            while 1:
                line = f.read(16)
                if not line:
                    break
                s = f.readline()
                yield line, s.strip().decode()

    def add_entry(self, hash: Hash, value: str) -> None:
        with open(str(self.name), "ab") as f:
            f.write(f"{hash},{value}\n".encode())

    def add_entries(self, worklist, calc_hash=False):
        import hashlib
        count = len(worklist)
        if calc_hash:
            hash_fun = lambda x: hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash
        else:
            hash_fun = lambda x: x
        print("\nExport data")

        f_id = 1
        f_counter = 0
        f = open(self.name, "ab")

        for step in range(0, count, 10000):
            f_counter += 10000
            print(f"process {step}/{count}\r", end="")
            chk_s = b"\n".join(hash_fun(outs)+s.encode() for outs, s in worklist[step:step+10000])+b"\n"
            f.write(chk_s)
            if f_counter > self.EXPORT_FILE_CHUNK_LIMIT:
                f.close()
                fname = f"{self.name}.{f_id}"
                LookupTableRaw.create(fname, self.grammar, self.inputs, self.hash_mode)
                f = open(fname, "ab")
                f_id += 1
                f_counter = 0

    def save(self, file: Union[Path, str]) -> None:
        logging.info("Saved")

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTableRaw':
        import json
        f = Path(file)
        with open(f, 'rb') as f:
            raw = json.loads(f.readline())
            hm = HashType[raw['hash_mode']] if "hash_mode" in raw else HashType.RAW
            gr = TritonGrammar.from_dict(raw)
            inp_l = json.loads(f.readline())
            inputs = TritonGrammar.load_inputs(inp_l)
            lkp = LookupTableRaw(gr, inputs, hm, f.name)
            return lkp

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'LookupTableRaw':
        import json
        with open(filename, "wb") as f:
            d = grammar.to_dict()
            d["hash_mode"] = hash_mode.name
            f.write(f"{json.dumps(d)}\n".encode())
            f.write(f"{json.dumps(inputs)}\n".encode())
        return LookupTableRaw(grammar, inputs, hash_mode, filename)
