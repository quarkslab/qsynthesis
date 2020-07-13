from pathlib import Path
import logging
from enum import IntEnum, Enum
import hashlib
import json
from typing import Optional, List, Dict, Union, Tuple, TypeVar, Iterable, Generator, Iterator

import plyvel

from qsynthesis.grammar import TritonGrammar, BvOp
from qsynthesis.tables.base import LookupTable, Expr, HashType, Hash

META_KEY = b"metadatas"
VARS_KEY = b"variables"
INPUTS_KEY = b"inputs"
SIZE_KEY = b"size"


class LookupTableLevelDB(LookupTable):
    def __init__(self, grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        super(LookupTableLevelDB, self).__init__(grammar, inputs, hash_mode, f_name)
        self.db = None

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'LookupTableLevelDB':
        # TODO: If it exists deleting it ?
        db = plyvel.DB(str(filename), create_if_missing=True)

        metas = dict(hash_mode=hash_mode.name, operators=[x.value for x in grammar.ops])
        db.put(META_KEY, json.dumps(metas).encode())
        db.put(VARS_KEY, json.dumps(grammar.vars_dict).encode())
        db.put(INPUTS_KEY, json.dumps(inputs).encode())
        lkp = LookupTableLevelDB(grammar=grammar, inputs=inputs, hash_mode=hash_mode, f_name=filename)
        lkp.db = db
        return lkp

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTableLevelDB':
        db = plyvel.DB(str(file))
        metas = json.loads(db.get(META_KEY))
        ops = [BvOp(x) for x in metas['operators']]
        vars = list(json.loads(db.get(VARS_KEY)).items())
        inps = json.loads(db.get(INPUTS_KEY))
        grammar = TritonGrammar(vars=vars, ops=ops)
        lkp = LookupTableLevelDB(grammar=grammar, inputs=inps, hash_mode=HashType[metas['hash_mode']], f_name=file)
        lkp.db = db
        return lkp

    def add_entry(self, hash: Hash, value: str):
        self.db.put(hash, value.encode())
        self.db.put(SIZE_KEY, (str(int(self.db.get(SIZE_KEY))+1)).encode())

    def add_entries(self, entries: List[Tuple[List[int], str]], calc_hash=False, chunk_size=10000, update_count=True) -> None:
        count = len(entries)
        if calc_hash:
            hash_fun = lambda x: hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash
        else:
            hash_fun = lambda x: x
        for step in range(0, count, chunk_size):
            with self.db.write_batch(sync=True) as wb:
                for outs, s in entries[step:step+chunk_size]:
                    wb.put(hash_fun(outs), s.encode())
        if update_count:
            self.db.put(SIZE_KEY, (str(int(self.db.get(SIZE_KEY)) + count)).encode())

    def __iter__(self) -> Iterator[Tuple[Hash, str]]:
        for key, value in self.db:
            if key not in [META_KEY, VARS_KEY, INPUTS_KEY, SIZE_KEY]:
                yield key, value.decode()

    @property
    def is_writable(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return int(self.db.get(SIZE_KEY))

    def _get_item(self, h: Hash) -> Optional[str]:
        entry = self.db.get(h)
        return entry.decode() if entry else None

    def save(self, file: Optional[Union[Path, str]]):
        pass
