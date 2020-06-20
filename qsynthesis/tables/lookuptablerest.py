from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, TypeVar, Iterator, Generator
import requests
import binascii

from qsynthesis.grammar import TritonGrammar
from qsynthesis.tables.base import BaseTable, Expr, HashType, Hash


class LookupTableREST(BaseTable):
    def __init__(self, grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        super(LookupTableREST, self).__init__(grammar, inputs, hash_mode, f_name)
        self.session = requests.Session()
        self._size = 0

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'LookupTableDB':
        raise RuntimeError("REST Lookup tables cannot be created only loaded")

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTableREST':
        res = requests.get(file)
        if res.status_code == 200:
            data = res.json()
            g = TritonGrammar.from_dict(data['grammar'])
            lkp = LookupTableREST(g, data['inputs'], HashType[data['hash_mode']], file)
            lkp._size = data["size"]
            lkp.session.headers['Host'] = file
            lkp._name = file
            return lkp
        else:
            raise ConnectionAbortedError(f"Cannot reach remote server (code:{res.status_code})")

    def add_entry(self, hash: Hash, value: str):
        raise NotImplementedError("REST Lookup Table are read-only at the moment")

    def add_entries(self, entries: List[Tuple[str, List[int]]], chunk_size=10000) -> None:
        raise NotImplementedError("REST Lookup tables are read-only at the moment")

    def __iter__(self) -> Iterator[Tuple[Hash, str]]:
        raise NotImplementedError("Entries iteration is not implemented")

    @property
    def size(self):
        return self._size

    def _get_item(self, h: Hash) -> Optional[str]:
        hex_hash = binascii.hexlify(h).decode()
        res = self.session.get(str(self.name) + "/entry/" + hex_hash)
        if res.status_code == 200:
            data = res.json()
            return data['expression'] if data else None
        else:
            raise ConnectionError("REST query did not suceeded correctly")
