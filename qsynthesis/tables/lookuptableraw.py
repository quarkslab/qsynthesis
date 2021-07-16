# built-in libs
from pathlib import Path
import logging
import json

# qsynthesis deps
from qsynthesis.grammar import TritonGrammar
from qsynthesis.tables.base import LookupTable, HashType, Hash
from qsynthesis.types import Input, Optional, List, Dict, Union, Tuple, Iterable


class LookupTableRaw(LookupTable):
    """
    Lookuptable based on raw values encoded directly in binary files.
    This class is not meant to be used as an end table but as an intermediate
    format for fast table generation before conversion in Level-db.
    """

    EXPORT_FILE_CHUNK_LIMIT = 40000000

    def __init__(self, gr: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        """
        Constructor making a lookuptable from a grammar a set of inputs and an hash type.

        :param gr: triton grammar
        :param inputs: List of inputs
        :param hash_mode: type of hash to be used as keys in tables
        :param f_name: file name of the table (when being loaded)
        """
        super(LookupTableRaw, self).__init__(gr, inputs, hash_mode, f_name)

    @property
    def size(self):
        """ Size of the table which is not implemented for such tables """
        raise NotImplementedError()

    def _get_item(self, hash: Hash) -> Optional[str]:
        """ Retrieving an item. Not implemented for such tables """
        raise NotImplementedError()

    def __iter__(self) -> Iterable[Tuple[Hash, str]]:
        """ Iterator of all the entries as an iterator of pair, hash, expression as string """
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
        """
        Add en entry in the table file.

        :param hash: already computed hash to add
        :param value: expression value to add in the table
        """
        with open(str(self.name), "ab") as f:
            f.write(f"{hash},{value}\n".encode())

    def add_entries(self, worklist: List[Tuple[Hash, str]], calc_hash: bool = False) -> None:
        """
        Add the given list of entries in the database. The boolean ``calc_hash`` indicates
        whether hashes are already computed or not. If false the function should hash the
        hash first.

        :param worklist: list of entries to add
        :param calc_hash: whether or not hash should be performed on entries keys
        :returns: None
        """
        import hashlib
        count = len(worklist)

        def do_hash(x):
            return x if not calc_hash else (hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash)

        logging.info("\nExport data")

        f_id = 1
        f_counter = 0
        f = open(self.name, "ab")

        for step in range(0, count, 10000):
            f_counter += 10000
            print(f"process {step}/{count}\r", end="")
            chk_s = b"\n".join(do_hash(outs)+s.encode() for outs, s in worklist[step:step+10000])+b"\n"
            f.write(chk_s)
            if f_counter > self.EXPORT_FILE_CHUNK_LIMIT:
                f.close()
                fname = f"{self.name}.{f_id}"
                LookupTableRaw.create(fname, self.grammar, self.inputs, self.hash_mode)
                f = open(fname, "ab")
                f_id += 1
                f_counter = 0

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTableRaw':
        """
        Load the given lookup table and returns an instance object.

        :param file: Database file to load
        :returns: LookupTableRaw object
        """
        f = Path(file)
        with open(f, 'rb') as f:
            raw = json.loads(f.readline())
            hm = HashType[raw['hash_mode']] if "hash_mode" in raw else HashType.RAW
            gr = TritonGrammar.from_dict(raw)
            inputs = json.loads(f.readline())
            lkp = LookupTableRaw(gr, inputs, hm, f.name)
            return lkp

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, constants: List[int] = []) -> 'LookupTableRaw':
        """
        Create a new empty lookup table with the given initial parameters, grammars, inputs
        and hash_mode.

        :param filename: filename of the table to create
        :param grammar: TritonGrammar object representing variables and operators
        :param inputs: list of inputs on which to perform evaluation
        :param hash_mode: Hashing mode for keys
        :param constants: list of constants used
        :returns: LookupTableRaw instance object
        """
        with open(filename, "wb") as f:
            d = grammar.to_dict()
            d["hash_mode"] = hash_mode.name
            d["constants"] = constants
            f.write(f"{json.dumps(d)}\n".encode())
            f.write(f"{json.dumps(inputs)}\n".encode())
        return LookupTableRaw(grammar, inputs, hash_mode, filename)
