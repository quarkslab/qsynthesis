# built-in libs
from pathlib import Path
import hashlib
import json

# third-party libs
import plyvel

# qsynthesis deps
from qsynthesis.grammar import TritonGrammar, BvOp
from qsynthesis.tables.base import InputOutputOracle, HashType
from qsynthesis.types import Optional, List, Union, Tuple, Iterator, Hash, Input


META_KEY = b"metadatas"
VARS_KEY = b"variables"
INPUTS_KEY = b"inputs"
SIZE_KEY = b"size"


class InputOutputOracleLevelDB(InputOutputOracle):
    """
    Key-Value store oracle database based on Google Level-DB
    """
    def __init__(self, grammar: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        """
        Constructor making a InputOutputOracleLevelDB from a grammar a set of inputs and an hash type.

        :param grammar: triton grammar
        :param inputs: List of inputs
        :param hash_mode: type of hash to be used as keys in tables
        :param f_name: file name of the table (when being loaded)
        """
        super(InputOutputOracleLevelDB, self).__init__(grammar, inputs, hash_mode, f_name)
        self.db = None

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, constants: List[int] = []) -> 'InputOutputOracleLevelDB':
        """
        Create a new empty lookup table with the given initial parameters, grammars, inputs
        and hash_mode.

        :param filename: filename of the table to create
        :param grammar: TritonGrammar object representing variables and operators
        :param inputs: list of inputs on which to perform evaluation
        :param hash_mode: Hashing mode for keys
        :param constants: list of constants used
        :returns: InputOutputOracleLevelDB instance object
        """
        # TODO: If it exists deleting it ?
        db = plyvel.DB(str(filename), create_if_missing=True)

        metas = dict(hash_mode=hash_mode.name, operators=[x.name for x in grammar.ops], constants=constants)
        db.put(META_KEY, json.dumps(metas).encode())
        db.put(VARS_KEY, json.dumps(grammar.vars_dict).encode())
        db.put(INPUTS_KEY, json.dumps(inputs).encode())
        lkp = InputOutputOracleLevelDB(grammar=grammar, inputs=inputs, hash_mode=hash_mode, f_name=filename)
        lkp.db = db
        return lkp

    @staticmethod
    def load(file: Union[Path, str]) -> 'InputOutputOracleLevelDB':
        """
        Load the given lookup table and returns an instance object.

        :param file: Database file to load
        :returns: InputOutputOracleLevelDB object
        """
        db = plyvel.DB(str(file))
        metas = json.loads(db.get(META_KEY))
        ops = [BvOp[x] for x in metas['operators']]
        vrs = list(json.loads(db.get(VARS_KEY)).items())
        inps = json.loads(db.get(INPUTS_KEY))
        grammar = TritonGrammar(vars=vrs, ops=ops)
        lkp = InputOutputOracleLevelDB(grammar=grammar, inputs=inps, hash_mode=HashType[metas['hash_mode']], f_name=file)
        lkp.db = db
        return lkp

    def add_entry(self, hash: Hash, value: str) -> None:
        """
        Put the given hash and value in the leveldb trie.

        :param hash: already computed hash to add
        :param value: expression value to add in the table
        """
        self.db.put(hash, value.encode())
        self.db.put(SIZE_KEY, (str(int(self.db.get(SIZE_KEY))+1)).encode())

    def add_entries(self, entries: List[Tuple[Hash, str]], calc_hash: bool = False, chunk_size: int = 10000, update_count: bool = True) -> None:
        """
        Add the given list of entries in the database. The boolean ``calc_hash`` indicates
        whether hashes are already computed or not. If false the function should hash the
        hash first.

        :param entries: list of entries to add
        :param calc_hash: whether or not hash should be performed on entries keys
        :param chunk_size: size of a chunk for bulk insert in DB
        :param update_count: whether or not to update the count of entries in DB
        :returns: None
        """
        count = len(entries)

        def do_hash(x):
            return x if not calc_hash else (hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash)

        for step in range(0, count, chunk_size):
            with self.db.write_batch(sync=True) as wb:
                for outs, s in entries[step:step+chunk_size]:
                    wb.put(do_hash(outs), s.encode())
        if update_count:
            cur_count = self.db.get(SIZE_KEY)
            new_count = count if cur_count is None else int(cur_count)+count
            self.db.put(SIZE_KEY, str(new_count).encode())

    def __iter__(self) -> Iterator[Tuple[Hash, str]]:
        """ Iterator of all the entries as an iterator of pair, hash, expression as string """
        for key, value in self.db:
            if key not in [META_KEY, VARS_KEY, INPUTS_KEY, SIZE_KEY]:
                yield key, value.decode()

    @property
    def is_writable(self) -> bool:
        """
        Whether the table enable being written (with new expressions)
        Level-db tables are considered to be always writable
        """
        return True

    @property
    def size(self) -> int:
        """Size of the table (number of entries)"""
        return int(self.db.get(SIZE_KEY))

    def _get_item(self, h: Hash) -> Optional[str]:
        """
        From a given hash return the associated expression string if
        found in the lookup table.

        :param h: hash of the item to get
        :returns: raw expression string if found
        """
        entry = self.db.get(h)
        return entry.decode() if entry else None
