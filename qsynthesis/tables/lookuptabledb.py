from pathlib import Path
from qsynthesis.grammar import TritonGrammar, BvOp
import logging
from enum import IntEnum, Enum
import hashlib
from pony.orm import Database, Required, PrimaryKey, db_session, IntArray, StrArray, count, commit, BindingError, DatabaseError
from pony.orm.dbapiprovider import IntConverter
from typing import Optional, List, Dict, Union, Tuple, TypeVar, Iterable, Generator, Iterator
from time import time, sleep

from qsynthesis.tables.base import LookupTable, Expr, HashType, Hash


db = Database()


class TableEntry(db.Entity):
    hash = PrimaryKey(bytes)
    expression = Required(str)


class Metadata(db.Entity):
    hash_mode = Required(HashType)
    operators = Required(IntArray)


class Variable(db.Entity):
    name = PrimaryKey(str)
    size = Required(int)


class Input(db.Entity):
    id = PrimaryKey(int, auto=True)
    variables = Required(StrArray)
    values = Required(IntArray)


class EnumConverter(IntConverter):
    def validate(self, val, obj=None):
        if not isinstance(val, Enum):
            raise ValueError('Must be an Enum.  Got {}'.format(type(val)))
        return val

    def py2sql(self, val):
        return val.value

    def sql2py(self, value):
        # Any enum type can be used, so py_type ensures the correct one is used to create the enum instance
        return self.py_type(value)


class LookupTableDB(LookupTable):
    def __init__(self, grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        super(LookupTableDB, self).__init__(grammar, inputs, hash_mode, f_name)
        if db.provider_name is None:
            logging.error("LookupTableDB object should be created with a databased already binded !")

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Dict[str, int]], hash_mode: HashType = HashType.RAW) -> 'LookupTableDB':
        db.bind(provider='sqlite', filename=str(filename), create_db=True)
        db.provider.converter_classes.append((Enum, EnumConverter))
        db.generate_mapping(create_tables=True)
        with db_session:
            Metadata(hash_mode=hash_mode, operators=[x.value for x in grammar.ops])
            for name, sz in grammar.vars_dict.items():
                Variable(name=name, size=sz)
            for i in inputs:
                Input(variables=list(i.keys()), values=list(i.values()))
        return LookupTableDB(grammar=grammar, inputs=inputs, hash_mode=hash_mode, f_name=filename)

    @staticmethod
    def load(file: Union[Path, str]) -> 'LookupTableDB':
        global db
        file = file.absolute() if isinstance(file, Path) else Path(file).absolute()
        if db.provider:  # Reset the provider if already set
            db.provider = None
        db.bind(provider='sqlite', filename=str(file), create_db=False)
        db.provider.converter_classes.append((Enum, EnumConverter))
        try:
            db.generate_mapping(create_tables=False)
        except BindingError:
            pass  # Binding was already generated

        with db_session:
            inputs = [{n: v for n, v in zip(i.variables, i.values)} for i in Input.select()]
            vars = [(x.name, x.size) for x in Variable.select()]
            m = Metadata.select().first()
            ops = [BvOp(x) for x in m.operators]
            gr = TritonGrammar(vars=vars, ops=ops)
            return LookupTableDB(grammar=gr, inputs=inputs, hash_mode=HashType(m.hash_mode), f_name=file)

    def add_entry(self, hash: Hash, value: str):
        with db_session:
            TableEntry(hash=hash, expression=value)

    # @db_session
    def add_entries(self, entries: List[Tuple[str, List[int]]], calc_hash=False, chunk_size=10000) -> None:
        count = len(entries)
        if calc_hash:
            hash_fun = lambda x: hashlib.md5(bytes(x)).digest() if self.hash_mode == HashType.MD5 else self.hash
        else:
            hash_fun = lambda x: x
        for step in range(0, count, chunk_size):
            print(f"process {step}/{count}\r", end="")
            with db_session:
                for outs, s in entries[step:step+chunk_size]:
                    TableEntry(hash=hash_fun(outs), expression=s)

    @db_session
    def __iter__(self) -> Iterator[Tuple[Hash, str]]:
        for entry in TableEntry.select():
            yield entry.hash, entry.expression

    @property
    def is_writable(self) -> bool:
        return True

    @property
    def size(self):
        with db_session:
            return count(TableEntry.select())

    @db_session
    def _get_item(self, h: Hash) -> Optional[str]:
        entry = TableEntry.get(hash=h)
        return entry.expression if entry else None

    def save(self, file: Optional[Union[Path, str]]):
        pass
