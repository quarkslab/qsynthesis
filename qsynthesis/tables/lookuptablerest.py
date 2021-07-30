# built-in libs
from pathlib import Path
import binascii

# third-party libs
import requests

# qsynthesis deps
from qsynthesis.grammar import TritonGrammar
from qsynthesis.tables.base import InputOutputOracle, HashType, Hash
from qsynthesis.types import Input, Optional, List, Dict, Union, Tuple, Iterator


class InputOutputOracleREST(InputOutputOracle):
    """
    REST-based lookup table. The table given in parameter is meant to be an IP
    address serving the API (that can be served with qsynthesis-table-manager runserver)
    """

    def __init__(self, grammar: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, f_name: str = ""):
        """
        Constructor making a lookuptable from a grammar a set of inputs and an hash type.

        :param grammar: triton grammar
        :param inputs: List of inputs
        :param hash_mode: type of hash to be used as keys in tables
        :param f_name: file name of the table (when being loaded)
        """
        super(InputOutputOracleREST, self).__init__(grammar, inputs, hash_mode, f_name)
        self.session = requests.Session()
        self._size = 0

    @staticmethod
    def create(filename: Union[str, Path], grammar: TritonGrammar, inputs: List[Input], hash_mode: HashType = HashType.RAW, constants: List[int] = []) -> 'InputOutputOracleREST':
        """
        Such tables cannot be created as they are read-only databases.
        """
        raise RuntimeError("REST Lookup tables cannot be created only loaded")

    @staticmethod
    def load(file: Union[Path, str]) -> 'InputOutputOracleREST':
        """
        Load a given table. The function perform a request to retrieve alll
        the basic informations about the table (size, grammar, inputs etc..)
        and to create a completely transparent table.
        """
        res = requests.get(file)
        if res.status_code == 200:
            data = res.json()
            g = TritonGrammar.from_dict(data['grammar'])
            lkp = InputOutputOracleREST(g, data['inputs'], HashType[data['hash_mode']], file)
            lkp._size = data["size"]
            lkp.session.headers['Host'] = file
            lkp._name = file
            return lkp
        else:
            raise ConnectionAbortedError(f"Cannot reach remote server (code:{res.status_code})")

    def add_entry(self, hash: Hash, value: str):
        """
        Function not implemented at the moment. No new expressions can be
        submitted at the moment.
        """
        raise NotImplementedError("REST Lookup Table are read-only at the moment")

    def add_entries(self, entries: List[Tuple[Hash, str]], calc_hash: bool = False, chunk_size: int = 10000) -> None:
        """
        Function not implemented at the moment. REST tables are read-only.
        """
        raise NotImplementedError("REST Lookup tables are read-only at the moment")

    def __iter__(self) -> Iterator[Tuple[Hash, str]]:
        """
        Iterating over all elements of the table is not implemented either.
        """
        raise NotImplementedError("Entries iteration is not implemented")

    @property
    def size(self) -> int:
        """Size of the table (number of entries)"""
        return self._size

    def _get_item(self, h: Hash) -> Optional[str]:
        """
        From a given hash return the associated expression string if found in
        the lookup table through a GET request.

        :param h: hash of the item to get
        :returns: raw expression string if found
        """
        hex_hash = binascii.hexlify(h).decode()
        res = self.session.get(str(self.name) + "/entry/" + hex_hash)
        if res.status_code == 200:
            data = res.json()
            return data['expression'] if data else None
        else:
            raise ConnectionError("REST query did not suceeded correctly")
