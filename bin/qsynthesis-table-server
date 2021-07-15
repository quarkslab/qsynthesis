#!/usr/bin/env python3

import click
import logging
import json
from binascii import unhexlify

import plyvel

META_KEY = b"metadatas"
VARS_KEY = b"variables"
INPUTS_KEY = b"inputs"
SIZE_KEY = b"size"


@click.command()
@click.argument('table_file', type=click.Path(exists=True))
@click.option('-p', '--port', type=int, default=8080, help="Service port to listen on")
def runserver(table_file, port):
    """Run the REST API to serve a given table database"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        table = plyvel.DB(str(table_file))
        metas = json.loads(table.get(META_KEY))
        vrs = list(json.loads(table.get(VARS_KEY)).items())
        inps = json.loads(table.get(INPUTS_KEY))
    except IOError:
        logging.error("Lookup table database is invalid (or already opened)")
        return

    try:
        from fastapi import FastAPI, Query
        import uvicorn
    except ImportError:
        logging.critical("Cannot import fastapi or uvicorn (pip3 install fastapi uvicorn)")
        raise click.Abort("")

    app = FastAPI()


    @app.get("/")
    def read_root():
        return {'size': table.get(SIZE_KEY),
                'hash_mode': metas['hash_mode'],
                'inputs': inps,
                'grammar': {'vars': vrs, 'operators': metas['operators']}}

    @app.get("/entry/{hash}")
    def read_item(hash: str = Query(None, min_length=32, max_length=32, regex="^[0-9a-z]+$")):
        decoded = unhexlify(hash)
        entry = table.get(decoded)
        print(entry)
        if entry:
            return {"hash": hash, "expression": entry}
        else:
            return {}

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    runserver()
