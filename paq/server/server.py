#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from http.server import BaseHTTPRequestHandler, HTTPServer, HTTPStatus
import json
import argparse
from paq.retrievers.retrieve import _load_index_if_exists, load_retriever, load_jsonl, run_queries
import logging

logger = logging.getLogger(__name__)


class http_server:
    def __init__(self, index, model, tokenizer, qas_to_retrieve_from, fp16):
        server = HTTPServer(("", 1359), WebServerHandler)
        server.index = index
        server.model = model
        server.tokenizer = tokenizer
        server.qas_to_retrieve_from = qas_to_retrieve_from
        server.fp16 = fp16
        logging.info("Web Server running:")
        server.serve_forever()


class WebServerHandler(BaseHTTPRequestHandler):

    # POST echoes the message adding a JSON field
    def do_POST(self):
        datalen = int(self.headers["Content-Length"])
        data = self.rfile.read(datalen)
        obj = json.loads(data)
        logger.info("Got object: {}".format(obj))

        if "query" in obj and "k" in obj:
            qas_to_answer = [{'question': obj['query']}]
            result = run_queries(
                self.server.model,
                self.server.tokenizer,
                self.server.qas_to_retrieve_from,
                qas_to_answer,
                top_k=obj['k'],
                index=self.server.index,
                batch_size=1,
                fp16=args.fp16,
                n_queries_to_parallelize=1
            )
            logger.info("result: " + json.dumps(result))

            # send the message back
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"result": result}).encode())
            return
        else:
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.end_headers()


def main(args):
    qas_to_retrieve_from = load_jsonl(args.qas_to_retrieve_from, memory_friendly=args.memory_friendly_parsing)
    index = _load_index_if_exists(
        args.faiss_index_path,
        args.precomputed_embeddings_dir,
        n_vectors_to_load=len(qas_to_retrieve_from),
        memory_friendly=args.memory_friendly_parsing,
        efsearch=args.faiss_efsearch
    )
    model, tokenizer = load_retriever(args.model_name_or_path)

    try:
        server = http_server(index, model, tokenizer, qas_to_retrieve_from, args.fp16)
        logging.info("Web Server running:")
    except KeyboardInterrupt:
        logging.info(" ^C entered, stopping web server....")
        server.server.socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Server that wraps Retrieval functionality")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='path to HF model dir')
    parser.add_argument('--qas_to_retrieve_from', type=str, required=True,
                        help="path to QA-pairs to retrieve answers from in jsonl format")
    parser.add_argument('--top_k', type=int, default=50, help="top K QA-pairs to retrieve for each input question")
    parser.add_argument('--faiss_index_path', default=None, type=str,
                        help="Path to faiss index, if retrieving from a faiss index")
    parser.add_argument('--precomputed_embeddings_dir', default=None, type=str,
                        help="path to a directory of vector embeddings if retrieving from raw embeddign vectors")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding questions for querying')
    parser.add_argument('--n_queries_to_parallelize', type=int, default=256, help="query batch size")
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--memory_friendly_parsing', action='store_true',
                        help='Pass this to load files more slowly, but save memory')
    parser.add_argument('--faiss_efsearch', type=int, default=128,
                        help='EFSearch searchtime parameter for hnsw , higher is more accuate but slower')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    main(args)
