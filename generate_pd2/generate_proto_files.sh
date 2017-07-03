#!/bin/bash

set -x
set -e

python -m grpc.tools.protoc -I./ --python_out=../pd2/ --grpc_python_out=../pd2/ ./*.proto
