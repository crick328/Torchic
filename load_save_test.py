# https://stackoverflow.com/questions/15757213/custom-filetype-in-python-3

import base64
import gzip
import ast
from test9_tensor import Tensor


filename = "prova.rpl"
data = ( repr(Tensor([3.,4.])), repr(Tensor([1.])) )

   
def save(data):
    data = "[{}]".format(data).encode()
    data = base64.b64encode(data)
    return gzip.compress(data)

def load(data):
    data = gzip.decompress(data)
    data = base64.b64decode(data)
    return ast.literal_eval(data.decode())[0]

open(filename, "wb").write(save(data)) # save data

data = load(open(filename, "rb").read()) # load data
print(data)


from zipfile import ZipFile
import json

data = json.dumps(['foo', {'bar': repr(Tensor([3.,4.]))}, {'sos': repr(Tensor([1.]))}])

with ZipFile('foo.filetype', 'w') as myzip:
    myzip.writestr('digest.json', data)

with ZipFile('foo.filetype', 'r') as myzip:
    json_data_read = myzip.read('digest.json')
    newdata = json.loads(json_data_read)
    print(newdata)