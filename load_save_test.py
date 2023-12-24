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





'''
    def zero_grad(self, set_to_none=True):
        for i in range (0, len(self.structure)):
            if type(self.structure[i]) == LLayer:
                self.structure[i].w_matrix.grad = None if set_to_none else np.zeros(self.structure[i].w_matrix.grad.data.shape)
                self.structure[i].b_vector.grad = None if set_to_none else np.zeros(self.structure[i].b_vector.grad.data.shape)
    
    def step(self, lr=1e-3):
        for i in range(0, len(self.structure)):
            if type(self.structure[i]) == LLayer:
                self.structure[i].w_matrix.data = self.structure[i].w_matrix.data - lr*self.structure[i].w_matrix.grad.data
                self.structure[i].b_vector.data = self.structure[i].b_vector.data - lr*self.structure[i].b_vector.grad.data
'''