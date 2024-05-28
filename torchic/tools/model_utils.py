import base64
import gzip
#import ast

# metti qua roba per quantizzare

def save_model_type1(model, filename="type1"):
    data = "{}".format(repr(model)).encode()
    data = base64.b64encode(data)
    open(f"{filename}.rpl", "wb").write(gzip.compress(data)) # save data

def load_model_type1(filename):
    data = open(f"{filename}.rpl", "rb").read()
    data = gzip.decompress(data)
    data = base64.b64decode(data)
    #return ast.literal_eval(data.decode())[0]#data.decode()
    return eval(data.decode())