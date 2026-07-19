## How a Tensor operation works
Let's say that I declare two tensors t1 and t2.
By default are considered **leaf tensors** because are instantiated by the user.
A tensor is not a leaf _only_ when a grad_fn is assigned.
