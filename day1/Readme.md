# Keras
## GPU support
`pip install tensorflow<2.11` tensorflow version 2.10 or below
## Saving & Loading
```py
# Saving
model.save('model.full.h5')

# Loading
from keras.models import load_model
loaded_model = load_model('model.full.h5')
```
# PyTorch
## GPU support
Cuda 11.7 or build from source -> [video link](https://www.youtube.com/watch?v=sGWLjbn5cgs)
## Saving & Loading
```py
import torch
model = torch.Model...()
PATH = "MODEL.pt"
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

```
# TensorFlow
## GPU support
`pip install tensorflow<2.11` tensorflow version 2.10 or below
## Saving & Loading
```py

# Save: 

tf.saved_model.save(model, path_to_dir)
# Load: 
model = tf.saved_model.load(path_to_dir)
```
# Sci-Kit Learn
## GPU support
There is support for training on GPU 
## Saving & Loading
```py
import pickle
with open(filename, 'wb') as wb:
    pickle.dump(model, wb)
with open(filename, 'rb') as rb:
    loaded_model = pickle.load(rb)
```
# PySpark
## GPU support
TODO...
## Saving & Loading
```py
import pickle
with open(filename, 'wb') as wb:
    pickle.dump(model, wb)
with open(filename, 'rb') as rb:
    loaded_model = pickle.load(rb)
```
# Tips for Training
- Using checkpoints
- Be aware of over-fitting (use a validation set while training)
- Initialize a repo and commit before training (use versions for faster recalling)
- If possible use GPU for training
- Be aware of training time
