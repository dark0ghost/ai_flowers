# ai_flowers

# use 
```python
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import InteractiveSession
from main import Flowers

if __name__ == '__main__':
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model = Flowers()
    model.compile().fit(epochs=50).save_model()
    # or 
    model.load_model().fit(epochs=50).evaluate(model.image_ds,model.image_label_ds)
```
**project use python 3.8  and tensorflow 2.4**
