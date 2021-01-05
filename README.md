# ai_flowers
## about
Model trained to identify certain types of flowers
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
   # prediction 
    print(model.get_prediction(path=["data/test/input/melon/images (10).jpeg", "data/test/input/melon/images (1).jpeg","data/test/input/roses/24781114_bc83aa811e_n.jpg"]))
   """
   ['melon', 'melon', 'roses']
   """
```
**project use python 3.8  and tensorflow 2.4**
