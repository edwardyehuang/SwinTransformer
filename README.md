# SwinTransformer

SWIN Transformer in TensorFlow 2.x

Size free for inputs to support tasks like Image Segmentation

[Offical Pytorch repo](https://github.com/microsoft/Swin-Transformer).

## Weights
All weights below are fully tested.

|Name|links|
|---|---|
|Swin Large 384|[weiyun](https://share.weiyun.com/JIRlKVKc)|
|Swin Base 384|[weiyun](https://share.weiyun.com/NXAosmtC)|
|Swin Tiny 224|[weiyun](https://share.weiyun.com/PQ5dXCoE)|

## Usage

```python
def load_h5_weight (model, path, skip_mismatch = False):

    with h5py.File(path, 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        
        layers = get_all_layers(model)
        
        load_weights_from_hdf5_group_by_name(f, model.layers, skip_mismatch = skip_mismatch)



```
