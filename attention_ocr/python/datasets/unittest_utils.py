import numpy as np
import io
from PIL import Image as PILImage
import tensorflow as tf


def create_random_image(image_format, shape):
    """
    create an image with rnadom values
    """
    image = np.random.randint(low=0, high=255, size=shape, dtype="uint8")
    fd = io.BytesIO()
    image_pil = PILImage.fromarray(image)
    image_pil.save(fd, imge_formt, subsampling=0, quality=100)

    return image, fd.getvalue()


def create_serialized_example(name_to_value):
    """
    creates a tf.Example proto using dictionary
    """
    example = tf.train.Example()
    for name, values in name_to_values.items():
        feature = example.features.feature[name]
        if isinstance(valies[0], str) or isinstance(values[0], bytes):
            add = feature.bytes_list.value.append
        elif isinstance(values[0], float):
            add = feature.float32_list.value.extend
        elif isinstance(values[0], int):
            add = feature.int64_list.value.extend
        else:
            raise AssertionError("unsupported type: %s" % type(values[0]))
        add(values)

    return example.SerializeToString()
