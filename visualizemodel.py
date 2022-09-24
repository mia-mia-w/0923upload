from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense, ReLU
from utils import modelloader
from collections import defaultdict
import visualkeras
from PIL import ImageFont

# model = Sequential()
# model.add(Conv1D(16, 3, input_shape=(128, 1)))
# model.add(ReLU())
# model.add(Conv1D(4, 3))
# model.add(ReLU())
# model.add(Flatten())
# # model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(ReLU())
# model.add(Dropout(0.4))
# model.add(Dense(10, activation="softmax"))

model = modelloader.get_model("cnn", input_shape=128, num_class=10)
keras.utils.plot_model(
    model,
    to_file="figures/keras-model.png",
    show_shapes=True,
    show_layer_names=False,
    rankdir="TB",
    dpi=200,
)

font = ImageFont.truetype("arial.ttf", 16)

color_map = defaultdict(dict)
color_map[Flatten]["fill"] = "#e7f5de"
color_map[MaxPooling1D]["fill"] = "#5c3d46"
color_map[Conv1D]["fill"] = "#5c868d"
color_map[ReLU]["fill"] = "#ffb6c1"
color_map[Dense]["fill"] = "#c8d6ca"
color_map[Dropout]["fill"] = "#99bfaa"

visualkeras.layered_view(
    model,
    legend=True,
    font=font,
    scale_xy=3,
    scale_z=1,
    to_file="figures/cnn_model.png",
    color_map=color_map,
    type_ignore=[visualkeras.SpacingDummyLayer, Flatten],
)
