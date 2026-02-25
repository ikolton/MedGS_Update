

from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
)

from models.scenes.dataset_readers import (
    readMirrorImages,
    readImage
)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Mirror": readMirrorImages,
    "Image": readImage,
}
