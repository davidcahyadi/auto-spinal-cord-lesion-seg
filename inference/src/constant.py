class MODEL_NAMES:
    ANATOMY = "anatomy"
    SEQUENCE = "spine_sequence"
    SPINAL_CORD = "spinal_cord"
    LOCATION_VIEW = "spine_location_view"
    LESION_SEGMENTATION = "lesion_segmentation"
    PRUNED_LESION_SEGMENTATION = "pruned_lesion_segmentation"


class MODEL_LABELS:
    ANATOMY = [141, 142, 143, 144, 145, 146, 147, 148, 149]
    LOCATION_VIEW = [2]
    SEQUENCE = [3]
    SPINAL_CORD = [1]
