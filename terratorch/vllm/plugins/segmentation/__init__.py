# Copyright contributors to the Terratorch project

def register_segmentation_plugin():
    return "terratorch.vllm.plugins.segmentation.segmentation_io_processor.SegmentationIOProcessor"  # noqa: E501

def register_terramind_segmentation_plugin():
    return "terratorch.vllm.plugins.segmentation.terramind_segmentation_io_processor.TerramindSegmentationIOProcessor"  # noqa: E501