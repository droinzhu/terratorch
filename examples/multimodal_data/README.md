## Multimodal Data

To load and work with multimodal data, TerraTorch provides the `GenericMultiModalDataModule`. The [TerraMind repository](https://github.com/IBM/terramind) contains several TerraTorch examples that show how to use the multimodal datamodule together with the TerraMind backbone.

### Examples

This folder additionally includes a sample YAML config for a multimodal segmentation task using Prithvi. Instructions for downloading the required Sen1Floods11 dataset can be found in the `examples/segmentation` folder, which contains a demo notebook for unimodal segmentation.

- Prithvi multimodal segmentation: [`multimodal_segmentation_sen1floods11.yaml`](multimodal_segmentation_sen1floods11.yaml)