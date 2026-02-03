models = {
    "prithvi_300m_sen1floods11": {
        "location": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        "io_processor_plugin": "terratorch_segmentation",
    },
    "prithvi_300m_burnscars": {
        "location": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        "io_processor_plugin": "terratorch_segmentation",
    },
    "terramind_base_flood": {
        "location": "ibm-esa-geospatial/TerraMind-base-Flood",
        "io_processor_plugin": "terratorch_tm_segmentation"
    }
}

inputs = {
    "india_url_in_base64_out": {
        "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/India_900498_S2Hand.tif",
        "indices": [1, 2, 3, 8, 11, 12],
        "data_format": "url",
        "out_data_format": "b64_json",
    },
    "valencia_url_in_base64_out": {
        "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
        "data_format": "url",
        "out_data_format": "b64_json",
    },
    "valencia_url_in_path_out": {
        "image_url": "https://huggingface.co/christian-pinto/Prithvi-EO-2.0-300M-TL-VLLM/resolve/main/valencia_example_2024-10-26.tiff",
        "data_format": "url",
        "out_data_format": "path",
    },
    "burnscars_url_in_base64_out": {
        "image_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",
        "data_format": "url",
        "out_data_format": "b64_json",
    },
    "burnscars_url_in_path_out": {
        "image_url": "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars/resolve/main/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif",
        "data_format": "url",
        "out_data_format": "path",
    },
    "terramind_base_flood_url_in_path_out": {
        "image_url": {
            "DEM": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/EMSR354_23_37LFH_DEM.tif",
            "S1RTC": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/EMSR354_23_37LFH_S1RTC.zarr.zip",
            "S2L2A": "https://huggingface.co/datasets/christian-pinto/TestTerraMindDataset/resolve/main/EMSR354_23_37LFH_S2L2A.zarr.zip",
        },
        "data_format": "url",
        "out_data_format": "path",
    }
}