import base64
import imagehash
import json
import os
from PIL import Image
import pytest
import requests
import tempfile
import uuid

from .utils import VLLMServer
from .config import models, inputs

# Each model has a different output depending also on the plugin
models_output = {
    "prithvi_300m_sen1floods11": {
        "india_url_in_base64_out": "f7dc282de2c36942",
        "valencia_url_in_base64_out": "aa6d92ad25926a5e",
        "valencia_url_in_path_out": "aa6d92ad25926a5e",
    },
    "prithvi_300m_burnscars": {
        "burnscars_url_in_base64_out": "c17c4f602ea7b616",
        "burnscars_url_in_path_out": "c17c4f602ea7b616"
    },
    "terramind_base_flood": {
        "terramind_base_flood_url_in_path_out": "dc25fd8e31cc0a72",
    }
}

tests_per_model = [(model, input) for model in models_output.keys() for input in models_output[model].keys() ]

@pytest.fixture(scope="session")
def server():
    class Holder:
        instance = None
        tmpdir = None
        model_name = None

        def _delete_server(self):
            if self.instance:
                self.instance.kill_proc()
                self.tmpdir.cleanup()

        def init_server(self, model_name, **kwargs):
            self._delete_server()
            self.tmpdir = tempfile.TemporaryDirectory()
            plugin_config = {"output_path": self.tmpdir.name}
            server_envs = {"TERRATORCH_SEGMENTATION_IO_PROCESSOR_CONFIG": json.dumps(plugin_config),
                            "VLLM_LOGGING_LEVEL": "DEBUG"}
            # 10 minutes timeout for vLLM to start
            self.instance = VLLMServer(model_name, server_envs=server_envs, timeout=600, **kwargs)
            self.model_name = model_name
            return self

    return Holder()

@pytest.fixture
def get_server(server):
    def _get(model_name, **kwargs):
        if server.instance is None or server.model_name != model_name:
            return server.init_server(model_name=model_name, **kwargs)
        return server
    return _get

@pytest.mark.parametrize(
    "model_name, input_name", tests_per_model
)
def test_serving_segmentation_plugin(get_server, model_name, input_name):
    model = models[model_name]["location"]
    io_processor_plugin = models[model_name]["io_processor_plugin"]
    input = inputs[input_name]

    image_url = input["image_url"]
    server_args = [
        "--skip-tokenizer-init",
        "--enforce-eager",
        # This is just in case the test ends up with a GPU of less memory than an A100-80GB.
        # Just to avoid OOMing in the CI
        "--max-num-seqs",
        "8",
        "--io-processor-plugin",
        io_processor_plugin,
        "--model-impl",
        "terratorch",
        "--enable-mm-embeds"
    ]

    server = get_server(model, server_args=server_args)
    request_payload = {
        "data": {
            "data": image_url, 
            "data_format": input["data_format"],
            "out_data_format": input["out_data_format"],
            "image_format": ""
        },
        "model": model,
        "softmax": False
    }

    if "indices" in input:
        request_payload["data"]["indices"] = input["indices"]

    ret = requests.post("http://localhost:8000/pooling", json=request_payload)
    assert ret.status_code == 200

    response = ret.json()

    if request_payload["data"]["out_data_format"] == "b64_json":
        decoded_image = base64.b64decode(response["data"]["data"])

        file_name = os.path.join(server.tmpdir.name, f"{uuid.uuid4()}.tiff")

        with open(file_name, "wb") as f:
            f.write(decoded_image)
    else:
        file_name = response["data"]["data"]
        
        # Verify the filename contains _pred suffix for path output
        base_filename = os.path.basename(file_name)
        assert "_pred" in base_filename, \
            f"Expected filename to contain '_pred' suffix, but got {base_filename}"
        
        # For URL and path inputs, verify the filename is derived from the input
        if input["data_format"] in ["url", "path"]:
            image_source = image_url
            # Handle both string URLs/paths and dict URLs (for terramind)
            if isinstance(image_source, dict):
                # For TerraMind models, use DEM URL for filename verification
                image_source = image_source.get("DEM", next(iter(image_source.values())))
            
            if isinstance(image_source, str):
                # Extract expected base name from URL or path
                if input["data_format"] == "url":
                    from urllib.parse import urlparse, unquote
                    parsed_url = urlparse(image_source)
                    source_filename = os.path.basename(unquote(parsed_url.path))
                else:  # path
                    source_filename = os.path.basename(image_source)
                
                name_without_ext, ext = os.path.splitext(source_filename)
                
                # For TerraMind models, remove _DEM suffix from expected pattern
                if isinstance(image_url, dict) and name_without_ext.endswith("_DEM"):
                    name_without_ext = name_without_ext[:-4]  # Remove _DEM
                
                # Check that the output filename is based on the input filename
                expected_pattern = f"{name_without_ext}_pred"
                assert expected_pattern in base_filename, \
                    f"Expected filename to contain '{expected_pattern}', but got {base_filename}"

    # I am using perceptual hashing to absorb minimal variations between the one calculated "at home"
    # and the one generated in the test
    image_hash = str(imagehash.phash(Image.open(file_name)))

    assert image_hash == models_output[model_name][input_name]


@pytest.mark.parametrize(
    "model_name, input_name", tests_per_model
)
def test_serving_segmentation_plugin_with_custom_out_path(get_server, model_name, input_name):
    """Test that custom out_path in request payload is respected"""
    model = models[model_name]["location"]
    io_processor_plugin = models[model_name]["io_processor_plugin"]
    input = inputs[input_name]

    # Skip test if output format is not 'path'
    if input["out_data_format"] != "path":
        pytest.skip("Test only applicable for 'path' output format")

    image_url = input["image_url"]
    server_args = [
        "--skip-tokenizer-init",
        "--enforce-eager",
        "--max-num-seqs",
        "8",
        "--io-processor-plugin",
        io_processor_plugin,
        "--model-impl",
        "terratorch",
        "--enable-mm-embeds"
    ]

    server = get_server(model, server_args=server_args)
    
    # Create a custom output directory
    with tempfile.TemporaryDirectory() as custom_out_dir:
        request_payload = {
            "data": {
                "data": image_url,
                "data_format": input["data_format"],
                "out_data_format": input["out_data_format"],
                "out_path": custom_out_dir,
                "image_format": ""
            },
            "model": model,
            "softmax": False
        }

        if "indices" in input:
            request_payload["data"]["indices"] = input["indices"]

        ret = requests.post("http://localhost:8000/pooling", json=request_payload)
        assert ret.status_code == 200

        response = ret.json()
        file_name = response["data"]["data"]
        
        # Verify the file was saved in the custom directory
        assert os.path.dirname(file_name) == custom_out_dir, \
            f"Expected file to be in {custom_out_dir}, but got {os.path.dirname(file_name)}"
        
        # Verify the file exists
        assert os.path.exists(file_name), f"Output file {file_name} does not exist"
        
        # Verify the content is correct using perceptual hashing
        image_hash = str(imagehash.phash(Image.open(file_name)))
        assert image_hash == models_output[model_name][input_name]
