import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline
import os
from google.cloud import storage
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path-to-save-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--save_steps", type=int, default=None, help="Save a checkpoint every X steps.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--bucket", type=str, default="booth_bucket", help="Google Cloud Bucket to store the data.")


    args = parser.parse_args()
    return args

def upload_images(localpath, bucket_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    for filename in os.listdir(localpath):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            local_file = os.path.join(localpath, filename)
            blob_name = os.path.join(destination_blob_name, filename)
            blob = bucket.blob(blob_name)
            # upload the file to the cloud
            blob.upload_from_filename(local_file)
            print(f"Uploaded {local_file} to gs://{bucket_name}/{blob_name}.")


def main():
    args = parse_args()
    # ----------------------- Loading Models ----------------------- #
    def load_checkpoint(step=None):
        # Load saved models
        outdir = os.path.join(args.output_dir, str(step)) if step else args.output_dir
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            outdir, 
            dtype=jnp.bfloat16,
        )
        return pipeline, params
    
    # ----------------------- Evaluation ----------------------- #
    local_path = "logs/generated_images/"
    os.makedirs(local_path, exist_ok=True)

    pipeline, params = load_checkpoint()
    test_prompt = 'a photo of sks dog is swimming'
    test_rng = jax.random.PRNGKey(0)
    num_samples = jax.device_count()
    prompt = num_samples * [test_prompt]
    prompt_ids = pipeline.prepare_inputs(prompt)

    params = replicate(params)
    test_seed = jax.random.split(test_rng, num_samples)
    prompt_ids = shard(prompt_ids)
    images = pipeline(prompt_ids, params, test_seed, jit=True).images
    images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
    for i, image in enumerate(images):
        image_filename = local_path + f"{prompt}-{i}.jpg"
        image.save(image_filename)
                
    print(f"Inference images saved to {local_path}")
    # ----------------------- Upload images to Google Cloud ----------------------- #
    upload_images(local_path, args.bucket, "generated_images")

if __name__ == "__main__":
    main()