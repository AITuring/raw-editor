use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use image::GrayImage;
use ndarray::{Array, IxDyn};
use ort::session::Session;
use tokenizers::Tokenizer;

const ENCODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam_vit_b_01ec64_encoder.onnx?download=true";
const DECODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/sam_vit_b_01ec64_decoder.onnx?download=true";
const ENCODER_FILENAME: &str = "sam_vit_b_01ec64_encoder.onnx";
const DECODER_FILENAME: &str = "sam_vit_b_01ec64_decoder.onnx";
const SAM_INPUT_SIZE: u32 = 1024;
const ENCODER_SHA256: &str = "16ab73d9c824886f0de2938c19df22fb9ec3deebfd0de58e65177e479213d7d1";
const DECODER_SHA256: &str = "85d0d672cf5b7fe763edcde429e5533e62f674af4b15c7d688b7673b0ef00bf7";

const U2NETP_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/u2net.onnx?download=true";
const U2NETP_FILENAME: &str = "u2net.onnx";
const U2NETP_INPUT_SIZE: u32 = 320;
const U2NETP_SHA256: &str = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491";

const SKYSEG_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/skyseg-u2net.onnx?download=true";
const SKYSEG_FILENAME: &str = "skyseg_u2net.onnx";
const SKYSEG_LEGACY_FILENAME: &str = "skyseg-u2net.onnx";
const SKYSEG_INPUT_SIZE: u32 = 320;
const SKYSEG_SHA256: &str = "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39";

const CLIP_MODEL_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_model.onnx?download=true";
const CLIP_MODEL_FILENAME: &str = "clip_model.onnx";
const CLIP_TOKENIZER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_tokenizer.json?download=true";
const CLIP_TOKENIZER_FILENAME: &str = "clip_tokenizer.json";
const CLIP_MODEL_SHA256: &str = "57879bb1c23cdeb350d23569dd251ed4b740a96d747c529e94a2bb8040ac5d00";

const DENOISE_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/nind_denoise_utnet_684.onnx?download=true";
const DENOISE_FILENAME: &str = "nind_denoise_utnet_684.onnx";
const DENOISE_SHA256: &str = "ee3586279d514df557ff3f7dec6df37fafc51ba5d3a3435b2cc9ac2d9017e7fe";

const LAMA_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/lama_fp16.onnx?download=true";
const LAMA_FILENAME: &str = "lama_fp16.onnx";
const LAMA_SHA256: &str = "2d6be6277c400d6f1b91819737f7c3da935e5c63d1b521d393be1196a2bfa82c";

const DEPTH_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/depth_anything_v2_vits.onnx?download=true";
const DEPTH_FILENAME: &str = "depth_anything_v2_vits.onnx";
const DEPTH_INPUT_SIZE: u32 = 518;
const DEPTH_SHA256: &str = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be";

pub struct AiModels {
    pub sam_encoder: Mutex<Session>,
    pub sam_decoder: Mutex<Session>,
    pub u2netp: Mutex<Session>,
    pub sky_seg: Mutex<Session>,
    pub depth_anything: Mutex<Session>,
}

pub struct ClipModels {
    pub model: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

#[derive(Clone)]
pub struct ImageEmbeddings {
    pub path_hash: String,
    pub embeddings: Array<f32, IxDyn>,
    pub original_size: (u32, u32),
}

#[derive(Clone)]
pub struct CachedDepthMap {
    pub path_hash: String,
    pub depth_image: GrayImage,
    pub original_size: (u32, u32),
}

pub struct AiState {
    pub models: Option<Arc<AiModels>>,
    pub denoise_model: Option<Arc<Mutex<Session>>>,
    pub clip_models: Option<Arc<ClipModels>>,
    pub lama_model: Option<Arc<Mutex<Session>>>,
    pub embeddings: Option<ImageEmbeddings>,
    pub depth_map: Option<CachedDepthMap>,
}

mod model_files;

use model_files::{
    download_and_verify_model, ensure_onnx_runtime, get_models_dir, persist_downloaded_asset,
};

mod model_sessions;

pub use model_sessions::{
    get_or_init_ai_models, get_or_init_clip_models, get_or_init_denoise_model,
    get_or_init_lama_model,
};

mod denoise;

pub use denoise::run_ai_denoise;

mod lama_inpainting;

pub use lama_inpainting::run_lama_inpainting;

mod sam;

pub use sam::{generate_image_embeddings, run_sam_decoder};

mod segmentation;

pub use segmentation::{run_depth_anything_model, run_sky_seg_model, run_u2netp_model};

mod mask_parameters;

pub use mask_parameters::{
    AiDepthMaskParameters, AiForegroundMaskParameters, AiSkyMaskParameters, AiSubjectMaskParameters,
};

async fn download_model(url: &str, dest: &Path) -> Result<()> {
    let response = reqwest::get(url).await?.error_for_status()?;
    let bytes = response.bytes().await?;
    persist_downloaded_asset(dest, &bytes)
}
