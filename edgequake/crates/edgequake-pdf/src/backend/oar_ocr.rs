//! OAR-OCR backend: PP-StructureV3 pipeline tuned for academic math-heavy papers.
//!
//! Uses the [`oar-ocr`](https://crates.io/crates/oar-ocr) Rust crate. Pipeline stages:
//!   - Layout detection: **PP-DocLayout_plus-L** (20 classes incl. `algorithm`, `formula`,
//!     `table`). NOTE: We also tried PP-DocLayoutV2 because it has richer formula classes
//!     (inline_formula/display_formula) that would help recognize math inside algorithm
//!     boxes — but OAR-OCR 0.6.3's `OARStructureBuilder::layout_model_name()` has no
//!     V2 match arm (`structure.rs:676-691`), so V2 ONNX output gets decoded with
//!     _plus-L post-processor → garbage detections. Revisit once upstream adds V2 routing.
//!   - Region detection: **PP-DocBlockLayout** — multi-column / block grouping so
//!     layout elements are sorted by column first, then position within. Critical for
//!     2-column academic papers where reading order across columns would otherwise
//!     interleave text line by line.
//!   - Text detection/recognition: PP-OCRv5 server variants (higher accuracy than
//!     mobile — important for small subscripts/superscripts).
//!   - Formula recognition: PP-FormulaNet_plus-L (LaTeX output for regions classified
//!     as `formula`). Algorithm-box math stays as plain OCR text because _plus-L
//!     classifies those boxes as a single `algorithm` region.
//!   - Table recognition: PP-LCNet table classifier → SLANet_plus (wireless) /
//!     SLANeXt_wired (wired) structure models + RT-DETR-L cell detection → HTML tables.
//!
//! Page rendering uses the pure-Rust [`hayro`](https://crates.io/crates/hayro) crate.
//!
//! ONNX models are downloaded on first use to:
//!   `$EDGEQUAKE_OAR_OCR_MODEL_DIR` (env var, default `$HOME/.cache/edgequake/oar-ocr-models`)
//!
//! Page render scale defaults to 3.0 (≈216 DPI). Override with `EDGEQUAKE_OAR_OCR_SCALE`.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use oar_ocr::oarocr::OARStructureBuilder;
use tracing::{info, warn};

use super::{PdfConversionConfig, PdfConverter};
use crate::error::PdfConversionError;

/// OAR-OCR powered PDF → Markdown converter.
#[derive(Debug, Default)]
pub struct OarOcrConverter;

const LAYOUT_MODEL: &str = "pp-doclayoutv2.onnx";
const LAYOUT_MODEL_NAME: &str = "pp-doclayoutv2";
const REGION_MODEL: &str = "pp-docblocklayout.onnx";
const REGION_MODEL_NAME: &str = "pp-docblocklayout";
const TEXT_DET_MODEL: &str = "pp-ocrv5_server_det.onnx";
const TEXT_REC_MODEL: &str = "pp-ocrv5_server_rec.onnx";
const TEXT_DICT: &str = "ppocrv5_dict.txt";
const FORMULA_MODEL: &str = "pp-formulanet_plus-l.onnx";
const FORMULA_TOKENIZER: &str = "unimernet_tokenizer.json";
const TABLE_CLS_MODEL: &str = "pp-lcnet_x1_0_table_cls.onnx";
const WIRED_STRUCTURE_MODEL: &str = "slanext_wired.onnx";
const WIRELESS_STRUCTURE_MODEL: &str = "slanet_plus.onnx";
const WIRED_CELL_MODEL: &str = "rt-detr-l_wired_table_cell_det.onnx";
const WIRELESS_CELL_MODEL: &str = "rt-detr-l_wireless_table_cell_det.onnx";
const TABLE_STRUCTURE_DICT: &str = "table_structure_dict_ch.txt";

const MODEL_URLS: &[(&str, &str)] = &[
    (LAYOUT_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayoutv2.onnx"),
    (REGION_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-docblocklayout.onnx"),
    (TEXT_DET_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_server_det.onnx"),
    (TEXT_REC_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_server_rec.onnx"),
    (TEXT_DICT, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_dict.txt"),
    (FORMULA_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-l.onnx"),
    (FORMULA_TOKENIZER, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet_tokenizer.json"),
    (TABLE_CLS_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_table_cls.onnx"),
    (WIRED_STRUCTURE_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanext_wired.onnx"),
    (WIRELESS_STRUCTURE_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanet_plus.onnx"),
    (WIRED_CELL_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-l_wired_table_cell_det.onnx"),
    (WIRELESS_CELL_MODEL, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-l_wireless_table_cell_det.onnx"),
    (TABLE_STRUCTURE_DICT, "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/table_structure_dict_ch.txt"),
];

#[async_trait]
impl PdfConverter for OarOcrConverter {
    async fn convert(
        &self,
        pdf_bytes: &[u8],
        _config: &PdfConversionConfig,
    ) -> Result<String, PdfConversionError> {
        let pdf_bytes = pdf_bytes.to_vec();

        tokio::task::spawn_blocking(move || {
            // 1. Resolve model directory and download models if missing.
            let model_dir = model_cache_dir()?;
            ensure_models_present(&model_dir)?;

            // 2. Render PDF pages to images using hayro (pure Rust).
            let images = render_pdf_to_images(&pdf_bytes)?;
            if images.is_empty() {
                return Err(PdfConversionError::EmptyOutput(
                    "OAR-OCR: PDF has no pages",
                ));
            }
            let page_count = images.len();
            info!(pages = page_count, "OAR-OCR: rendered PDF pages");

            // 3. Build the structure analyzer: layout + region + OCR + formulas + tables.
            let layout_path = model_dir.join(LAYOUT_MODEL);
            let region_path = model_dir.join(REGION_MODEL);
            let det_path = model_dir.join(TEXT_DET_MODEL);
            let rec_path = model_dir.join(TEXT_REC_MODEL);
            let dict_path = model_dir.join(TEXT_DICT);
            let formula_path = model_dir.join(FORMULA_MODEL);
            let formula_tokenizer_path = model_dir.join(FORMULA_TOKENIZER);
            let table_cls_path = model_dir.join(TABLE_CLS_MODEL);
            let wired_structure_path = model_dir.join(WIRED_STRUCTURE_MODEL);
            let wireless_structure_path = model_dir.join(WIRELESS_STRUCTURE_MODEL);
            let wired_cell_path = model_dir.join(WIRED_CELL_MODEL);
            let wireless_cell_path = model_dir.join(WIRELESS_CELL_MODEL);
            let table_dict_path = model_dir.join(TABLE_STRUCTURE_DICT);

            info!(
                formula_model = %formula_path.display(),
                formula_tokenizer = %formula_tokenizer_path.display(),
                formula_exists = formula_path.exists(),
                tokenizer_exists = formula_tokenizer_path.exists(),
                "OAR-OCR: building analyzer with formula recognition"
            );

            let analyzer = OARStructureBuilder::new(&layout_path)
                .layout_model_name(LAYOUT_MODEL_NAME)
                .with_region_detection(&region_path)
                .region_model_name(REGION_MODEL_NAME)
                .with_ocr(&det_path, &rec_path, &dict_path)
                .with_formula_recognition(&formula_path, &formula_tokenizer_path, "pp_formulanet")
                .with_table_classification(&table_cls_path)
                .with_wired_table_structure(&wired_structure_path)
                .with_wireless_table_structure(&wireless_structure_path)
                .with_wired_table_cell_detection(&wired_cell_path)
                .with_wireless_table_cell_detection(&wireless_cell_path)
                .table_structure_dict_path(&table_dict_path)
                .build()
                .map_err(|e| {
                    PdfConversionError::Backend(format!("OAR-OCR builder failed: {e}"))
                })?;

            // 4. Run inference across all pages. predict_images returns Vec<Result<_, _>>.
            let page_results = analyzer.predict_images(images);

            // 5. Concatenate per-page markdown. Skip individual page failures with a warning.
            let mut markdown = String::new();
            let mut succeeded = 0usize;
            for (i, res) in page_results.into_iter().enumerate() {
                match res {
                    Ok(r) => {
                        if !markdown.is_empty() {
                            markdown.push_str("\n\n");
                        }
                        markdown.push_str(&r.to_markdown());
                        succeeded += 1;
                    }
                    Err(e) => {
                        warn!(page = i + 1, error = %e, "OAR-OCR: page failed, skipping");
                    }
                }
            }

            if markdown.trim().is_empty() {
                return Err(PdfConversionError::EmptyOutput(
                    "OAR-OCR returned no text — PDF may be encrypted or unsupported",
                ));
            }

            info!(
                markdown_len = markdown.len(),
                pages_succeeded = succeeded,
                pages_total = page_count,
                "OAR-OCR conversion completed"
            );

            Ok(markdown)
        })
        .await
        .map_err(|e| PdfConversionError::Internal(e.to_string()))?
    }

    fn backend_name(&self) -> &'static str {
        "oarocr"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model management
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn model_cache_dir() -> Result<PathBuf, PdfConversionError> {
    if let Ok(dir) = std::env::var("EDGEQUAKE_OAR_OCR_MODEL_DIR") {
        if !dir.is_empty() {
            let p = PathBuf::from(dir);
            fs::create_dir_all(&p).map_err(|e| {
                PdfConversionError::Internal(format!(
                    "Failed to create OAR-OCR model dir: {e}"
                ))
            })?;
            return Ok(p);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".cache/edgequake/oar-ocr-models");
        fs::create_dir_all(&p).map_err(|e| {
            PdfConversionError::Internal(format!(
                "Failed to create OAR-OCR model dir: {e}"
            ))
        })?;
        return Ok(p);
    }
    let p = PathBuf::from("/tmp/edgequake-oar-ocr-models");
    fs::create_dir_all(&p).map_err(|e| {
        PdfConversionError::Internal(format!("Failed to create OAR-OCR model dir: {e}"))
    })?;
    Ok(p)
}

fn ensure_models_present(dir: &Path) -> Result<(), PdfConversionError> {
    static DOWNLOAD_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    let mutex = DOWNLOAD_LOCK.get_or_init(|| std::sync::Mutex::new(()));
    let _guard = mutex.lock().unwrap_or_else(|e| e.into_inner());

    for (name, url) in MODEL_URLS {
        let dest = dir.join(name);
        if dest.exists() {
            continue;
        }
        info!(model = name, "OAR-OCR: downloading model (first-time setup)");
        download_file(url, &dest)?;
        info!(
            model = name,
            bytes = dest.metadata().map(|m| m.len()).unwrap_or(0),
            "OAR-OCR: model downloaded"
        );
    }
    Ok(())
}

fn download_file(url: &str, dest: &Path) -> Result<(), PdfConversionError> {
    let tmp_dest = dest.with_extension("downloading");

    let mut response = ureq::get(url)
        .call()
        .map_err(|e| PdfConversionError::Backend(format!("Download failed ({url}): {e}")))?;

    let mut file = fs::File::create(&tmp_dest).map_err(|e| {
        PdfConversionError::Internal(format!("Failed to create temp download: {e}"))
    })?;

    let mut buf = [0u8; 65536];
    let mut reader = response.body_mut().as_reader();
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| PdfConversionError::Backend(format!("Download read error: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| {
            PdfConversionError::Internal(format!("Failed to write download: {e}"))
        })?;
    }
    file.sync_all().ok();
    drop(file);

    fs::rename(&tmp_dest, dest).map_err(|e| {
        PdfConversionError::Internal(format!("Failed to finalize download: {e}"))
    })?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF rendering (hayro, pure Rust)
// Port of oar-ocr's example code in examples/utils/pdf.rs
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) fn render_pdf_to_images(
    pdf_bytes: &[u8],
) -> Result<Vec<image::RgbImage>, PdfConversionError> {
    use hayro::hayro_syntax::Pdf;

    let pdf_data: Arc<Vec<u8>> = Arc::new(pdf_bytes.to_vec());
    let pdf = Pdf::new(pdf_data)
        .map_err(|e| PdfConversionError::Backend(format!("hayro parse failed: {e:?}")))?;

    let pages = pdf.pages();
    if pages.is_empty() {
        warn!("OAR-OCR: PDF has zero pages");
        return Ok(Vec::new());
    }

    let mut images = Vec::with_capacity(pages.len());
    for (i, page) in pages.iter().enumerate() {
        match render_single_page(page) {
            Ok(img) => images.push(img),
            Err(e) => {
                warn!(page = i + 1, error = %e, "OAR-OCR: failed to render page, skipping");
            }
        }
    }
    Ok(images)
}

fn render_single_page(page: &hayro::hayro_syntax::page::Page) -> Result<image::RgbImage, String> {
    use hayro::RenderSettings;

    let media_box = page.media_box();
    let width = (media_box.x1 - media_box.x0) as f32;
    let height = (media_box.y1 - media_box.y0) as f32;

    if width <= 0.0 || height <= 0.0 {
        return Err(format!("Invalid page size: {}x{}", width, height));
    }

    // 3x scale (~216 DPI) — small enough to fit dense pages in memory, large enough
    // for PP-FormulaNet to read subscripts/superscripts. Override via env for tuning.
    let scale = std::env::var("EDGEQUAKE_OAR_OCR_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|s| *s > 0.0 && *s <= 8.0)
        .unwrap_or(3.0f32);

    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        ..Default::default()
    };

    let interpreter_settings = hayro::hayro_interpret::InterpreterSettings::default();
    let pixmap = hayro::render(page, &interpreter_settings, &settings);

    // Convert RGBA → RGB.
    let rgba = pixmap.data_as_u8_slice();
    let mut rgb = Vec::with_capacity((pixmap.width() as usize) * (pixmap.height() as usize) * 3);
    for chunk in rgba.chunks(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }

    image::RgbImage::from_raw(
        u32::from(pixmap.width()),
        u32::from(pixmap.height()),
        rgb,
    )
    .ok_or_else(|| "Failed to construct RgbImage from pixmap".to_string())
}
