//! Loads the workspace's active GGUF model (plus any project LoRA adapter).
//!
//! Port of `smartloop/loader/model_loader.py`.

use std::path::Path;
use std::sync::Arc;

use crate::config::app_settings::AppSettings;
use crate::llms::llama_cpp::LlamaCpp;
use crate::llms::llm_factory::{LlamaInstanceOptions, LlmFactory};
use crate::utils::device_utils::get_device_config;
use crate::workspace::Workspace;

const LOG: &str = "smartloop.loader.model_loader";

/// Resolves and loads the model file for the workspace's current project.
///
/// The Python version took a `logging.Logger`; logging here goes through
/// `tracing` under target `smartloop.loader.model_loader`.
pub struct ModelLoader {
    pub workspace: Arc<Workspace>,
    pub gguf_path: Option<String>,
}

impl ModelLoader {
    /// A loader for `workspace`; `gguf_path` overrides the workspace model path.
    pub fn new(workspace: Arc<Workspace>, gguf_path: Option<String>) -> Self {
        ModelLoader { workspace, gguf_path }
    }

    /// Find the GGUF adapter file (`<project_dir>/model/<adapter>.gguf`) for
    /// the current project, if one exists.
    fn resolve_adapter_path(&self) -> Option<String> {
        let project = self.workspace.current_project()?;
        if project.adapters.is_empty() {
            return None;
        }

        for adapter_name in &project.adapters {
            let adapter_gguf = Path::new(&project.project_dir).join("model").join(format!("{adapter_name}.gguf"));
            if adapter_gguf.exists() {
                let adapter_gguf = adapter_gguf.to_string_lossy().into_owned();
                tracing::info!(target: LOG, "Found LoRA adapter GGUF: {adapter_gguf}");
                return Some(adapter_gguf);
            }
        }

        None
    }

    /// Load the model: errors with "Model file not found: <path>" when the
    /// resolved file is absent; caps the context at `llm_main_n_ctx_cap` when
    /// set; prints a Project/Device parameter table.
    pub fn load(&self) -> anyhow::Result<Arc<LlamaCpp>> {
        self.load_inner().inspect_err(|e| {
            tracing::error!(target: LOG, "Failed to load model: {e}");
        })
    }

    fn load_inner(&self) -> anyhow::Result<Arc<LlamaCpp>> {
        tracing::info!(target: LOG, " Preparing and loading model...");

        let ws_model = self.workspace.get_model(None);
        let resolved_path = match (&self.gguf_path, &ws_model) {
            (Some(path), _) if !path.is_empty() => path.clone(),
            (_, Some(model)) => model.model_path(),
            _ => String::new(),
        };
        if !Path::new(&resolved_path).exists() {
            anyhow::bail!(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Model file not found: {resolved_path}"),
            ));
        }

        // Resolve LoRA adapter GGUF path (if any)
        let lora_path = self.resolve_adapter_path();

        let project = self.workspace.current_project();

        // Cap the main model's auto-sized context so its KV cache doesn't
        // claim the whole device and leave nothing for the on-demand
        // large/vision models (avoids llama.cpp OOM crashes).  A cap of 0
        // (default) means no cap — the main model keeps its full auto-sized
        // window.
        let mut main_max_n_ctx = None;
        if let Ok(settings) = AppSettings::try_new() {
            if settings.llm_main_n_ctx_cap > 0 {
                main_max_n_ctx = Some(settings.llm_main_n_ctx_cap as usize);
                tracing::info!(target: LOG, "[model] Capping main model context at {}", settings.llm_main_n_ctx_cap);
            }
        }

        let options = LlamaInstanceOptions {
            max_n_ctx: main_max_n_ctx,
            thinking: ws_model
                .as_ref()
                .and_then(|m| self.workspace.effective_model_thinking(&m.model_name, project)),
            model_name: ws_model.as_ref().map(|m| m.model_name.clone()),
            ..Default::default()
        };
        let model = LlmFactory::get_llama_cpp_instance(
            &resolved_path,
            project.map(|p| p.id.as_str()).unwrap_or(""),
            lora_path.as_deref(),
            options,
        )?;

        // print a table of model parameters and device configuration
        let current = self
            .workspace
            .projects
            .iter()
            .find(|p| p.current)
            .map(|p| p.id.clone())
            .unwrap_or_else(|| "N/A".to_string());
        let table = render_table(
            &["Parameter", "Value"],
            &[
                vec!["Project".to_string(), current],
                vec!["Device".to_string(), get_device_config().get_device_description()],
            ],
        );
        println!("{table}");

        Ok(Arc::new(model))
    }
}

/// Render a left-aligned table in PrettyTable's default ASCII style.
fn render_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let width = |col: usize| -> usize {
        rows.iter()
            .filter_map(|r| r.get(col))
            .map(|c| c.chars().count())
            .chain(std::iter::once(headers[col].chars().count()))
            .max()
            .unwrap_or(0)
    };
    let widths: Vec<usize> = (0..headers.len()).map(width).collect();
    let border = format!(
        "+{}+",
        widths.iter().map(|w| "-".repeat(w + 2)).collect::<Vec<_>>().join("+")
    );
    let line = |cells: Vec<&str>| -> String {
        let parts: Vec<String> = cells
            .iter()
            .zip(&widths)
            .map(|(c, w)| format!(" {}{} ", c, " ".repeat(w - c.chars().count())))
            .collect();
        format!("|{}|", parts.join("|"))
    };
    let mut out = vec![border.clone(), line(headers.to_vec()), border.clone()];
    for row in rows {
        out.push(line(row.iter().map(String::as_str).collect()));
    }
    out.push(border);
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::render_table;

    #[test]
    fn table_matches_prettytable_layout() {
        let t = render_table(&["Parameter", "Value"], &[vec!["Project".into(), "abc".into()]]);
        assert_eq!(
            t,
            "+-----------+-------+\n| Parameter | Value |\n+-----------+-------+\n| Project   | abc   |\n+-----------+-------+"
        );
    }
}
