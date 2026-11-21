//! A coding client continues a turn over tool rounds that skip
//! orchestration.  What the vision model read from the turn's image must
//! reach the answer model on every round, not only the first.

use crate::shared::model_service_a::bare_service;

fn image(bytes: &[u8]) -> Vec<(Vec<u8>, String)> {
    vec![(bytes.to_vec(), "image/png".to_string())]
}

#[test]
fn a_tool_round_reuses_the_first_rounds_image_reading() {
    let svc = bare_service();
    let spec = image(b"spec-screenshot");
    svc.remember_image_perception(&spec, "Implement a shell script:\n1. Install starship");

    let ctx = svc
        .image_context_for_round("Based on this implement the script", &spec)
        .expect("the cached reading");
    assert!(ctx.contains("1. Install starship"), "{ctx}");
    assert!(ctx.contains("SOLE source"), "a bare implement-this request reads the image alone: {ctx}");
}

#[test]
fn a_different_image_does_not_reuse_another_reading() {
    let svc = bare_service();
    svc.remember_image_perception(&image(b"first"), "first image text");
    // No workspace, so no vision model can read the new image.
    assert!(svc.image_context_for_round("implement it", &image(b"second")).is_none());
}

#[test]
fn no_images_means_no_image_context() {
    let svc = bare_service();
    assert!(svc.image_context_for_round("implement it", &[]).is_none());
}
