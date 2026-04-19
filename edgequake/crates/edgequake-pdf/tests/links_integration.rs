//! Integration tests: parse real arXiv PDFs and verify link/repo extraction.
//!
//! Skips silently if fixture PDFs are missing (e.g. in a shallow clone).

use std::path::PathBuf;

fn fixture(name: &str) -> Option<PathBuf> {
    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let p = PathBuf::from(crate_dir)
        .join("../../../legacy/edgequake-pdf/test-data/real_dataset")
        .join(name);
    p.exists().then_some(p)
}

#[test]
fn extract_links_no_github_paper() {
    // This paper has a project page but no github link.
    let Some(path) = fixture("01_2512.25075v1.pdf") else {
        eprintln!("SKIP: fixture not found");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let extraction = edgequake_pdf::extract_links(&bytes).expect("extract links");

    println!(
        "[no-github] pages={} links={} refs_boundary={:?}",
        extraction.page_count,
        extraction.links.len(),
        extraction.refs_boundary
    );
    assert!(extraction.page_count > 0);
    // Refs section must be detected for an arXiv paper.
    assert!(
        extraction.refs_boundary.is_some(),
        "references heading should be detected"
    );
    // No github links anywhere.
    let repos = edgequake_pdf::detect_repos(&extraction);
    assert_eq!(repos.len(), 0, "this paper has no github link");
}

#[test]
fn extract_links_with_github_paper() {
    // AlphaEvolve (Google DeepMind) — its markdown gold contains github refs.
    let Some(path) = fixture("AlphaEvolve.pdf") else {
        eprintln!("SKIP: fixture not found");
        return;
    };
    let bytes = std::fs::read(&path).expect("read fixture");
    let extraction = edgequake_pdf::extract_links(&bytes).expect("extract links");

    println!(
        "[alphaevolve] pages={} links={} refs_boundary={:?}",
        extraction.page_count,
        extraction.links.len(),
        extraction.refs_boundary
    );
    for link in &extraction.links {
        let past = extraction.is_past_refs(link);
        println!(
            "  {}page={} y={:.1} url={}",
            if past { "[REFS] " } else { "" },
            link.page_index,
            link.y_top,
            link.url
        );
    }

    let repos = edgequake_pdf::detect_repos(&extraction);
    println!("  detected {} repo candidate(s):", repos.len());
    for r in &repos {
        println!(
            "    {:?} {}/{} page={} y={:.1} -> {}",
            r.host, r.owner, r.repo, r.page_index, r.y_top, r.url
        );
    }

    // AlphaEvolve cites its own GitHub (via Colab wrapper URLs) in the body,
    // and also cites jax-ml/jax and openxla/xla in the references. Our
    // detection must return the self-cite and drop the references ones.
    assert!(
        !repos.is_empty(),
        "expected at least one repo candidate from pre-references links"
    );
    let top = &repos[0];
    assert_eq!(top.owner, "google-deepmind");
    assert_eq!(top.repo, "alphaevolve_results");
    assert!(
        top.page_index < 22,
        "top repo must come from pre-refs pages (refs starts at page 22)"
    );
    // Must not surface refs-section repos.
    for r in &repos {
        assert_ne!(
            (r.owner.as_str(), r.repo.as_str()),
            ("jax-ml", "jax"),
            "refs-section repo leaked through"
        );
        assert_ne!(
            (r.owner.as_str(), r.repo.as_str()),
            ("openxla", "xla"),
            "refs-section repo leaked through"
        );
    }
}
