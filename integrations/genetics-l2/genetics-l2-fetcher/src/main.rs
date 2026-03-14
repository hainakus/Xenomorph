use anyhow::{Context, Result};
use clap::{Arg, Command};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob, now_secs};
use tokio::time::{sleep, Duration};

// ── Source connectors ─────────────────────────────────────────────────────────

mod kaggle;
mod nih;
mod boinc;
mod dream;
mod horizon;

pub use kaggle::KaggleFetcher;
pub use nih::{NihFetcher, NihChallengeFetcher};
pub use boinc::BoincFetcher;
pub use dream::DreamFetcher;
pub use horizon::HorizonFetcher;

#[async_trait::async_trait]
pub trait SourceFetcher: Send + Sync {
    fn name(&self) -> &str;
    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>>;
}

// ── Main loop ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();
    let coordinator_url = m.get_one::<String>("coordinator").unwrap().clone();
    let poll_secs: u64  = m.get_one::<String>("poll-secs")
        .and_then(|s| s.parse().ok()).unwrap_or(300);
    let kaggle_key = m.get_one::<String>("kaggle-key").cloned()
        .or_else(|| std::env::var("KAGGLE_KEY").ok())
        .or_else(|| {
            let p = dirs::home_dir()?.join(".kaggle").join("kaggle.json");
            let s = std::fs::read_to_string(p).ok()?;
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            let user  = v["username"].as_str()?;
            let token = v["key"].as_str()?;
            Some(format!("{user}:{token}"))
        });
    let boinc_url        = m.get_one::<String>("boinc-url").cloned();
    let competition      = m.get_one::<String>("competition").cloned();
    let nih_challenges   = m.get_flag("nih-challenges");
    let dream_enabled    = m.get_flag("dream");
    let horizon_enabled  = m.get_flag("horizon");
    let synapse_pat      = m.get_one::<String>("synapse-pat").cloned()
        .or_else(|| std::env::var("SYNAPSE_PAT").ok());

    let http = reqwest::Client::new();

    let fetchers: Vec<Box<dyn SourceFetcher>> = {
        let mut v: Vec<Box<dyn SourceFetcher>> = Vec::new();
        if let Some(key) = kaggle_key {
            let mut fetcher = KaggleFetcher::new(key);
            if let Some(ref slug) = competition {
                fetcher = fetcher.with_competition(slug.clone());
            }
            v.push(Box::new(fetcher));
        } else if let Some(slug) = competition {
            v.push(Box::new(KaggleFetcher::new(String::new()).with_competition(slug)));
        }
        if let Some(url) = boinc_url {
            v.push(Box::new(BoincFetcher::new(url)));
        }
        v.push(Box::new(NihFetcher::new()));
        if nih_challenges {
            v.push(Box::new(NihChallengeFetcher::new()));
        }
        if dream_enabled {
            v.push(Box::new(DreamFetcher::new(synapse_pat)));
        }
        if horizon_enabled {
            v.push(Box::new(HorizonFetcher::new()));
        }
        v
    };

    log::info!("Job fetcher started — {} source(s)", fetchers.len());
    log::info!("  coordinator: {coordinator_url}");
    log::info!("  poll every {poll_secs}s");

    loop {
        for fetcher in &fetchers {
            match fetcher.fetch_jobs().await {
                Ok(jobs) => {
                    log::info!("[{}] fetched {} job(s)", fetcher.name(), jobs.len());
                    for job in jobs {
                        if let Err(e) = register_job(&http, &coordinator_url, &job).await {
                            log::warn!("[{}] register failed: {e:#}", fetcher.name());
                        }
                    }
                }
                Err(e) => log::warn!("[{}] fetch error: {e:#}", fetcher.name()),
            }
        }
        sleep(Duration::from_secs(poll_secs)).await;
    }
}

async fn register_job(
    http:            &reqwest::Client,
    coordinator_url: &str,
    job:             &ScientificJob,
) -> Result<()> {
    let resp = http
        .post(format!("{coordinator_url}/jobs"))
        .json(job)
        .send()
        .await
        .context("POST /jobs")?;

    if resp.status().is_success() || resp.status().as_u16() == 409 {
        log::debug!("  registered job {}", job.job_id);
    } else {
        let status = resp.status();
        let body   = resp.text().await.unwrap_or_default();
        log::warn!("  register {} → {status}: {body}", job.job_id);
    }
    Ok(())
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genetics-l2-fetcher")
        .about("Genetics L2 external job fetcher — polls Kaggle, NIH, BOINC and registers jobs")
        .arg(Arg::new("coordinator")
            .short('c').long("coordinator").value_name("URL")
            .default_value("http://localhost:8091")
            .help("genetics-l2-coordinator base URL"))
        .arg(Arg::new("poll-secs")
            .short('p').long("poll-secs").value_name("SECS")
            .default_value("300")
            .help("Poll interval in seconds"))
        .arg(Arg::new("kaggle-key")
            .long("kaggle-key").value_name("KEY")
            .help("Kaggle API key (username:token)"))
        .arg(Arg::new("boinc-url")
            .long("boinc-url").value_name("URL")
            .help("BOINC project XML URL"))
        .arg(Arg::new("competition")
            .long("competition").value_name("SLUG")
            .help("Seed a specific Kaggle competition (e.g. birdclef-2026)"))
        .arg(Arg::new("nih-challenges")
            .long("nih-challenges")
            .action(clap::ArgAction::SetTrue)
            .help("Poll NIH Prize Challenges from challenges.nih.gov"))
        .arg(Arg::new("dream")
            .long("dream")
            .action(clap::ArgAction::SetTrue)
            .help("Poll DREAM Challenges from synapse.org"))
        .arg(Arg::new("synapse-pat")
            .long("synapse-pat").value_name("TOKEN")
            .help("Synapse personal access token for authenticated DREAM challenge access (or env SYNAPSE_PAT)"))
        .arg(Arg::new("horizon")
            .long("horizon")
            .action(clap::ArgAction::SetTrue)
            .help("Poll EU Horizon Prize Challenges from CORDIS (cordis.europa.eu)"))
}
