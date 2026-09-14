//! # SRXformer Sprint 4: Wall-Clock Iso-Time Benchmark Executable
//!
//! Evaluates ScaledClassicTransformer vs QrenoSrxLM on Russian Wikipedia
//! for equal wall-clock time (e.g. 10 minutes vs 10 minutes).
//!
//! Usage:
//!   cargo run --release --bin isotime_wiki_bench -- [--duration <seconds>] [--tier <micro|standard|pro>]

use std::env;
use srxformer::scaled::{run_isotime_benchmark, IsoTimeConfig, Tier};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = IsoTimeConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--duration" | "-d" => {
                if i + 1 < args.len() {
                    config.duration_secs = args[i + 1].parse().unwrap_or(600.0);
                    i += 2;
                } else {
                    eprintln!("Error: --duration requires seconds argument");
                    return;
                }
            }
            "--tier" | "-t" => {
                if i + 1 < args.len() {
                    config.tier = match args[i + 1].to_lowercase().as_str() {
                        "micro" => Tier::Micro,
                        "standard" => Tier::Standard,
                        "pro" => Tier::Pro,
                        "ultra" => Tier::Ultra,
                        other => {
                            eprintln!("Unknown tier: '{other}'. Defaulting to Tier::Standard");
                            Tier::Standard
                        }
                    };
                    i += 2;
                } else {
                    eprintln!("Error: --tier requires tier name argument");
                    return;
                }
            }
            "--checkpoint-interval" | "-c" => {
                if i + 1 < args.len() {
                    config.checkpoint_interval_secs = args[i + 1].parse().unwrap_or(30.0);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--train-path" => {
                if i + 1 < args.len() {
                    config.train_path = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--val-path" => {
                if i + 1 < args.len() {
                    config.val_path = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--typo-path" => {
                if i + 1 < args.len() {
                    config.typo_path = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("SRXformer Iso-Time Benchmark Runner");
                println!("Options:");
                println!("  --duration, -d <seconds>    Duration in seconds per model (default: 600)");
                println!("  --tier, -t <tier>           Tier: micro, standard, pro, ultra (default: standard)");
                println!("  --checkpoint-interval <sec> Checkpoint interval in seconds (default: 30)");
                println!("  --train-path <path>         Path to wiki train corpus");
                println!("  --val-path <path>           Path to wiki val corpus");
                println!("  --typo-path <path>          Path to typo eval corpus");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    let (_classic_res, _srx_res) = run_isotime_benchmark(&config);
}
