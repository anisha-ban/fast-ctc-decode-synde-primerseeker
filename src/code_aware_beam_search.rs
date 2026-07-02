use log::debug; // at the top of your file
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

//use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{ArrayBase, Axis, Data, FoldWhile, Ix1, Ix2, Ix3, Zip};
//use ndarray_stats::QuantileExt;


// maybe replace exp with lookup table
fn log_sum_exp(a: f32, b: f32) -> f32 {
    let max_val = a.max(b);
    if max_val.is_infinite() && max_val < 0.0 {
        // Both are -inf
        f32::NEG_INFINITY
    } else {
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    }
}


// looks correct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvCodeConfig {
    #[serde(rename = "K")]
    pub k: usize,
    #[serde(rename = "allowedEdges")]
    pub allowed_edges: Vec<HashMap<String, Vec<Vec<usize>>>>,
    #[serde(rename = "allowedEdges_term")]
    pub allowed_edges_term: Vec<HashMap<String, Vec<Vec<usize>>>>,
    pub b: usize,
    pub c: usize,
    pub df: usize,
    pub ext: usize,
    pub m: usize,
    #[serde(rename = "numEdges")]
    pub num_edges: Vec<usize>,
    pub num_term_code_symbols: usize,
    pub q: usize,
}

impl ConvCodeConfig {
    /// Load configuration from JSON file
    pub fn from_json_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: ConvCodeConfig = serde_json::from_reader(reader)?;
        Ok(config)
    }

    /// Load configuration from JSON string
    pub fn from_json_str(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }
}

// looks correct
#[derive(Debug, Clone, Copy)]
struct ConvSearchPoint {
    pub node: i32,
    pub state: usize,
    pub gap_prob: f32,
    pub label_prob: f32,

    // additional variables added for convolutional code decoder
    pub sequence_length: usize,     // Length of predicted sequence in bases
    pub syndrome_state: usize,      // Current state in syndrome trellis
}

impl ConvSearchPoint {
    pub fn probability(&self) -> f32 {
        self.gap_prob + self.label_prob
    }
}

#[derive(Debug, Clone, Copy)]
struct ConvSearchPointLog {
    pub node: i32,
    pub state: usize,
    pub log_gap_prob: f32,
    pub log_label_prob: f32,

    // additional variables added for convolutional code decoder
    pub sequence_length: usize,
    pub syndrome_state: usize,
}

impl ConvSearchPointLog {
    fn log_probability(&self) -> f32 {
        log_sum_exp(self.log_gap_prob, self.log_label_prob)
    }
}


#[derive(Debug)]
pub enum ConvSearchError {
    IncomparableValues,
    RanOutOfBeam,
    InvalidPrimerLength,
    InvalidOffsetLength,
    InvalidTrellisTransition,
    NoCompleteSequence,
}

// added transition probabilities for the code
pub fn convolutional_beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (25 bases)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
    offset_sequence: &[usize],      // Offset sequence for payload calculation
    conv_config: &ConvCodeConfig,   // Convolutional code configuration
) -> Result<(String, Vec<usize>, u64), ConvSearchError> {

    // both primers must have same length
    if forward_primer.len() != reverse_primer.len() {
        return Err(ConvSearchError::InvalidPrimerLength);
    }
    let primer_length = forward_primer.len();
    let payload_length = offset_sequence.len();
    let total_length = 2 * primer_length + payload_length;

    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label

    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let mut beam = vec![ConvSearchPoint {
        node: ROOT_NODE,
        state: 0,
        gap_prob: 1.0,
        label_prob: 0.0,
        sequence_length: 0,
        syndrome_state: 0,  // Start in syndrome state 0
    }];

    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPoint> = None;
    // Counter for tracking complexity (number of score computations)
    let mut total_score_computations: u64 = 0;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
    // For each time step and its probability distribution

        next_beam.clear();

        for &search_point in &beam {
            let ConvSearchPoint {
                node,
                state,
                gap_prob,
                label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            if pr[0] > beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPoint {
                    node: node,
                    state: state,
                    label_prob: 0.0,
                    gap_prob: (label_prob + gap_prob) * pr[0],
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
            }

            // ======= Determine valid base extensions based on current position =====
            let mut offset = 0;
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length {
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
            }
            else if sequence_length >= primer_length + payload_length {
                valid_bases = vec![reverse_primer[sequence_length - primer_length - payload_length]];
                num_valid_bases = 1 as f32;
            }
            else if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                valid_bases = get_valid_bases(
                sequence_length - primer_length,
                syndrome_state,
                conv_config,
                payload_length,
                )?;
                num_valid_bases = valid_bases.len() as f32;
                // since we're in payload region, apply offset to valid bases
                offset = offset_sequence[sequence_length - primer_length];
                for b in &mut valid_bases {
                    *b = (*b + offset) % 4;
                }
            }
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                if pr_b < beam_cut_threshold {
                    continue;
                }

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {

                    // Handle repeat collapse (dwelling)
                    // “don’t double-count repeats”
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPoint {
                        node: node,
                        label_prob: label_prob * pr_b,
                        gap_prob: 0.0,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });

                    if valid_bases.contains(&label){
                        if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                            new_syndrome_state = calculate_new_syndrome_state(
                            sequence_length - primer_length,
                            syndrome_state,
                            (label + 4 - offset) % 4,
                            conv_config,
                            payload_length,
                            )?;
                        }
                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if gap_prob > 0.0 {
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {
                            total_score_computations += 1;
                            next_beam.push(ConvSearchPoint {
                                node: idx,
                                state: state,
                                label_prob: gap_prob * pr_b / num_valid_bases,
                                gap_prob: 0.0,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: new_syndrome_state,
                            });
                        }}
                } else if valid_bases.contains(&label){
                    if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                        new_syndrome_state = calculate_new_syndrome_state(
                        sequence_length - primer_length,
                        syndrome_state,
                        (label + 4 - offset) % 4,
                        conv_config,
                        payload_length,
                        )?;
                    }
                    // Normal extension
                    total_score_computations += 1;
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));
                    next_beam.push(ConvSearchPoint {
                        node: new_node_idx,
                        state: state,
                        label_prob: (label_prob + gap_prob) * pr_b / num_valid_bases,
                        gap_prob: 0.0,
                        sequence_length: new_sequence_length,
                        syndrome_state: new_syndrome_state,
                    });
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);

        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let p = sp.probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => p > prev.probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability())
                .partial_cmp(&(a.probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }



        // Normalize probabilities
        /*let top = beam[0].probability();
        for search_point in &mut beam {
            search_point.label_prob /= top;
            search_point.gap_prob /= top;
        }*/

    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let p = sp.probability();
            let better = match best_complete {
                None => true,
                Some(prev) => p > prev.probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }


    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };


    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);

    Ok((final_seq, path, total_score_computations))
}

// added transition probabilities for the code
pub fn vanilla_beam_search_log<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    payload_length: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (25 bases)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
) -> Result<(String, Vec<usize>, f32, u64), ConvSearchError> {

    /*println!("=== Starting vanilla_beam_search_log ===");
    println!("Network output shape: {:?}", network_output.dim());
    println!("Beam size: {}, Beam cut threshold: {}", beam_size, beam_cut_threshold);
    println!("Collapse repeats: {}", collapse_repeats);*/

    // both primers must have same length
    if forward_primer.len() != reverse_primer.len() {
        //println!("ERROR: Primer length mismatch! Forward: {}, Reverse: {}",
        //       forward_primer.len(), reverse_primer.len());
        return Err(ConvSearchError::InvalidPrimerLength);
    }
    let primer_length = forward_primer.len();
    let total_length = 2 * primer_length + payload_length;
    let mut total_score_computations: u64 = 0;
    /*println!("Primer length: {}, Payload length: {}, Total target length: {}",
           primer_length, payload_length, total_length);
    println!("Forward primer: {:?}", forward_primer);
    println!("Reverse primer: {:?}", reverse_primer);*/

    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label
    let log_beam_cut_threshold = beam_cut_threshold.ln();

    //println!("Alphabet size: {}, Log beam cut threshold: {}", alphabet_size, log_beam_cut_threshold);

    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let mut beam = vec![ConvSearchPointLog {
        node: ROOT_NODE,
        state: 0,
        log_gap_prob: 0.0,  // ln(1.0) = 0.0
        log_label_prob: f32::NEG_INFINITY,  // ln(0.0) = -inf
        sequence_length: 0,
        syndrome_state: 0,  // Start in syndrome state 0
    }];

    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPointLog> = None;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
    // For each time step and its probability distribution

        next_beam.clear();

        for (beam_idx, &search_point) in beam.iter().enumerate() {//for &search_point in &beam {
            let ConvSearchPointLog {
                node,
                state,
                log_gap_prob,
                log_label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            let log_pr_blank = pr[0].ln();
            if log_pr_blank > log_beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPointLog {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank,
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
            }

            // ======= Determine valid base extensions based on current position =====
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length {
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Forward primer region, valid base: {}",
                //           sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length + payload_length {
                valid_bases = vec![reverse_primer[sequence_length - primer_length - payload_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Reverse primer region, valid base: {}",
                //           sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                valid_bases = vec![0,1,2,3];
                num_valid_bases = 4 as f32;
                //println!("  Position {}: Payload region, syndrome_state={}, valid bases: {:?}",
                //           sequence_length, syndrome_state, valid_bases);
            }
            let log_num_valid_bases = num_valid_bases.ln();
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                let log_pr_b = pr_b.ln();
                if log_pr_b < log_beam_cut_threshold {
                    continue;
                }

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {
                    total_score_computations += 1;
                    // Handle repeat collapse (dwelling)
                    // "don't double-count repeats"
                    next_beam.push(ConvSearchPointLog {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });

                    if valid_bases.contains(&label){
                        if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                            new_syndrome_state = 0;
                        }

                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if log_gap_prob > f32::NEG_INFINITY {
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {
                            total_score_computations += 1;
                            next_beam.push(ConvSearchPointLog {
                                node: idx,
                                state: state,
                                log_label_prob: log_gap_prob + log_pr_b - log_num_valid_bases,
                                log_gap_prob: f32::NEG_INFINITY,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: 0,
                            });
                        }}
                } else if valid_bases.contains(&label){
                    if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                        new_syndrome_state = 0;
                        //println!("    Normal extension: label={}, old_syn={}, new_syn={}",
                        //           label, syndrome_state, new_syndrome_state);
                    }
                    // Normal extension
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));
                    total_score_computations += 1;
                    let combined_log_prob = log_sum_exp(log_label_prob, log_gap_prob);
                    next_beam.push(ConvSearchPointLog {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: combined_log_prob + log_pr_b - log_num_valid_bases,
                        log_gap_prob: f32::NEG_INFINITY,
                        sequence_length: new_sequence_length,
                        syndrome_state: 0,
                    });
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);
        //println!("  After extension: next_beam size = {}", beam.len());
        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths_log(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED log probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let log_p = sp.log_probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => log_p > prev.log_probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by log probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in log probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }

        // Note: Normalization in log domain would be subtracting the max log probability
        // Commented out as in original:
        /*let top_log = beam[0].log_probability();
        for search_point in &mut beam {
            search_point.log_label_prob -= top_log;
            search_point.log_gap_prob -= top_log;
        }*/
    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let log_p = sp.log_probability();
            let better = match best_complete {
                None => true,
                Some(prev) => log_p > prev.log_probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }

    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };

    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);
    // At the end, before returning:
    let final_score = if let Some(best) = best_complete {
        best.log_probability()
    } else {
        beam[0].log_probability()
    };

    // Change the return type and statement:
    Ok((final_seq, path, final_score, total_score_computations))
}


// added transition probabilities for the code
pub fn convolutional_beam_search_log<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (25 bases)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
    offset_sequence: &[usize],      // Offset sequence for payload calculation
    conv_config: &ConvCodeConfig,   // Convolutional code configuration
) -> Result<(String, Vec<usize>, f32, u64), ConvSearchError> {

    /*println!("=== Starting convolutional_beam_search_log ===");
    println!("Network output shape: {:?}", network_output.dim());
    println!("Beam size: {}, Beam cut threshold: {}", beam_size, beam_cut_threshold);
    println!("Collapse repeats: {}", collapse_repeats);*/

    // both primers must have same length
    if forward_primer.len() != reverse_primer.len() {
        //println!("ERROR: Primer length mismatch! Forward: {}, Reverse: {}",
        //          forward_primer.len(), reverse_primer.len());
        return Err(ConvSearchError::InvalidPrimerLength);
    }
    let primer_length = forward_primer.len();
    let payload_length = offset_sequence.len();
    let total_length = 2 * primer_length + payload_length;
    let mut total_score_computations: u64 = 0;
    /*println!("Primer length: {}, Payload length: {}, Total target length: {}",
           primer_length, payload_length, total_length);
    println!("Forward primer: {:?}", forward_primer);
    println!("Reverse primer: {:?}", reverse_primer);
    println!("Offset sequence: {:?}", offset_sequence);*/

    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label
    let log_beam_cut_threshold = beam_cut_threshold.ln();


    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let first_base = forward_primer[0];
    let init_prob = network_output[[0, first_base + 1]];
    let first_node_idx = suffix_tree
                        .get_child(ROOT_NODE, first_base)
                        .unwrap_or_else(|| suffix_tree.add_node(ROOT_NODE, first_base, 0));

    let mut beam = vec![ConvSearchPointLog {
        node: first_node_idx,
        state: 0,
        log_gap_prob: f32::NEG_INFINITY,  // ln(1.0) = 0.0
        log_label_prob: init_prob.ln(),  // ln(0.0) = -inf
        sequence_length: 1 as usize,
        syndrome_state: 0,  // Start in syndrome state 0
    }];
    //println!("\nfirst base: label={}, score={}",
    //                                   first_base, beam[0].log_label_prob);
    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPointLog> = None;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
        if time_idx == 0 {
            continue;
        }
    // For each time step and its probability distribution

        //if time_idx % 100 == 0 {
            //println!("Time step {}/{}: beam size = {}",
            //       time_idx, network_output.nrows(), beam.len());
        //    if let Some(ref best) = best_complete {
                //println!("  Best complete so far: length={}, log_prob={:.4}",
                //       best.sequence_length, best.log_probability());
        //    }
        //}
        //println!("\n======== Time step {}/{}: beam size = {} =========",
        //           time_idx, network_output.nrows(), beam.len());
        next_beam.clear();

        for (beam_idx, &search_point) in beam.iter().enumerate() {//for &search_point in &beam {
            let ConvSearchPointLog {
                node,
                state,
                log_gap_prob,
                log_label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            let log_pr_blank = pr[0].ln();
            if log_pr_blank > log_beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPointLog {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank,
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
                //println!("\nadd blank: label=N, old_syn={}, score={}",
                //                       syndrome_state, log_gap_prob);
            }

            // ======= Determine valid base extensions based on current position =====
            let mut offset = 0;
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length {
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Forward primer region, valid base: {}",
                //           sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length + payload_length {
                valid_bases = vec![reverse_primer[sequence_length - primer_length - payload_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Reverse primer region, valid base: {}",
                //           sequence_length, valid_bases[0]);

            }
            else if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                valid_bases = get_valid_bases(
                sequence_length - primer_length,
                syndrome_state,
                conv_config,
                payload_length,
                )?;
                num_valid_bases = valid_bases.len() as f32;
                // since we're in payload region, apply offset to valid bases
                offset = offset_sequence[sequence_length - primer_length];
                for b in &mut valid_bases {
                    *b = (*b + offset) % 4;
                }
                //println!("\n  Position {}: Payload region, syndrome_state={}, offset={}, valid bases: {:?}",
                //           sequence_length-primer_length, syndrome_state, offset, valid_bases);
            }
            let log_num_valid_bases = num_valid_bases.ln();
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                let log_pr_b = pr_b.ln();
                if log_pr_b < log_beam_cut_threshold {
                    continue;
                }

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {

                    // Handle repeat collapse (dwelling)
                    // "don't double-count repeats"
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });
                    //println!("\nlabel-dwells extension: position={}, label={}, old_syn={}, new_syn={}, score={}",
                    //                   sequence_length-primer_length, alphabet[label+1], syndrome_state, new_syndrome_state, log_label_prob+log_pr_b);

                    if valid_bases.contains(&label){
                        if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                            new_syndrome_state = calculate_new_syndrome_state(
                            sequence_length - primer_length,
                            syndrome_state,
                            (label + 4 - offset) % 4,
                            conv_config,
                            payload_length,
                            )?;
                        }

                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if log_gap_prob > f32::NEG_INFINITY {
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {
                            total_score_computations += 1;
                            next_beam.push(ConvSearchPointLog {
                                node: idx,
                                state: state,
                                log_label_prob: log_gap_prob + log_pr_b - log_num_valid_bases,
                                log_gap_prob: f32::NEG_INFINITY,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: new_syndrome_state,
                            });
                            //println!("\n Repeat-then-blank extension:  position={}, label={}, old_syn={}, new_syn={}, score={}",
                            //           sequence_length-primer_length, alphabet[label+1], syndrome_state, new_syndrome_state, log_label_prob);
                        }}
                } else if valid_bases.contains(&label){
                    if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                        new_syndrome_state = calculate_new_syndrome_state(
                        sequence_length - primer_length,
                        syndrome_state,
                        (label + 4 - offset) % 4,
                        conv_config,
                        payload_length,
                        )?;
                    }
                    // Normal extension
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));

                    let combined_log_prob = log_sum_exp(log_label_prob, log_gap_prob);
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: combined_log_prob + log_pr_b - log_num_valid_bases,
                        log_gap_prob: f32::NEG_INFINITY,
                        sequence_length: new_sequence_length,
                        syndrome_state: new_syndrome_state,
                    });
                    //println!("\n  Normal extension: position={}, label={}, old_syn={}, new_syn={}, score={}",
                    //               sequence_length-primer_length, alphabet[label+1], syndrome_state, new_syndrome_state, log_label_prob);
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);
        //println!("  After extension: next_beam size = {}", beam.len());
        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths_log(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED log probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let log_p = sp.log_probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => log_p > prev.log_probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by log probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in log probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }

        // Note: Normalization in log domain would be subtracting the max log probability
        // Commented out as in original:
        /*let top_log = beam[0].log_probability();
        for search_point in &mut beam {
            search_point.log_label_prob -= top_log;
            search_point.log_gap_prob -= top_log;
        }*/
    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let log_p = sp.log_probability();
            let better = match best_complete {
                None => true,
                Some(prev) => log_p > prev.log_probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }

    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };

    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);
    // At the end, before returning:
    let final_score = if let Some(best) = best_complete {
        best.log_probability()
    } else {
        beam[0].log_probability()
    };

    // Change the return type and statement:
    Ok((final_seq, path, final_score, total_score_computations))
}


pub fn marker_convolutional_beam_search_log<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (25 bases)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
    offset_sequence: &[usize],      // Offset sequence for payload calculation
    conv_config: &ConvCodeConfig,   // Convolutional code configuration
    marker_interval: usize,           // after every 'marker_interval' symbols, the marker sequence occurs
    marker_sequence: &[usize],      // user-defined marker sequence
) -> Result<(String, Vec<usize>, f32, u64), ConvSearchError> {

    /*println!("=== Starting marker_convolutional_beam_search_log ===");
    println!("kjdfl");
    println!("Network output shape: {:?}", network_output.dim());
    println!("Beam size: {}, Beam cut threshold: {}", beam_size, beam_cut_threshold);
    println!("Collapse repeats: {}", collapse_repeats);
    println!("{} {}", forward_primer.len(), reverse_primer.len());*/
    // both primers must have same length
    if forward_primer.len() != reverse_primer.len() {
        return Err(ConvSearchError::InvalidPrimerLength);
    }

    let primer_length = forward_primer.len();
    let marker_length = marker_sequence.len();
    let marker_block_length = marker_length + marker_interval;
    let payload_length = offset_sequence.len();
    let total_length = 2 * primer_length + payload_length;
    let mut total_score_computations: u64 = 0;
    /*println!("Primer length: {}, Payload length: {}, Total target length: {}",
           primer_length, payload_length, total_length);
    println!("marker_interval: {}, Marker sequence: {:?}",
           marker_interval, marker_sequence);
    println!("Forward primer: {:?}", forward_primer);
    println!("Reverse primer: {:?}", reverse_primer);
    println!("Offset sequence: {:?}", offset_sequence);*/

    let mut ind_to_conv_tree_map: [usize; 512] = [0; 512];
    let mut xx = 0;
    for ii in 0..payload_length {
        if (ii % marker_block_length) >= marker_interval {
            xx -= 1;
        }
        ind_to_conv_tree_map[ii] = xx;
        xx += 1;
    }

    // length of codeword without marker symbols
    let conv_cw_length = xx;

    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label
    let log_beam_cut_threshold = beam_cut_threshold.ln();

    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let first_base = forward_primer[0];
    let init_prob = network_output[[0, first_base + 1]];
    let first_node_idx = suffix_tree
                        .get_child(ROOT_NODE, first_base)
                        .unwrap_or_else(|| suffix_tree.add_node(ROOT_NODE, first_base, 0));

    let mut beam = vec![ConvSearchPointLog {
        node: first_node_idx,
        state: 0,
        log_gap_prob: f32::NEG_INFINITY,  // ln(1.0) = 0.0
        log_label_prob: init_prob.ln(),  // ln(0.0) = -inf
        sequence_length: 1 as usize,
        syndrome_state: 0,  // Start in syndrome state 0
    }];
    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPointLog> = None;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
    // For each time step and its probability distribution
        if time_idx == 0 {
            continue;
        }
        next_beam.clear();

        for (beam_idx, &search_point) in beam.iter().enumerate() {
            let ConvSearchPointLog {
                node,
                state,
                log_gap_prob,
                log_label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            let log_pr_blank = pr[0].ln();
            if log_pr_blank > log_beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPointLog {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank,
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
            }

            // ======= Determine valid base extensions based on current position =====
            let mut offset = 0;
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length { // in forward primer region
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Forward primer region, valid base: {}",                        sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length + payload_length { // in ending primer region
                valid_bases = vec![reverse_primer[sequence_length - primer_length - payload_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Reverse primer region, valid base: {}",sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length && sequence_length < primer_length + payload_length { // in payload region
                // TODO: cw_idx = sequence_length - primer_length - 1?
                let cw_idx = sequence_length - primer_length; // current beam's position in payload/codeword
                let marker_region = (cw_idx % marker_block_length) >= marker_interval;

                if marker_region == true {
                    valid_bases = vec![marker_sequence[(cw_idx % marker_block_length) - marker_interval]];
                    num_valid_bases = 1 as f32;
                    //println!("  Position {}: Marker region, valid bases: {:?}",                           sequence_length, valid_bases);
                }
                else { // in convolutional codeword part
                    valid_bases = get_valid_bases(
                                        ind_to_conv_tree_map[cw_idx], // index in the convolutional codeword without markers
                                        syndrome_state,
                                        conv_config,
                                        conv_cw_length, // length of the convolutional codeword without markers
                                    )?;
                    num_valid_bases = valid_bases.len() as f32;
                    //println!("  Position {}: Conv code region, syndrome_state={}, valid bases: {:?}", sequence_length, syndrome_state, valid_bases);
                }

                //println!("primer length - sequence length = {}, offset length = {}", sequence_length - primer_length, payload_length);
                // since we're in payload region, apply offset to valid bases
                offset = offset_sequence[sequence_length - primer_length];
                for b in &mut valid_bases {
                    *b = (*b + offset) % 4;
                }
                //println!("  Position {}: offset-adjusted valid bases: {:?}",sequence_length, valid_bases);

            }
            let log_num_valid_bases = num_valid_bases.ln();
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                let log_pr_b = pr_b.ln();
                if log_pr_b < log_beam_cut_threshold {
                    continue;
                }
                //println!("DEBUG: Processing label={}, log_pr_b={:.4}, time_idx={}", label, log_pr_b, time_idx);

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {
                    //println!("DEBUG:   Repeat collapse case - tip_label={:?}", tip_label);
                    // Handle repeat collapse (dwelling)
                    // "don't double-count repeats"
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });
                    //println!("DEBUG:   Pushed dwelling state (seq_len={})", sequence_length);

                    if valid_bases.contains(&label){
                        //println!("DEBUG:   Label is valid base, checking syndrome update");
                        if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                            //println!("DEBUG: line 1174 about to update syndrome_state");

                            let cw_idx_ = sequence_length - primer_length; // current beam's position in payload/codeword
                            let marker_region_ = (cw_idx_ % marker_block_length) >= marker_interval;

                            if marker_region_ == true {
                                new_syndrome_state = syndrome_state;
                            }
                            else{
                                new_syndrome_state = calculate_new_syndrome_state(
                                ind_to_conv_tree_map[sequence_length - primer_length],
                                syndrome_state,
                                (label + 4 - offset) % 4,
                                conv_config,
                                payload_length,
                                )?;
                            }
                            //println!("DEBUG:   Updated syndrome_state: {} -> {}", syndrome_state, new_syndrome_state);
                        }
                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if log_gap_prob > f32::NEG_INFINITY {
                                //println!("DEBUG:   Creating new node via gap transition");
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                //println!("DEBUG:   No gap transition available (log_gap_prob=-inf)");
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {
                            //println!("DEBUG:   Pushed extension via blank (node={}, new_seq_len={})", idx, new_sequence_length);
                            total_score_computations += 1;
                            next_beam.push(ConvSearchPointLog {
                                node: idx,
                                state: state,
                                log_label_prob: log_gap_prob + log_pr_b - log_num_valid_bases,
                                log_gap_prob: f32::NEG_INFINITY,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: new_syndrome_state,
                            });
                        }}
                } else if valid_bases.contains(&label){
                    //println!("DEBUG:   Normal extension case");
                    if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                        //println!("DEBUG: line 1209 About to update syndrome state");
                        let cw_idx_ = sequence_length - primer_length; // current beam's position in payload/codeword
                        let marker_region_ = (cw_idx_ % marker_block_length) >= marker_interval;

                        if marker_region_ == true {
                            new_syndrome_state = syndrome_state;
                        }
                        else{
                            new_syndrome_state = calculate_new_syndrome_state(
                            ind_to_conv_tree_map[sequence_length - primer_length],
                            syndrome_state,
                            (label + 4 - offset) % 4,
                            conv_config,
                            payload_length,
                            )?;
                        }
                        //println!("DEBUG:   Updated syndrome_state: {} -> {}", syndrome_state, new_syndrome_state);
                    }
                    //println!("line 1218");
                    // Normal extension
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));

                    let combined_log_prob = log_sum_exp(log_label_prob, log_gap_prob);
                    //println!("DEBUG:   Pushed normal extension (node={}, new_seq_len={}, combined_log_prob={:.4})", new_node_idx, new_sequence_length, combined_log_prob);
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: combined_log_prob + log_pr_b - log_num_valid_bases,
                        log_gap_prob: f32::NEG_INFINITY,
                        sequence_length: new_sequence_length,
                        syndrome_state: new_syndrome_state,
                    });
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);
        //println!("  After extension: next_beam size = {}", beam.len());
        // Merge identical paths (same node AND same syndrome state)

        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths_log(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED log probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let log_p = sp.log_probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => log_p > prev.log_probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by log probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in log probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }

        // Note: Normalization in log domain would be subtracting the max log probability
        // Commented out as in original:
        /*let top_log = beam[0].log_probability();
        for search_point in &mut beam {
            search_point.log_label_prob -= top_log;
            search_point.log_gap_prob -= top_log;
        }*/
    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let log_p = sp.log_probability();
            let better = match best_complete {
                None => true,
                Some(prev) => log_p > prev.log_probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }

    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };

    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);

    let final_score = if let Some(best) = best_complete {
        best.log_probability()
    } else {
        beam[0].log_probability()
    };

    // Change the return type and statement:
    Ok((final_seq, path, final_score, total_score_computations))
}

pub fn marker_beam_search_log<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (25 bases)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
    offset_sequence: &[usize],      // Offset sequence for payload calculation
    marker_interval: usize,           // after every 'marker_interval' symbols, the marker sequence occurs
    marker_sequence: &[usize],      // user-defined marker sequence
) -> Result<(String, Vec<usize>, f32, u64), ConvSearchError> {

    // both primers must have same length
    if forward_primer.len() != reverse_primer.len() {
        return Err(ConvSearchError::InvalidPrimerLength);
    }
    let primer_length = forward_primer.len();
    let marker_length = marker_sequence.len();
    let marker_block_length = marker_length + marker_interval;
    let payload_length = offset_sequence.len();
    let total_length = 2 * primer_length + payload_length;
    let mut total_score_computations: u64 = 0;
    /*println!("=== Starting marker_convolutional_beam_search_log ===");
    println!("Network output shape: {:?}", network_output.dim());
    println!("Beam size: {}, Beam cut threshold: {}", beam_size, beam_cut_threshold);
    println!("Collapse repeats: {}", collapse_repeats);
    println!("Primer length: {}, Payload length: {}, Total target length: {}",
           primer_length, payload_length, total_length);
    println!("marker_interval: {}, Marker sequence: {:?}",
           marker_interval, marker_sequence);
    println!("Forward primer: {:?}", forward_primer);
    println!("Reverse primer: {:?}", reverse_primer);
    println!("Offset sequence: {:?}", offset_sequence);*/


    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label
    let log_beam_cut_threshold = beam_cut_threshold.ln();

    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let first_base = forward_primer[0];
    let init_prob = network_output[[0, first_base + 1]];
    let first_node_idx = suffix_tree
        .get_child(ROOT_NODE, first_base)
        .unwrap_or_else(|| suffix_tree.add_node(ROOT_NODE, first_base, 0));
    let mut beam = vec![ConvSearchPointLog {
        node: first_node_idx,
        state: 0,
        log_gap_prob: f32::NEG_INFINITY,  // ln(1.0) = 0.0
        log_label_prob: init_prob.ln(),  // ln(0.0) = -inf
        sequence_length: 1 as usize,
        syndrome_state: 0,  // Start in syndrome state 0
    }];

    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPointLog> = None;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
    // For each time step and its probability distribution
        if time_idx == 0 {
            continue;
        }
        next_beam.clear();

        for (beam_idx, &search_point) in beam.iter().enumerate() {
            let ConvSearchPointLog {
                node,
                state,
                log_gap_prob,
                log_label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            let log_pr_blank = pr[0].ln();
            if log_pr_blank > log_beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPointLog {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank,
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
            }

            // ======= Determine valid base extensions based on current position =====
            let mut offset = 0;
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length { // in forward primer region
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Forward primer region, valid base: {}",                        sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length + payload_length { // in ending primer region
                valid_bases = vec![reverse_primer[sequence_length - primer_length - payload_length]];
                num_valid_bases = 1 as f32;
                //println!("  Position {}: Reverse primer region, valid base: {}",
                //           sequence_length, valid_bases[0]);
            }
            else if sequence_length >= primer_length && sequence_length < primer_length + payload_length { // in payload region
                let cw_idx = sequence_length - primer_length; // current beam's position in payload/codeword
                let marker_region = (cw_idx % marker_block_length) >= marker_interval;

                if marker_region == true {
                    valid_bases = vec![marker_sequence[(cw_idx % marker_block_length) - marker_interval]];
                    num_valid_bases = 1 as f32;
                    //println!("  Position {}: Marker region, valid bases: {:?}",                           sequence_length, valid_bases);
                }
                else { // in non-marker part of payload
                    valid_bases = vec![0,1,2,3];
                    num_valid_bases = 4 as f32;
                    //println!("  Position {}: Conv code region, syndrome_state={}, valid bases: {:?}", sequence_length, syndrome_state, valid_bases);
                }

                //println!("primer length - sequence length = {}, offset length = {}", sequence_length - primer_length, payload_length);
                // since we're in payload region, apply offset to valid bases
                offset = offset_sequence[sequence_length - primer_length];
                for b in &mut valid_bases {
                    *b = (*b + offset) % 4;
                }
                //println!("  Position {}: offset-adjusted valid bases: {:?}",sequence_length, valid_bases);

            }
            let log_num_valid_bases = num_valid_bases.ln();
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                let log_pr_b = pr_b.ln();
                if log_pr_b < log_beam_cut_threshold {
                    continue;
                }
                //println!("DEBUG: Processing label={}, log_pr_b={:.4}, time_idx={}", label, log_pr_b, time_idx);

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {
                    //println!("DEBUG:   Repeat collapse case - tip_label={:?}", tip_label);
                    // Handle repeat collapse (dwelling)
                    // "don't double-count repeats"
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });
                    //println!("DEBUG:   Pushed dwelling state (seq_len={})", sequence_length);

                    if valid_bases.contains(&label){
                        if sequence_length >= primer_length && sequence_length < primer_length + payload_length {

                            let cw_idx_ = sequence_length - primer_length; // current beam's position in payload/codeword
                            let marker_region_ = (cw_idx_ % marker_block_length) >= marker_interval;
                            new_syndrome_state = 0;

                        }
                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if log_gap_prob > f32::NEG_INFINITY {
                                //println!("DEBUG:   Creating new node via gap transition");
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                //println!("DEBUG:   No gap transition available (log_gap_prob=-inf)");
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {
                            //println!("DEBUG:   Pushed extension via blank (node={}, new_seq_len={})", idx, new_sequence_length);
                            total_score_computations += 1;
                            next_beam.push(ConvSearchPointLog {
                                node: idx,
                                state: state,
                                log_label_prob: log_gap_prob + log_pr_b - log_num_valid_bases,
                                log_gap_prob: f32::NEG_INFINITY,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: new_syndrome_state,
                            });
                        }}
                } else if valid_bases.contains(&label){
                    //println!("DEBUG:   Normal extension case");
                    if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                        //println!("DEBUG: line 1209 About to update syndrome state");
                        let cw_idx_ = sequence_length - primer_length; // current beam's position in payload/codeword
                        let marker_region_ = (cw_idx_ % marker_block_length) >= marker_interval;
                        new_syndrome_state = 0;
                        //println!("DEBUG:   Updated syndrome_state: {} -> {}", syndrome_state, new_syndrome_state);
                    }
                    //println!("line 1218");
                    // Normal extension
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));

                    let combined_log_prob = log_sum_exp(log_label_prob, log_gap_prob);
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: combined_log_prob + log_pr_b - log_num_valid_bases,
                        log_gap_prob: f32::NEG_INFINITY,
                        sequence_length: new_sequence_length,
                        syndrome_state: new_syndrome_state,
                    });
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);
        //println!("  After extension: next_beam size = {}", beam.len());
        // Merge identical paths (same node AND same syndrome state)

        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths_log(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED log probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let log_p = sp.log_probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => log_p > prev.log_probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by log probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in log probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }

        // Note: Normalization in log domain would be subtracting the max log probability
        // Commented out as in original:
        /*let top_log = beam[0].log_probability();
        for search_point in &mut beam {
            search_point.log_label_prob -= top_log;
            search_point.log_gap_prob -= top_log;
        }*/
    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let log_p = sp.log_probability();
            let better = match best_complete {
                None => true,
                Some(prev) => log_p > prev.log_probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }

    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };

    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);

    let final_score = if let Some(best) = best_complete {
        best.log_probability()
    } else {
        beam[0].log_probability()
    };

    // Change the return type and statement:
    Ok((final_seq, path, final_score, total_score_computations))
}

pub fn marker_beam_search_log_track<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (25 bases)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
    offset_sequence: &[usize],      // Offset sequence for payload calculation
    marker_interval: usize,           // after every 'marker_interval' symbols, the marker sequence occurs
    marker_sequence: &[usize],      // user-defined marker sequence
) -> Result<(String, f32, Vec<[f32; 4]>), ConvSearchError> {

    // both primers must have same length
    if forward_primer.len() != reverse_primer.len() {
        return Err(ConvSearchError::InvalidPrimerLength);
    }
    let primer_length = forward_primer.len();
    let marker_length = marker_sequence.len();
    let marker_block_length = marker_length + marker_interval;
    let payload_length = offset_sequence.len();
    let total_length = 2 * primer_length + payload_length;
    let mut total_score_computations: u64 = 0;

    /*println!("=== Starting marker_beam_search_log_track ===");
    println!("Network output shape: {:?}", network_output.dim());
    println!("Beam size: {}, Beam cut threshold: {}", beam_size, beam_cut_threshold);
    println!("Collapse repeats: {}", collapse_repeats);
    println!("Primer length: {}, Payload length: {}, Total target length: {}",
           primer_length, payload_length, total_length);
    println!("marker_interval: {}, Marker sequence: {:?}",
           marker_interval, marker_sequence);
    println!("Forward primer: {:?}", forward_primer);
    println!("Reverse primer: {:?}", reverse_primer);
    println!("Offset sequence: {:?}", offset_sequence);*/

    let mut base_prob_acc =  vec![[f32::NEG_INFINITY; 4]; total_length];

    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label
    let log_beam_cut_threshold = beam_cut_threshold.ln();

    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let first_base = forward_primer[0];
    let init_prob = network_output[[0, first_base + 1]];
    let first_node_idx = suffix_tree
        .get_child(ROOT_NODE, first_base)
        .unwrap_or_else(|| suffix_tree.add_node(ROOT_NODE, first_base, 0));

    let mut beam = vec![ConvSearchPointLog {
        node: first_node_idx,
        state: 0,
        log_gap_prob: f32::NEG_INFINITY,  // ln(1.0) = 0.0
        log_label_prob: init_prob.ln(),  // ln(0.0) = -inf
        sequence_length: 1 as usize,
        syndrome_state: 0,  // Start in syndrome state 0
    }];
    //println!(" First beam {}: Tip label {}, score: {}", first_node_idx, alphabet[first_base+1], init_prob.ln());

    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPointLog> = None;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
        // For each time step and its probability distribution
        if time_idx == 0 {
            continue;
        }
        next_beam.clear();

        for (beam_idx, &search_point) in beam.iter().enumerate() {
            let ConvSearchPointLog {
                node,
                state,
                log_gap_prob,
                log_label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            let log_pr_blank = pr[0].ln();
            if log_pr_blank > log_beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPointLog {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank,
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
                //println!("Time idx {}: adding blank to beam id {} with tip label {:?}, score: {}", time_idx, node, tip_label.map(|l| &alphabet[l + 1]), (log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank));
            }

            // ======= Determine valid base extensions based on current position =====
            let mut offset = 0;
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length { // in forward primer region
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
                //println!(" ---  Position {}: Forward primer region, valid base: {}", sequence_length, alphabet[valid_bases[0] + 1] );
            }
            else if sequence_length >= primer_length + payload_length { // in ending primer region
                valid_bases = vec![reverse_primer[sequence_length - primer_length - payload_length]];
                num_valid_bases = 1 as f32;
                //println!(" --- Position {}: Reverse primer region, valid base: {}", sequence_length, alphabet[valid_bases[0]+1]);
            }
            else if sequence_length >= primer_length && sequence_length < primer_length + payload_length { // in payload region
                let cw_idx = sequence_length - primer_length; // current beam's position in payload/codeword
                let marker_region = (cw_idx % marker_block_length) >= marker_interval;

                if marker_region == true {
                    valid_bases = vec![marker_sequence[(cw_idx % marker_block_length) - marker_interval]];
                    num_valid_bases = 1 as f32;
                    //println!(" ----  Position {}: Marker region, valid bases: {}", sequence_length, alphabet[valid_bases[0]+1]);
                }
                else { // in non-marker part of payload
                    valid_bases = vec![0,1,2,3];
                    num_valid_bases = 4 as f32;
                    //println!(" --- Position {}: non-marker region, valid bases: {:?}", sequence_length, valid_bases);
                }


                // since we're in payload region, apply offset to valid bases
                offset = offset_sequence[sequence_length - primer_length];
                for b in &mut valid_bases {
                    *b = (*b + offset) % 4;
                }
                //println!("  Position {}: offset-adjusted valid bases: {:?}",sequence_length, valid_bases);

            }
            let log_num_valid_bases = num_valid_bases.ln();
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                let log_pr_b = pr_b.ln();
                if log_pr_b < log_beam_cut_threshold {
                    continue;
                }

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {
                    //println!("DEBUG:   Repeat collapse case - tip_label={:?}", tip_label);
                    // Handle repeat collapse (dwelling)
                    // "don't double-count repeats"
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });
                    //println!("Time idx {}: beam id {} dwells at tip label {:?}, score: {}, seq length {}", time_idx, node, tip_label.map(|l| &alphabet[l + 1]), log_label_prob + log_pr_b, sequence_length);

                    if valid_bases.contains(&label){
                        if sequence_length >= primer_length && sequence_length < primer_length + payload_length {

                            let cw_idx_ = sequence_length - primer_length; // current beam's position in payload/codeword
                            let marker_region_ = (cw_idx_ % marker_block_length) >= marker_interval;

                            if marker_region_ == true {
                                new_syndrome_state = syndrome_state;
                            }
                            else{
                                new_syndrome_state = 0;
                            }
                        }
                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if log_gap_prob > f32::NEG_INFINITY {
                                //println!("DEBUG:   Creating new node via gap transition");
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                //println!("DEBUG:   No gap transition available (log_gap_prob=-inf)");
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {

                            total_score_computations += 1;
                            next_beam.push(ConvSearchPointLog {
                                node: idx,
                                state: state,
                                log_label_prob: log_gap_prob + log_pr_b - log_num_valid_bases,
                                log_gap_prob: f32::NEG_INFINITY,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: new_syndrome_state,
                            });
                            //println!("Time idx {}: beam id {} extends tip label {:?} by label {}, score: {} seq length {}", time_idx, idx, tip_label.map(|l| &alphabet[l + 1]), alphabet[label + 1], log_gap_prob + log_pr_b - log_num_valid_bases, new_sequence_length);
                            // =============== base probability tracking ===========================
                            let emission_log_prob = log_sum_exp(log_label_prob, log_gap_prob) + log_pr_b;
                            base_prob_acc[sequence_length][label] = log_sum_exp(emission_log_prob, base_prob_acc[sequence_length][label]);
                            // =====================================================================

                        }
                    }
                } else if valid_bases.contains(&label){
                    //println!("DEBUG:   Normal extension case");
                    if sequence_length >= primer_length && sequence_length < primer_length + payload_length {
                        //println!("DEBUG: line 1209 About to update syndrome state");
                        let cw_idx_ = sequence_length - primer_length; // current beam's position in payload/codeword
                        let marker_region_ = (cw_idx_ % marker_block_length) >= marker_interval;

                        if marker_region_ == true {
                            new_syndrome_state = syndrome_state;
                        }
                        else{
                            new_syndrome_state = 0;
                        }
                        //println!("DEBUG:   Updated syndrome_state: {} -> {}", syndrome_state, new_syndrome_state);
                    }

                    // Normal extension
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));

                    let combined_log_prob = log_sum_exp(log_label_prob, log_gap_prob);
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: combined_log_prob + log_pr_b - log_num_valid_bases,
                        log_gap_prob: f32::NEG_INFINITY,
                        sequence_length: new_sequence_length,
                        syndrome_state: new_syndrome_state,
                    });
                    //println!("Time idx {}: beam id {} extends tip label {:?} by label {}, score: {} seq length {}", time_idx, new_node_idx, tip_label.map(|l| &alphabet[l + 1]), alphabet[label + 1], combined_log_prob + log_pr_b - log_num_valid_bases, new_sequence_length);
                    // =============== base probability tracking ===========================
                    let emission_log_prob = log_sum_exp(log_label_prob, log_gap_prob) + log_pr_b;
                    base_prob_acc[sequence_length][label] = log_sum_exp(emission_log_prob, base_prob_acc[sequence_length][label]);
                    // =====================================================================
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);
        //println!("  After extension: next_beam size = {}", beam.len());

        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths_log(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED log probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let log_p = sp.log_probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => log_p > prev.log_probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by log probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in log probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }

        // Note: Normalization in log domain would be subtracting the max log probability
        // Commented out as in original:
        /*let top_log = beam[0].log_probability();
        for search_point in &mut beam {
            search_point.log_label_prob -= top_log;
            search_point.log_gap_prob -= top_log;
        }*/
    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let log_p = sp.log_probability();
            let better = match best_complete {
                None => true,
                Some(prev) => log_p > prev.log_probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }

    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };

    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);

    let final_score = if let Some(best) = best_complete {
        best.log_probability()
    } else {
        beam[0].log_probability()
    };

    // normalize each column of base_prob_acc
    for i in 0..total_length {
        let mut prob_sum = base_prob_acc[i][0];
        for j in 1..4{
            prob_sum = log_sum_exp(base_prob_acc[i][j], prob_sum);
        }
        if prob_sum == f32::NEG_INFINITY {
        // If no path reached this position, set to uniform probability (0.25)
        let log_025 = 0.25f32.ln();
        for j in 0..4 {
            base_prob_acc[i][j] = log_025;
        }
    } else {
        // Normal normalization
        for j in 0..4 {
            base_prob_acc[i][j] -= prob_sum;
        }
    }
    }
    let payload_probs: Vec<[f32; 4]> = base_prob_acc[primer_length..primer_length + payload_length]
        .to_vec();

    // Change the return type and statement:
    Ok((final_seq, final_score, payload_probs))
}

pub fn beam_search_log_with_base_probabilities<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer: &[usize],       // Forward primer sequence (nominally 25 bases, but we can also feed longer sequences)
    reverse_primer: &[usize],       // Reverse primer sequence (25 bases)
    offset_sequence: &[usize],      // Offset sequence for payload calculation
) -> Result<(String, f32, Vec<[f32; 4]>), ConvSearchError> {

    // both primers must have same length
    let primer_length_f = forward_primer.len();
    let primer_length_b = reverse_primer.len();
    let payload_length = offset_sequence.len();
    let total_length = primer_length_f + payload_length + primer_length_b;
    let mut total_score_computations: u64 = 0;

    /*println!("=== Starting marker_beam_search_log_track ===");
    println!("Network output shape: {:?}", network_output.dim());
    println!("Beam size: {}, Beam cut threshold: {}", beam_size, beam_cut_threshold);
    println!("Collapse repeats: {}", collapse_repeats);
    println!("Primer length: {}, Payload length: {}, Total target length: {}",
           primer_length, payload_length, total_length);
    println!("Forward primer: {:?}", forward_primer);
    println!("Reverse primer: {:?}", reverse_primer);
    println!("Offset sequence: {:?}", offset_sequence);*/

    let mut base_prob_acc =  vec![[f32::NEG_INFINITY; 4]; total_length];

    let alphabet_size = alphabet.len() - 1; // alphabet size minus the blank label
    let log_beam_cut_threshold = beam_cut_threshold.ln();

    let mut suffix_tree = SuffixTree::new(alphabet_size); // tracks partial sequences
    let first_base = forward_primer[0];
    let init_prob = network_output[[0, first_base + 1]];
    let first_node_idx = suffix_tree
        .get_child(ROOT_NODE, first_base)
        .unwrap_or_else(|| suffix_tree.add_node(ROOT_NODE, first_base, 0));

    let mut beam = vec![ConvSearchPointLog {
        node: first_node_idx,
        state: 0,
        log_gap_prob: f32::NEG_INFINITY,  // ln(1.0) = 0.0
        log_label_prob: init_prob.ln(),  // ln(0.0) = -inf
        sequence_length: 1 as usize,
        syndrome_state: 0,  // Start in syndrome state 0
    }];
    //println!(" First beam {}: Tip label {}, score: {}", first_node_idx, alphabet[first_base+1], init_prob.ln());

    let mut next_beam = Vec::new();

    // to track best beam of target length
    let mut best_complete: Option<ConvSearchPointLog> = None;

    for (time_idx, pr) in network_output.outer_iter().enumerate() {
        // For each time step and its probability distribution
        if time_idx == 0 {
            continue;
        }
        next_beam.clear();

        for (beam_idx, &search_point) in beam.iter().enumerate() {
            let ConvSearchPointLog {
                node,
                state,
                log_gap_prob,
                log_label_prob,
                sequence_length,
                syndrome_state,
            } = search_point;

            if sequence_length == total_length {
                continue; // no extension
            }

            let tip_label = suffix_tree.label(node);

            // Add blank/gap transition
            // add N to beam
            let log_pr_blank = pr[0].ln();
            if log_pr_blank > log_beam_cut_threshold {
                total_score_computations += 1;
                next_beam.push(ConvSearchPointLog {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank,
                    sequence_length: sequence_length,
                    syndrome_state: syndrome_state,
                });
                //println!("Time idx {}: adding blank to beam id {} with tip label {:?}, score: {}", time_idx, node, tip_label.map(|l| &alphabet[l + 1]), (log_sum_exp(log_label_prob, log_gap_prob) + log_pr_blank));
            }

            // ======= Determine valid base extensions based on current position =====
            let mut offset = 0;
            let mut valid_bases = Vec::new();
            let mut num_valid_bases = 0 as f32;
            if sequence_length < primer_length_f { // in forward primer region
                valid_bases = vec![forward_primer[sequence_length]];
                num_valid_bases = 1 as f32;
                //println!(" ---  Position {}: Forward primer region, valid base: {}", sequence_length, alphabet[valid_bases[0] + 1] );
            }
            else if sequence_length >= primer_length_f + payload_length { // in ending primer region
                valid_bases = vec![reverse_primer[sequence_length - primer_length_f - payload_length]];
                num_valid_bases = 1 as f32;
                //println!(" --- Position {}: Reverse primer region, valid base: {}", sequence_length, alphabet[valid_bases[0]+1]);
            }
            else if sequence_length >= primer_length_f && sequence_length < primer_length_f + payload_length { // in payload region
                let cw_idx = sequence_length - primer_length_f; // current beam's position in payload/codeword
                valid_bases = vec![0,1,2,3];
                num_valid_bases = 4 as f32;
                //println!(" --- Position {}: non-marker region, valid bases: {:?}", sequence_length, valid_bases);

                // since we're in payload region, apply offset to valid bases
                offset = offset_sequence[sequence_length - primer_length_f];
                for b in &mut valid_bases {
                    *b = (*b + offset) % 4;
                }
                //println!("  Position {}: offset-adjusted valid bases: {:?}",sequence_length, valid_bases);
            }
            let log_num_valid_bases = num_valid_bases.ln();
            // ==========================================================================

            // Add label transitions
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                let log_pr_b = pr_b.ln();
                if log_pr_b < log_beam_cut_threshold {
                    continue;
                }

                // Calculate new syndrome state and sequence length
                let new_sequence_length = sequence_length + 1;
                let mut new_syndrome_state = 0;

                // Same label as current tip - handle repeat collapse
                if collapse_repeats && Some(label) == tip_label {
                    //println!("DEBUG:   Repeat collapse case - tip_label={:?}", tip_label);
                    // Handle repeat collapse (dwelling)
                    // "don't double-count repeats"
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                        sequence_length: sequence_length, // preserve the old sequence length & syndrome state (dwelling)
                        syndrome_state: syndrome_state,
                    });
                    //println!("Time idx {}: beam id {} dwells at tip label {:?}, score: {}, seq length {}", time_idx, node, tip_label.map(|l| &alphabet[l + 1]), log_label_prob + log_pr_b, sequence_length);

                    if valid_bases.contains(&label){
                        if sequence_length >= primer_length_f && sequence_length < primer_length_f + payload_length {

                            let cw_idx_ = sequence_length - primer_length_f; // current beam's position in payload/codeword
                            new_syndrome_state = 0;
                        }
                        // a blank occurred before, so start a new occurrence of the same label (like a ␣ a).
                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if log_gap_prob > f32::NEG_INFINITY {
                                //println!("DEBUG:   Creating new node via gap transition");
                                Some(suffix_tree.add_node(node, label, time_idx))
                            } else {
                                //println!("DEBUG:   No gap transition available (log_gap_prob=-inf)");
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {

                            total_score_computations += 1;
                            next_beam.push(ConvSearchPointLog {
                                node: idx,
                                state: state,
                                log_label_prob: log_gap_prob + log_pr_b - log_num_valid_bases,
                                log_gap_prob: f32::NEG_INFINITY,
                                sequence_length: new_sequence_length, // transition via blank character implies an extension
                                syndrome_state: new_syndrome_state,
                            });
                            //println!("Time idx {}: beam id {} extends tip label {:?} by label {}, score: {} seq length {}", time_idx, idx, tip_label.map(|l| &alphabet[l + 1]), alphabet[label + 1], log_gap_prob + log_pr_b - log_num_valid_bases, new_sequence_length);
                            // =============== base probability tracking ===========================
                            let emission_log_prob = log_sum_exp(log_label_prob, log_gap_prob) + log_pr_b;
                            base_prob_acc[sequence_length][label] = log_sum_exp(emission_log_prob, base_prob_acc[sequence_length][label]);
                            // =====================================================================

                        }
                    }
                } else if valid_bases.contains(&label){
                    //println!("DEBUG:   Normal extension case");
                    if sequence_length >= primer_length_f && sequence_length < primer_length_f + payload_length {
                        //println!("DEBUG: line 1209 About to update syndrome state");
                        let cw_idx_ = sequence_length - primer_length_f; // current beam's position in payload/codeword
                        new_syndrome_state = 0;
                        //println!("DEBUG:   Updated syndrome_state: {} -> {}", syndrome_state, new_syndrome_state);
                    }

                    // Normal extension
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, time_idx));

                    let combined_log_prob = log_sum_exp(log_label_prob, log_gap_prob);
                    total_score_computations += 1;
                    next_beam.push(ConvSearchPointLog {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: combined_log_prob + log_pr_b - log_num_valid_bases,
                        log_gap_prob: f32::NEG_INFINITY,
                        sequence_length: new_sequence_length,
                        syndrome_state: new_syndrome_state,
                    });
                    //println!("Time idx {}: beam id {} extends tip label {:?} by label {}, score: {} seq length {}", time_idx, new_node_idx, tip_label.map(|l| &alphabet[l + 1]), alphabet[label + 1], combined_log_prob + log_pr_b - log_num_valid_bases, new_sequence_length);
                    // =============== base probability tracking ===========================
                    let emission_log_prob = log_sum_exp(log_label_prob, log_gap_prob) + log_pr_b;
                    base_prob_acc[sequence_length][label] = log_sum_exp(emission_log_prob, base_prob_acc[sequence_length][label]);
                    // =====================================================================
                }
            }
        } // dwelling/extension done

        std::mem::swap(&mut beam, &mut next_beam);
        //println!("  After extension: next_beam size = {}", beam.len());

        // Merge identical paths (same node AND same syndrome state)
        merge_identical_paths_log(&mut beam);

        // >>> After merge, update the best completed beam based on UNNORMALIZED log probability
        for &sp in &beam {
            if sp.sequence_length == total_length {
                let log_p = sp.log_probability();
                let better = match best_complete {
                    None => true,
                    Some(prev) => log_p > prev.log_probability(),
                };
                if better {
                    best_complete = Some(sp);
                }
            }
        }

        // Sort by log probability and prune
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });

        if has_nans {
            debug!("NaN detected in log probabilities!");
            return Err(ConvSearchError::IncomparableValues);
        }

        beam.truncate(beam_size);

        if beam.is_empty() {
            debug!("All beams pruned -> error");
            return Err(ConvSearchError::RanOutOfBeam);
        }

        // Note: Normalization in log domain would be subtracting the max log probability
        // Commented out as in original:
        /*let top_log = beam[0].log_probability();
        for search_point in &mut beam {
            search_point.log_label_prob -= top_log;
            search_point.log_gap_prob -= top_log;
        }*/
    }

    for &sp in &beam {
        if sp.sequence_length == total_length {
            let log_p = sp.log_probability();
            let better = match best_complete {
                None => true,
                Some(prev) => log_p > prev.log_probability(),
            };
            if better {
                best_complete = Some(sp);
            }
        }
    }

    // TODO do we need to ensure that the final beam has sequence_length = total_length?
    let chosen_node = if let Some(best) = best_complete {
        //debug!("Choosing saved best complete beam at node {}", best.node);
        best.node
    } else {
        //debug!("No complete beam found; falling back to best partial (beam[0])");
        beam[0].node
    };

    // Reconstruct the best path
    let mut path = Vec::new();
    let mut sequence = String::new();

    if chosen_node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(chosen_node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    let final_seq: String = sequence.chars().rev().collect();
    //debug!("Final sequence: {}", final_seq);
    //debug!("Final path: {:?}", path);

    let final_score = if let Some(best) = best_complete {
        best.log_probability()
    } else {
        beam[0].log_probability()
    };

    // normalize each column of base_prob_acc
    for i in 0..total_length {
        let mut prob_sum = base_prob_acc[i][0];
        for j in 1..4{
            prob_sum = log_sum_exp(base_prob_acc[i][j], prob_sum);
        }
        if prob_sum == f32::NEG_INFINITY {
            // If no path reached this position, set to uniform probability (0.25)
            let log_025 = 0.25f32.ln();
            for j in 0..4 {
                base_prob_acc[i][j] = log_025;
            }
        } else {
            // Normal normalization
            for j in 0..4 {
                base_prob_acc[i][j] -= prob_sum;
            }
        }
    }
    let payload_probs: Vec<[f32; 4]> = base_prob_acc[primer_length_f..primer_length_f + payload_length]
        .to_vec();

    // Change the return type and statement:
    Ok((final_seq, final_score, payload_probs))
}


// looks correct
fn get_valid_bases(
    cw_idx: usize,
    syndrome_state: usize,
    conv_config: &ConvCodeConfig,
    payload_length: usize,
) -> Result<Vec<usize>, ConvSearchError> {

    // Determine if we're in terminating region
    let is_terminating = cw_idx >=
        (payload_length - conv_config.num_term_code_symbols);

    let edges = if is_terminating {
        &conv_config.allowed_edges_term[cw_idx - (payload_length - conv_config.num_term_code_symbols)]
    } else {
        &conv_config.allowed_edges[cw_idx % conv_config.c]
    };

    let syndrome_key = syndrome_state.to_string();
    if let Some(allowed_transitions) = edges.get(&syndrome_key) {
        Ok(allowed_transitions.iter().map(|edge| edge[0]).collect())
    } else {
        Ok(vec![]) // No valid transitions from this state
    }
}

// looks correct, can potentially be merged with previous function
fn calculate_new_syndrome_state(
    cw_idx: usize,
    current_syndrome_state: usize,
    base: usize,
    conv_config: &ConvCodeConfig,
    payload_length: usize,
) -> Result<usize, ConvSearchError> {

    // Look up the transition in the trellis
    let is_terminating = cw_idx >=
        (payload_length - conv_config.num_term_code_symbols);

    let edges = if is_terminating {
        &conv_config.allowed_edges_term[cw_idx - (payload_length - conv_config.num_term_code_symbols)]
    } else {
        &conv_config.allowed_edges[cw_idx % conv_config.c]
    };

    let syndrome_key = current_syndrome_state.to_string();
    if let Some(allowed_transitions) = edges.get(&syndrome_key) {
        // Find the transition that matches our base
        for transition in allowed_transitions {
            if transition[0] == base {
                return Ok(transition[1]);
            }
        }
    }

    Err(ConvSearchError::InvalidTrellisTransition)

}


fn merge_identical_paths(beam: &mut Vec<ConvSearchPoint>) {
    const DELETE_MARKER: i32 = i32::min_value();

    // Sort by (node, syndrome_state) to group identical paths
    beam.sort_by_key(|x| x.node);

    let mut last_key = DELETE_MARKER;
    let mut last_key_pos = 0;

    for i in 0..beam.len() {
        let beam_item = beam[i];
        if beam_item.node == last_key {
            // Merge probabilities for same node
            beam[last_key_pos].label_prob += beam_item.label_prob;
            beam[last_key_pos].gap_prob += beam_item.gap_prob;
            beam[i].node = DELETE_MARKER; // Mark for deletion
        } else {
            last_key_pos = i;
            last_key = beam_item.node;
        }
    }

    beam.retain(|x| x.node != DELETE_MARKER);
}

// Helper function to merge identical paths in log domain
fn merge_identical_paths_log(beam: &mut Vec<ConvSearchPointLog>) {
    use std::collections::HashMap;

    let mut merged: HashMap<(i32, usize), ConvSearchPointLog> = HashMap::new();

    for &point in beam.iter() {
        let key = (point.node, point.syndrome_state);

        match merged.get_mut(&key) {
            Some(existing) => {
                // Merge probabilities using log-sum-exp
                existing.log_gap_prob = log_sum_exp(existing.log_gap_prob, point.log_gap_prob);
                existing.log_label_prob = log_sum_exp(existing.log_label_prob, point.log_label_prob);
            }
            None => {
                merged.insert(key, point);
            }
        }
    }

    beam.clear();
    beam.extend(merged.values());
}

// configure unit tests
#[cfg(test)]
mod tests {
    use super::*;
    //use test::Bencher;
    fn init_logger() {
    let _ = env_logger::builder()
        .is_test(true) // makes output play nice with `cargo test`
        .try_init();
    }
    /*
    #[test]
    fn test_conv_beam_search() {
        init_logger();
        let conv_config = ConvCodeConfig::from_json_file("syndrome_JSON/cc_2_1_3.json").unwrap();
        let forward_primer = vec![0, 3]; // AT
        let reverse_primer = vec![1, 2]; // CG
        let offset_sequence = vec![0, 0, 0, 0, 0];

        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("C"), String::from("G"), String::from("T")];
        let network_output = array![
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C
            [0.0f32, 0.0, 0.0, 1.0, 0.0], // G
        ];

        let (seq, _starts) = convolutional_beam_search(&network_output, &alphabet, 5, 0.0, false, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        assert_eq!(seq, "ATATTTACG");

        //let (seq, _starts) = convolutional_beam_search(&network_output, &alphabet, 5, 0.0, false, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        //assert_eq!(seq, "GGGAGAG");
    }*/
    #[test]
    fn test_conv_beam_search_offset() {
        init_logger();
        let conv_config = ConvCodeConfig::from_json_file("syndrome_JSON/cc_2_1_3.json").unwrap();
        let forward_primer = vec![0, 3]; // AT
        let reverse_primer = vec![1, 2]; // CG
        let offset_sequence = vec![0, 1, 2, 2, 3];

        let alphabet = vec![String::from("N"), String::from("A"), String::from("C"), String::from("G"), String::from("T")];
        let network_output = array![
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 0
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 1
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 2 AT (2)
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 3 ATA (3)
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N - 4 ATA (3)
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 5 ATAA (4)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 6 ATAAC (5)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 7 ATAAC (5)
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N - 8 ATAAC (5)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 9 ATAACC (6)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 10 ATAACC (6)
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 11 ATAACCT
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C
            [0.0f32, 0.0, 0.0, 1.0, 0.0], // G
        ];

        let (seq, _starts, total_score_computations) = convolutional_beam_search(&network_output, &alphabet, 5, 0.0, true, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        assert_eq!(&seq[0..2], "AT");
        //assert_eq!(seq[-2:], "CG");
        assert_eq!(seq, "ATAACCTCG");

        //let (seq, _starts) = convolutional_beam_search(&network_output, &alphabet, 5, 0.0, false, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        //assert_eq!(seq, "GGGAGAG");
    }
    #[test]
    fn test_vanilla_beam_search_log() {
        init_logger();
        let forward_primer = vec![0, 3]; // AT
        let reverse_primer = vec![1, 2]; // CG
        let payload_length = 5;

        let alphabet = vec![String::from("N"), String::from("A"), String::from("C"), String::from("G"), String::from("T")];
        let network_output = array![
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 0
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 1
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 2 AT (2)
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 3 ATA (3)
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N - 4 ATA (3)
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 5 ATAA (4)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 6 ATAAC (5)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 7 ATAAC (5)
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N - 8 ATAAC (5)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 9 ATAACC (6)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 10 ATAACC (6)
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 11 ATAACCT
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C
            [0.0f32, 0.0, 0.0, 1.0, 0.0], // G
        ];

        let (seq, _starts, sc, total_score_computations) = vanilla_beam_search_log(&network_output, &alphabet, 5, payload_length, 0.0, true, &forward_primer, &reverse_primer).unwrap();
        assert_eq!(&seq[0..2], "AT");
        //assert_eq!(seq[-2:], "CG");
        assert_eq!(seq, "ATAACCTCG"); // function returns cw+offset, not cw

        //let (seq, _starts) = convolutional_beam_search(&network_output, &alphabet, 5, 0.0, false, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        //assert_eq!(seq, "GGGAGAG");
    }
    #[test]
    fn test_conv_beam_search_offset_log() {
        init_logger();
        let conv_config = ConvCodeConfig::from_json_file("syndrome_JSON/cc_2_1_3.json").unwrap();
        let forward_primer = vec![0, 3]; // AT
        let reverse_primer = vec![1, 2]; // CG
        let offset_sequence = vec![0, 1, 2, 2, 3];

        let alphabet = vec![String::from("N"), String::from("A"), String::from("C"), String::from("G"), String::from("T")];
        let network_output = array![
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 0
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 1
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 2 AT (2)
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 3 ATA (3)
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N - 4 ATA (3)
            [0.0f32, 1.0, 0.0, 0.0, 0.0], // A - 5 ATAA (4)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 6 ATAAC (5)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 7 ATAAC (5)
            [1.0f32, 0.0, 0.0, 0.0, 0.0], // N - 8 ATAAC (5)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 9 ATAACC (6)
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C - 10 ATAACC (6)
            [0.0f32, 0.0, 0.0, 0.0, 1.0], // T - 11 ATAACCT
            [0.0f32, 0.0, 1.0, 0.0, 0.0], // C
            [0.0f32, 0.0, 0.0, 1.0, 0.0], // G
        ];

        let (seq, _starts, sc, total_score_computations) = convolutional_beam_search_log(&network_output, &alphabet, 5, 0.0, true, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        assert_eq!(&seq[0..2], "AT");
        //assert_eq!(seq[-2:], "CG");
        assert_eq!(seq, "ATAACCTCG"); // function returns cw+offset, not cw?

        //let (seq, _starts) = convolutional_beam_search(&network_output, &alphabet, 5, 0.0, false, &forward_primer, &reverse_primer, &offset_sequence, &conv_config).unwrap();
        //assert_eq!(seq, "GGGAGAG");
    }
    // TODO add test for marker convolutonal beam search log
}
