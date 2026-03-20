use log::debug; // at the top of your file
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;
use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{ArrayBase, Axis, Data, FoldWhile, Ix1, Ix2, Ix3, Zip};
use std::cmp::Ordering;
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
#[derive(Debug, Clone, Copy)]
struct PrimerSearchPoint {
    pub node: i32,
    pub gap_prob: f32,
    pub label_prob: f32,
    pub sequence_length: usize,     // Length of predicted sequence in bases
    pub start_pos: usize,           // starting sample
}

impl PrimerSearchPoint {
    pub fn probability(&self) -> f32 {
        self.gap_prob + self.label_prob
    }
}

#[derive(Debug, Clone, Copy)]
struct PrimerSearchPoint_ {
    pub node: i32,
    pub gap_prob: f32,
    pub label_prob: f32,
    pub sequence_length: usize,     // Length of predicted sequence in bases
    pub start_pos: usize,           // starting sample
    pub cur_time_idx: usize,
}

impl PrimerSearchPoint_ {
    pub fn probability(&self) -> f32 {
        self.gap_prob + self.label_prob
    }
}

struct OffsetBeamState {
    offset: usize,
    beam: Vec<PrimerSearchPoint>,
    next_beam: Vec<PrimerSearchPoint>,
    beam_size: usize,
    total_prob_mass: f32,
    active: bool,
    current_time_idx: usize,
}
struct OffsetBeamState_ {
    offset: usize,
    beam: Vec<PrimerSearchPoint_>,
    beam_size: usize,
    total_prob_mass: f32,
    active: bool,
    current_time_idx: usize,
}

#[derive(Debug)]
pub enum PrimerSearchError {
    IncomparableValues,
    InvalidParameters,
    RanOutOfBeam,
    NoCompleteSequence,
}

#[derive(Debug)]
pub enum OffsetBeamStateError {
    NoActiveBeams,
}

fn compute_combined_beam<D: ndarray::Data<Elem = f32>>(
    subsample: usize,
    depth: usize,
    network_output: &ArrayBase<D, Ix2>,
    subblock_start: usize,
    subblock_end: usize,
    primer_sequence: &[usize],
    best_complete_beam_score: f32,
    collapse_repeats: bool,
) -> Result<(Vec<PrimerSearchPoint_>, u64), PrimerSearchError> {

    let mut beam_list = vec![Vec::new(); subsample];
    let block_length = network_output.nrows();
    let target_length = primer_sequence.len();
    let beam_cut_threshold = 0.0;
    let mut total_score_computations = 0 as u64;
    const DELETE_MARKER: i32 = i32::min_value();

    // one beam list for each start_pos % subsample
    for i in 0..subsample {
        // STEP 1: initialize beams
        for start_pos in (subblock_start + i..subblock_end).step_by(subsample) {
            beam_list[i].push(PrimerSearchPoint_ {
                node: (start_pos as i32) * 1000,
                gap_prob: 1.0,
                label_prob: 0.0,
                sequence_length: 0,
                start_pos,
                cur_time_idx: start_pos,
            });
        }

        if beam_list[i].is_empty() {
            break;
        }

        // STEP 2: propagate beams
        let mut next_beam = Vec::new();
        for sample_depth in 0..subsample - i + depth {
            next_beam.clear();
            for &search_point in &beam_list[i] {
                let PrimerSearchPoint_ {
                    node,
                    gap_prob,
                    label_prob,
                    sequence_length,
                    start_pos,
                    cur_time_idx,
                } = search_point;

                let sample_pos = cur_time_idx;

                // Early pruning
                if gap_prob + label_prob < best_complete_beam_score
                    || sample_pos >= block_length
                    || sequence_length >= target_length {
                    continue;
                }

                let pr = network_output.row(sample_pos);

                // Blank/gap transition
                if pr[0] > beam_cut_threshold && sequence_length > 0 {
                    total_score_computations += 1;
                    next_beam.push(PrimerSearchPoint_ {
                        node,
                        gap_prob: (label_prob + gap_prob) * pr[0],
                        label_prob: 0.0,
                        sequence_length,
                        start_pos,
                        cur_time_idx: cur_time_idx + 1,
                    });
                }

                // Current base dwells
                if sequence_length > 0 {
                    let current_base_idx = primer_sequence[sequence_length - 1];
                    let prob_current_base = pr[current_base_idx + 1];
                    if prob_current_base >= beam_cut_threshold {
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node,
                            gap_prob: 0.0,
                            label_prob: label_prob * prob_current_base,
                            sequence_length,
                            start_pos,
                            cur_time_idx: cur_time_idx + 1,
                        });
                    }
                }

                // Extension transitions
                let next_base_idx = primer_sequence[sequence_length];
                let prob_next_base = pr[next_base_idx + 1];
                let current_base_idx = if sequence_length > 0 {
                    Some(primer_sequence[sequence_length - 1])
                } else {
                    None
                };
                if prob_next_base < beam_cut_threshold {
                    continue;
                }
                if Some(next_base_idx) == current_base_idx {
                    if gap_prob > 0.0 {
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node: if collapse_repeats {
                                node
                            } else {
                                node + 1
                            },
                            gap_prob: 0.0,
                            label_prob: gap_prob * prob_next_base,
                            sequence_length: if collapse_repeats {
                                sequence_length
                            } else {
                                sequence_length + 1
                            },
                            start_pos,
                            cur_time_idx: cur_time_idx + 1,
                        });
                    }
                } else {
                    total_score_computations += 1;
                    next_beam.push(PrimerSearchPoint_ {
                        node: node + 1,
                        gap_prob: 0.0,
                        label_prob: (label_prob + gap_prob) * prob_next_base,
                        sequence_length: sequence_length + 1,
                        start_pos,
                        cur_time_idx: cur_time_idx + 1,
                    });
                }
            }
            std::mem::swap(&mut beam_list[i], &mut next_beam);
            if beam_list[i].is_empty() {
                break;
            }

            // Step 4a: Merge identical paths
            beam_list[i].sort_by_key(|x| x.node);
            let mut last_key = DELETE_MARKER;
            let mut last_key_pos = 0;
            for j in 0..beam_list[i].len() {
                let beam_item = beam_list[i][j];
                if beam_item.node == last_key {
                    beam_list[i][last_key_pos].label_prob += beam_item.label_prob;
                    beam_list[i][last_key_pos].gap_prob += beam_item.gap_prob;
                    beam_list[i][j].node = DELETE_MARKER;
                } else {
                    last_key_pos = j;
                    last_key = beam_item.node;
                }
            }
            beam_list[i].retain(|x| x.node != DELETE_MARKER);

            // Step 4: Sort by total probability and prune
            let mut has_nans = false;
            beam_list[i].sort_unstable_by(|a, b| {
                (b.probability())
                    .partial_cmp(&(a.probability()))
                    .unwrap_or_else(|| {
                        has_nans = true;
                        std::cmp::Ordering::Equal
                    })
            });
            if has_nans {
                return Err(PrimerSearchError::IncomparableValues);
            }
            if beam_list[i].is_empty() {
                break;
            }
            beam_list[i].truncate((subblock_end - subblock_start + 1) / subsample);
        }
    }

    // MERGE BEAMS ACROSS BEAM_LIST
    let mut combined_beam: Vec<PrimerSearchPoint_> = beam_list.into_iter().flatten().collect();
    if !combined_beam.is_empty() {
        // Prepare for merging (crit: if nodes have same seq. length and ended at same sample)
        for beam in &mut combined_beam {
            beam.node = (beam.cur_time_idx as i32) * 1000 + beam.sequence_length as i32;
        }
        combined_beam.sort_by_key(|x| x.node);
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..combined_beam.len() {
            let beam_item = combined_beam[i];
            let current_prob = beam_item.probability();
            if beam_item.node == last_key {
                if combined_beam[last_key_pos].probability() < current_prob {
                //if combined_beam[last_key_pos].start_pos < beam_item.start_pos { // keeping max start pos
                    combined_beam[last_key_pos].start_pos = beam_item.start_pos;
                }
                combined_beam[last_key_pos].label_prob += beam_item.label_prob;
                combined_beam[last_key_pos].gap_prob += beam_item.gap_prob;
                combined_beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }
        combined_beam.retain(|x| x.node != DELETE_MARKER);
        let mut has_nans = false;

        // Sort by probability
        combined_beam.sort_unstable_by(|a, b| {
            (b.probability())
                .partial_cmp(&(a.probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal
                })
        });
        if has_nans {
            return Err(PrimerSearchError::IncomparableValues);
        }
        combined_beam.truncate((subblock_end - subblock_start + 1) / subsample);
    }
    Ok((combined_beam, total_score_computations))
}

pub fn primer_beam_search_brute<D: ndarray::Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    beam_size: usize,
    beam_cut_threshold: f32,
    primer_sequence_: &[usize],
    max_sample_depth: usize,
    shift: usize,
    subsample: usize,
) -> Result<(Vec<f32>, u64), PrimerSearchError> {

    let collapse_repeats = false;
    let block_length = network_output.nrows();
    let mut primer_sequence;
    if collapse_repeats {
        primer_sequence = Vec::new();
        for &x in primer_sequence_ {
            if primer_sequence.last() != Some(&x) {
                primer_sequence.push(x);
            }
        }
    }
    else { primer_sequence = Vec::from(primer_sequence_); }
    let primer_length = primer_sequence_.len();
    let target_length = primer_sequence.len();

    // Validate parameters
    if primer_length >= block_length {
        return Err(PrimerSearchError::InvalidParameters);
    }
    if shift == 0 {
        return Err(PrimerSearchError::InvalidParameters);
    }
    let mut total_score_computations: u64 = 0;
    let a = 1;
    let subblock_length = (beam_size * shift).min(block_length / a - primer_length + 1);
    let num_subblocks = (block_length / a - primer_length + 1).div_ceil(subblock_length);

    const DELETE_MARKER: i32 = i32::min_value();
    let total_num_start_positions = block_length - primer_length + 1;
    // array to store probabilities for each starting position
    let mut result_probs = vec![0.0f32; total_num_start_positions];
    let mut best_complete_beam_score = 0.0f32;

    // Iterate over start offsets
    for subblock_idx in 0..num_subblocks{
        let subblock_start = subblock_length * subblock_idx;
        let subblock_end = (block_length / a - primer_length + 1).min(subblock_length * (subblock_idx + 1));
        if subblock_end >= block_length / a {
            break;
        }
        let mut offset_beams: Vec<OffsetBeamState_>;

        if subsample > 1 {
            // initialize offset_beams with empty lists
            offset_beams = (0..shift)
                .map(|start_offset| {
                    let mut beam = Vec::new();
                    OffsetBeamState_ {
                        offset: start_offset,
                        beam,
                        beam_size: beam_size,
                        total_prob_mass: 0.0,
                        active: true,
                        current_time_idx: 0,
                    }
                })
                .collect();

            let (combined_beam, score_computations) = compute_combined_beam(
                subsample,
                5,
                &network_output,
                subblock_idx * subblock_length,
                subblock_end,
                &primer_sequence,
                best_complete_beam_score,
                collapse_repeats,
            )?;
            total_score_computations += score_computations;

            if combined_beam.is_empty() {
                break;
            }

            // ---- assign remaining beams (combined_beams) to offset_beams ----
            for beam in combined_beam {
                let offset = beam.start_pos % shift;
                offset_beams[offset].beam.push(beam);
                offset_beams[offset].total_prob_mass += beam.probability();
            }

            for i in 0..shift {
                if offset_beams[i].beam.is_empty() {
                    offset_beams[i].active = false;
                }
            }
        }
        else {
            offset_beams = (0..shift)
                .map(|start_offset| {
                    let mut pos = start_offset + subblock_idx * subblock_length;
                    let mut end_pos_at_offset = pos;
                    while end_pos_at_offset + shift < subblock_end {
                        end_pos_at_offset += shift;
                    }
                    let mut beam = Vec::new();
                    let mut temp_pos = pos;

                    while temp_pos <= end_pos_at_offset {
                        beam.push(PrimerSearchPoint_ {
                            node: (temp_pos as i32) * 1000,
                            gap_prob: 1.0,
                            label_prob: 0.0,
                            sequence_length: 0,
                            start_pos: temp_pos,
                            cur_time_idx: temp_pos,
                        });
                        temp_pos += shift;
                    }

                    let is_empty = beam.is_empty();
                    let init_prob_mass = if is_empty { 0.0 } else { beam_size as f32 };

                    OffsetBeamState_ {
                        offset: start_offset,
                        beam,
                        beam_size: beam_size,
                        total_prob_mass: init_prob_mass, // just needs to be equal at start
                        active: !is_empty,
                        current_time_idx: 0,
                    }
                })
                .collect();
        }

        let mut next_beam = Vec::new();

        let offset_order: Vec<usize> = if subsample > 1 {
            // Create indices and sort by probability mass (descending)
            let mut indices: Vec<usize> = (0..shift).collect();
            indices.sort_by(|&a, &b| {
                offset_beams[b].total_prob_mass
                    .partial_cmp(&offset_beams[a].total_prob_mass)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices
        } else {
            // Regular ascending order
            (0..shift).collect()
        };

        for start_offset in offset_order {
            if offset_beams[start_offset].beam.is_empty() {
                break; // No more valid positions in this offset class for this subblock
            }

            // Main beam search loop
            for time_idx in 0..max_sample_depth {
                next_beam.clear();
                for &search_point in &offset_beams[start_offset].beam {
                    let PrimerSearchPoint_ {
                        node,
                        gap_prob,
                        label_prob,
                        sequence_length,
                        start_pos,
                        cur_time_idx,
                    } = search_point;

                    // Early pruning: skip beams below the max probability at this depth
                    //if gap_prob + label_prob < max_prob_at_depth[time_idx] || gap_prob + label_prob < max_prob_at_depth[max_sample_depth-1]{
                    if gap_prob + label_prob < best_complete_beam_score{
                        continue;
                    }
                    let sample_pos = cur_time_idx;

                    // Skip if we're beyond the network output
                    if sample_pos >= block_length || sequence_length >= target_length {
                        continue;
                    }

                    let pr = network_output.row(sample_pos);

                    // Add blank/gap transition (and we probably should not consider gaps at the beginning)
                    if pr[0] > beam_cut_threshold && sequence_length > 0{
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node,
                            gap_prob: (label_prob + gap_prob) * pr[0],
                            label_prob: 0.0,
                            sequence_length,
                            start_pos,
                            cur_time_idx: cur_time_idx + 1,
                        });
                    }

                    // current base dwells
                    if sequence_length > 0 {
                        let current_base_idx = primer_sequence[sequence_length - 1];
                        let prob_current_base = pr[current_base_idx + 1]; // +1 because index 0 is blank
                        if prob_current_base >= beam_cut_threshold {
                            total_score_computations += 1;
                            next_beam.push(PrimerSearchPoint_ {
                                node,
                                gap_prob: 0.0,
                                label_prob: label_prob * prob_current_base,
                                sequence_length,
                                start_pos,
                                cur_time_idx: cur_time_idx + 1,
                            });
                        }
                    }

                    // Extension transition (next base in primer)
                    if sequence_length < target_length {
                        // extend by next base
                        let next_base_idx = primer_sequence[sequence_length];
                        let prob_next_base = pr[next_base_idx + 1]; // +1 because index 0 is blank
                        let current_base_idx = if sequence_length > 0 {
                            Some(primer_sequence[sequence_length - 1])
                        } else {
                            None
                        };

                        if prob_next_base < beam_cut_threshold {
                            continue;
                        }

                        if Some(next_base_idx) == current_base_idx {
                            //  Allow transition through blank to new occurrence
                            if gap_prob > 0.0 {
                                total_score_computations += 1;
                                next_beam.push(PrimerSearchPoint_ {
                                    node: if collapse_repeats { node } else { node + 1 }, // Unique node ID
                                    gap_prob: 0.0,
                                    label_prob: gap_prob * prob_next_base,
                                    sequence_length: if collapse_repeats { sequence_length } else { sequence_length + 1 },
                                    start_pos,
                                    cur_time_idx: cur_time_idx + 1,
                                });
                                if !collapse_repeats && sequence_length + 1 == target_length && gap_prob * prob_next_base > best_complete_beam_score {
                                    best_complete_beam_score = gap_prob * prob_next_base;
                                }
                            }
                        }
                        else{
                            total_score_computations += 1;
                            // Different label - extend sequence
                            next_beam.push(PrimerSearchPoint_ {
                                node: node + 1, // Unique node ID
                                gap_prob: 0.0,
                                label_prob: (label_prob + gap_prob) * prob_next_base,
                                sequence_length: sequence_length + 1,
                                start_pos,
                                cur_time_idx: cur_time_idx + 1,
                            });
                            if sequence_length + 1 == target_length && (label_prob + gap_prob) * prob_next_base > best_complete_beam_score {
                                best_complete_beam_score = (label_prob + gap_prob) * prob_next_base;
                            }
                        }
                    }
                }

                std::mem::swap(&mut offset_beams[start_offset].beam, &mut next_beam);
                if offset_beams[start_offset].beam.is_empty(){
                    break;
                }

                // ============ print beams =================

                /*println!("time_idx={} before merge: {} beams", time_idx, beam.len());
                println!(
                    "[DEBUG] time_idx={} surviving {} beams (top prob {:.6})",
                    time_idx,
                    beam.len(),
                    beam[0].probability()
                );
                for (i, b) in beam.iter().enumerate() {
                    if b.start_pos > 205 && b.start_pos < 222 {
                        println!(
                            "   Beam #{:<2} start_pos={} seq_len={} prob={:.6} (gap_prob={:.6}, label_prob={:.6})",
                            i,
                            b.start_pos,
                            b.sequence_length,
                            b.probability(),
                            b.gap_prob,
                            b.label_prob
                        );
                    }
                }*/

                // ==========================================

                // Step 4a: Merge identical paths (same start_pos and sequence_length)
                offset_beams[start_offset].beam.sort_by_key(|x| x.node);
                let mut write_idx = 0;
                for read_idx in 0..offset_beams[start_offset].beam.len() {
                    if write_idx > 0
                        && offset_beams[start_offset].beam[write_idx - 1].node == offset_beams[start_offset].beam[read_idx].node
                    //&& beam[write_idx - 1].sequence_length == beam[read_idx].sequence_length && beam[write_idx - 1].start_pos == beam[read_idx].start_pos
                    {
                        // Merge with previous
                        offset_beams[start_offset].beam[write_idx - 1].gap_prob += offset_beams[start_offset].beam[read_idx].gap_prob;
                        offset_beams[start_offset].beam[write_idx - 1].label_prob += offset_beams[start_offset].beam[read_idx].label_prob;
                    } else {
                        // Keep this beam point
                        if write_idx != read_idx {
                            offset_beams[start_offset].beam[write_idx] = offset_beams[start_offset].beam[read_idx];
                        }
                        write_idx += 1;
                    }
                }
                offset_beams[start_offset].beam.truncate(write_idx);

                // Step 4b: Sort by total probability and prune
                let mut has_nans = false;
                offset_beams[start_offset].beam.sort_unstable_by(|a, b| {
                    (b.probability())
                        .partial_cmp(&(a.probability()))
                        .unwrap_or_else(|| {
                            has_nans = true;
                            std::cmp::Ordering::Equal // don't really care
                        })
                });
                if has_nans {
                    return Err(PrimerSearchError::IncomparableValues);
                }
                if offset_beams[start_offset].beam.is_empty() {
                    // we've run out of beam (probably the threshold is too high)
                    //return Err(PrimerSearchError::RanOutOfBeam);
                    break;
                }

                offset_beams[start_offset].beam.truncate(beam_size);

                for b in offset_beams[start_offset].beam.iter() {
                    let prob = b.probability();
                    if b.sequence_length == target_length {
                        if prob > result_probs[b.start_pos] {
                            result_probs[b.start_pos] = prob;
                        }
                        if prob > best_complete_beam_score {
                            best_complete_beam_score = prob;
                        }
                    }
                }
                // ===================== PRINT ========================================
                /*for (i, b) in beam.iter().enumerate() {
                    if b.start_pos == 219{
                        println!(
                            "   Beam #{:<2} start_pos={} seq_len={} prob={:.6} (gap_prob={:.6}, label_prob={:.6})",
                            i,
                            b.start_pos,
                            b.sequence_length,
                            b.probability(),
                            b.gap_prob,
                            b.label_prob
                        );
                    }
                }*/
                // ====================================================================
            }
            //pos += num_init_beams * shift;

        }
    }
    /*for i in 0..(block_length - primer_length + 1){
        println!("{:.3}", result_probs[i]);
    }*/

    Ok((result_probs, total_score_computations))

}

//working version
pub fn primer_beam_search_opt<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    beam_size: usize,
    fraction_to_examine: f32,
    beam_cut_threshold: f32,
    concentration_threshold: f32,
    primer_sequence_: &[usize],
    max_sample_depth: usize,
    shift: usize,
    subsample: usize,
) -> Result<(Vec<f32>, u64), PrimerSearchError> {

    let block_length = network_output.nrows();
    let max_considered_block_length= (block_length as f32 * fraction_to_examine) as usize;
    let collapse_repeats = false;
    let mut primer_sequence;
    if collapse_repeats {
        primer_sequence = Vec::new();
        for &x in primer_sequence_ {
            if primer_sequence.last() != Some(&x) {
                primer_sequence.push(x);
            }
        }
    }
    else { primer_sequence = Vec::from(primer_sequence_); }
    let primer_length = primer_sequence_.len();
    let target_length = primer_sequence.len();

    let beam_size_per_offset = beam_size;
    let subblock_length = (beam_size * shift * subsample).min(max_considered_block_length - primer_length + 1);
    //subblock_length = subblock_length.min(block_length);
    let num_subblocks = (max_considered_block_length - primer_length + 1).div_ceil(subblock_length);

    //println!("block_length: {} subblock_length: {} max_considered_block_length: {}", block_length, subblock_length, max_considered_block_length);

    // Validate parameters
    if primer_length >= block_length {
        return Err(PrimerSearchError::InvalidParameters);
    }
    if shift == 0 {
        return Err(PrimerSearchError::InvalidParameters);
    }

    const DELETE_MARKER: i32 = i32::min_value();
    let mut total_score_computations: u64 = 0;
    let mut best_complete_beam_score: f32 = 0.0;
    let total_num_start_positions = block_length - primer_length + 1;
    let mut results_probs = vec![0.0f32; total_num_start_positions];


    let mut subblock_end = subblock_length;
    // Iterate over subblocks
    for subblock_idx in 0..num_subblocks {

        //println!("subblock_end {} block_length/a {}", subblock_end, block_length / a);
        if subblock_end >= max_considered_block_length {
            break;
        }
        let mut offset_beams: Vec<OffsetBeamState_>;

        // -------------- STAGE 1: SUBSAMPLE ------------------
        if subsample > 1 {

            // initialize offset_beams with empty lists
            offset_beams = (0..shift)
                .map(|start_offset| {
                    let mut beam = Vec::new();
                    OffsetBeamState_ {
                        offset: start_offset,
                        beam,
                        beam_size: beam_size_per_offset,
                        total_prob_mass: 0.0,
                        active: true,
                        current_time_idx: 0,
                    }
                })
                .collect();

            let (mut combined_beam, score_computations) = compute_combined_beam(
                subsample,
                5,
                &network_output,
                subblock_idx * subblock_length,
                subblock_end,
                &primer_sequence,
                best_complete_beam_score,
                collapse_repeats,
            )?;
            total_score_computations += score_computations;

            if combined_beam.is_empty() {
                break;
            }

            // -------- adaptive reduction of combined_beam based on prob. concentration ------
            let total_prob: f32 = combined_beam.iter().map(|b| b.probability()).sum();
            let mut cumulative_prob = 0.0;
            let mut effective_beam_size = combined_beam.len();
            for (idx, b) in combined_beam.iter().enumerate() {
                cumulative_prob += b.probability();
                if cumulative_prob >= concentration_threshold * total_prob {
                    effective_beam_size = idx + 1;
                    break;
                }
            }
            // truncate combined_beam based on effective_beam_size if need be
            if effective_beam_size < combined_beam.len() / 2 {
                let reduced_beam_size = (effective_beam_size * 4 / 3).max(combined_beam.len() / 2);
                if reduced_beam_size < combined_beam.len() {
                    combined_beam.truncate(reduced_beam_size);
                }
            }
            // -------------------------------------------------------------------------------

            // ---- assign remaining beams (combined_beams) to offset_beams ----
            for beam in combined_beam {
                let offset = beam.start_pos % shift;
                offset_beams[offset].beam.push(beam);
                offset_beams[offset].total_prob_mass += beam.probability();
            }

            for i in 0..shift {
                if offset_beams[i].beam.is_empty() {
                    offset_beams[i].active = false;
                }
            }
        }
        else {
            // assign beam in offset_beams the old-fashioned way
            offset_beams = (0..shift)
                .map(|start_offset| {
                    let mut pos = start_offset + subblock_idx * subblock_length;
                    let mut end_pos_at_offset = pos;
                    while end_pos_at_offset + shift < subblock_end {
                        end_pos_at_offset += shift;
                    }

                    let mut beam = Vec::new();
                    let mut temp_pos = pos;

                    while temp_pos <= end_pos_at_offset {
                        beam.push(PrimerSearchPoint_ {
                            node: (temp_pos as i32) * 1000,
                            gap_prob: 1.0,
                            label_prob: 0.0,
                            sequence_length: 0,
                            start_pos: temp_pos,
                            cur_time_idx: temp_pos,
                        });
                        temp_pos += shift;
                    }

                    let is_empty = beam.is_empty();
                    let init_prob_mass = if is_empty { 0.0 } else { beam_size_per_offset as f32 };

                    OffsetBeamState_ {
                        offset: start_offset,
                        beam,
                        beam_size: beam_size_per_offset,
                        total_prob_mass: init_prob_mass, // just needs to be equal at start
                        active: !is_empty,
                        current_time_idx: 0,
                    }
                })
                .collect();
        }
        // =============================STAGE 1 DONE ====================================================


        // -------------- STAGE 2: STANDARD BEAM PROPAGATION AFTER SUBSAMPLING ------------------
        // process each beam list in offset beams: propagate till max_sample_depth
        let mut step = 0;
        let mut next_beam = Vec::new();
        while true {
            //  termination conditions: if all beams'lists are inactive
            if offset_beams.iter().all(|s| !s.active) {
                break;
            }

            // pick beam list (among all active beams) with highest total probability
            let offset_state =  offset_beams
                .iter_mut()
                .filter(|state| state.active)
                .max_by(|a, b| a.total_prob_mass.partial_cmp(&b.total_prob_mass).unwrap()).unwrap();

            // ------------- prepare for beam expansion (beam list with highest prob. mass) ---------------------
            next_beam.clear();
            for &search_point in &offset_state.beam {
                let PrimerSearchPoint_ {
                    node,
                    gap_prob,
                    label_prob,
                    sequence_length,
                    start_pos,
                    cur_time_idx,
                } = search_point;

                // Early pruning: skip beams below the max probability at this depth
                if gap_prob + label_prob < best_complete_beam_score {
                    continue;
                }

                let sample_pos = cur_time_idx;
                // Skip if we're beyond the network output
                if sample_pos >= block_length || sequence_length > target_length || sample_pos - start_pos >= max_sample_depth {
                    continue;
                }

                let pr = network_output.row(sample_pos);

                // Blank/gap transition
                if pr[0] > beam_cut_threshold && sequence_length > 0 {
                    total_score_computations += 1;
                    next_beam.push(PrimerSearchPoint_ {
                        node: node,
                        gap_prob: (label_prob + gap_prob) * pr[0],
                        label_prob: 0.0,
                        sequence_length,
                        start_pos,
                        cur_time_idx: sample_pos + 1,
                    });
                }

                // Current base dwells
                if sequence_length > 0 && sequence_length <= target_length{
                    let current_base_idx = primer_sequence[sequence_length - 1];
                    let prob_current_base = pr[current_base_idx + 1];
                    if prob_current_base >= beam_cut_threshold {
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node: node,
                            gap_prob: 0.0,
                            label_prob: label_prob * prob_current_base,
                            sequence_length,
                            start_pos,
                            cur_time_idx: sample_pos + 1,
                        });
                    }
                }

                // Extension transition
                if sequence_length < target_length {
                    let next_base_idx = primer_sequence[sequence_length];
                    let prob_next_base = pr[next_base_idx + 1];
                    let current_base_idx = if sequence_length > 0 {
                        Some(primer_sequence[sequence_length - 1])
                    } else {
                        None
                    };

                    if prob_next_base < beam_cut_threshold {
                        continue;
                    }

                    if Some(next_base_idx) == current_base_idx {
                        if gap_prob > 0.0 {
                            total_score_computations += 1;
                            next_beam.push(PrimerSearchPoint_ {
                                node: if collapse_repeats { node } else { node + 1 },
                                gap_prob: 0.0,
                                label_prob: gap_prob * prob_next_base,
                                sequence_length: if collapse_repeats { sequence_length } else { sequence_length + 1 },
                                start_pos,
                                cur_time_idx: sample_pos + 1,
                            });
                            if !collapse_repeats && sequence_length + 1 == target_length && gap_prob * prob_next_base > best_complete_beam_score {
                                best_complete_beam_score = gap_prob * prob_next_base;
                            }
                        }
                    } else {
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node: node + 1,
                            gap_prob: 0.0,
                            label_prob: (label_prob + gap_prob) * prob_next_base,
                            sequence_length: sequence_length + 1,
                            start_pos,
                            cur_time_idx: sample_pos + 1,
                        });
                        if sequence_length + 1 == target_length && (label_prob + gap_prob) * prob_next_base > best_complete_beam_score {
                            best_complete_beam_score = (label_prob + gap_prob) * prob_next_base;
                        }
                    }
                }
            }

            if offset_state.active && next_beam.is_empty() {
                offset_state.beam.clear();
                offset_state.active = false;
                continue;
            }

            std::mem::swap(&mut offset_state.beam, &mut next_beam);

            // Merge identical paths
            offset_state.beam.sort_by_key(|x| x.node);
            let mut last_key = DELETE_MARKER;
            let mut last_key_pos = 0;
            for i in 0..offset_state.beam.len() {
                let beam_item = offset_state.beam[i];
                if beam_item.node == last_key {
                    offset_state.beam[last_key_pos].label_prob += beam_item.label_prob;
                    offset_state.beam[last_key_pos].gap_prob += beam_item.gap_prob;
                    offset_state.beam[i].node = DELETE_MARKER;
                } else {
                    last_key_pos = i;
                    last_key = beam_item.node;
                }

            }
            offset_state.beam.retain(|x| x.node != DELETE_MARKER);
            let mut has_nans = false;

            // Sort by probability and prune
            offset_state.beam.sort_unstable_by(|a, b| {
                (b.probability())
                    .partial_cmp(&(a.probability()))
                    .unwrap_or_else(|| {
                        has_nans = true;
                        std::cmp::Ordering::Equal
                    })
            });

            if has_nans {
                return Err(PrimerSearchError::IncomparableValues);
            }

            if offset_state.active && offset_state.beam.is_empty() {
                offset_state.active = false;
                continue;
            }

            // -------------- Adaptive beam size reduction -------------------
            //if step > 0 && step % truncation_interval == 0 {
            //   let total_prob: f32 = offset_state.beam.iter().map(|b| b.probability()).sum();
            //   let mut cumulative_prob = 0.0;
            //    let mut effective_beam_size = offset_state.beam.len();
            //    for (idx, b) in offset_state.beam.iter().enumerate() {
            //        cumulative_prob += b.probability();
            //        if cumulative_prob >= concentration_threshold * total_prob {
            //            effective_beam_size = idx + 1;
            //            break;
            //        }
            //    }
            //    if effective_beam_size < offset_state.beam_size / 2
            //        && (effective_beam_size * 4 / 3).max(min_beams) < offset_state.beam_size
            //    {
            //        offset_state.beam_size = (effective_beam_size * 4 / 3).max(min_beams);
            //    }
            //}
            // -----------------------------------------------------------------
            offset_state.beam.truncate(offset_state.beam_size);

            // Update results
            offset_state.total_prob_mass = 0.0;
            for b in offset_state.beam.iter() {
                let prob = b.probability();
                offset_state.total_prob_mass += prob;
                if b.sequence_length == target_length {
                    results_probs[b.start_pos] += prob;
                    if results_probs[b.start_pos] > best_complete_beam_score {
                        best_complete_beam_score = results_probs[b.start_pos]
                    }
                }
            }

            // ============== beam list expansion completed (might be useless) =====================
            offset_state.current_time_idx += 1;
            if offset_state.current_time_idx >= max_sample_depth {
                offset_state.active = false;
                offset_state.beam.clear();
            }

            step += 1;
        }

        // cleaning up
        for offset_state in offset_beams.iter_mut() {
            if !offset_state.beam.is_empty() {
                offset_state.beam.clear();
            }
        }

        // update subblock_idx
        subblock_end = (subblock_end + subblock_length).min(max_considered_block_length);
    }
    Ok((results_probs, total_score_computations))
}


pub fn primer_beam_search_ss<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    init_beam_size: usize,
    min_beams: usize,
    beam_cut_threshold: f32,
    concentration_threshold: f32,
    relative_prob_threshold: f32,
    primer_sequence_: &[usize],
    max_sample_depth: usize,
    truncation_interval: usize,
    truncation_interval_offsets: usize,
    shift: usize,
    subsample: usize,
) -> Result<(Vec<f32>, u64), PrimerSearchError> {

    let block_length = network_output.nrows();
    let collapse_repeats = false;
    let mut primer_sequence;
    if collapse_repeats {
        primer_sequence = Vec::new();
        for &x in primer_sequence_ {
            if primer_sequence.last() != Some(&x) {
                primer_sequence.push(x);
            }
        }
    }
    else { primer_sequence = Vec::from(primer_sequence_); }
    let primer_length = primer_sequence_.len();
    let target_length = primer_sequence.len();

    let beam_size_per_offset = init_beam_size;
    let a = 2;
    let subblock_length = (init_beam_size * shift * subsample).min(block_length / a - primer_length + 1);
    let num_subblocks = (block_length / a - primer_length + 1).div_ceil(subblock_length);

    // Primer search additional params
    /*let min_beams = 4;
    let truncation_interval = 15;
    let truncation_interval_offsets = 4;
    let concentration_threshold = 0.98;
    let relative_prob_threshold = 0.001; // threshold for pruning weak offsets
    let time_lag_threshold = 15;*/

    // Validate parameters
    if primer_length >= block_length {
        return Err(PrimerSearchError::InvalidParameters);
    }
    if shift == 0 {
        return Err(PrimerSearchError::InvalidParameters);
    }

    /*eprintln!("=== FUNCTION PARAMETERS ===");
    eprintln!("init_beam_size: {}", init_beam_size);
    eprintln!("shift: {}", shift);
    eprintln!("subsample: {}", subsample);
    eprintln!("truncation_interval: {}", truncation_interval); // staged only
    eprintln!("truncation_interval_offsets: {}", truncation_interval_offsets); // staged only
    eprintln!("min_beams: {}", min_beams); // staged only
    eprintln!("concentration_threshold: {}", concentration_threshold); // staged only
    eprintln!("relative_prob_threshold: {}", relative_prob_threshold); // staged only
    eprintln!("subblock_length: {}", subblock_length); // staged only
    eprintln!("num_subblocks: {}", num_subblocks); // staged only
    eprintln!("=========================");*/

    const DELETE_MARKER: i32 = i32::min_value();
    let mut total_score_computations: u64 = 0;
    let mut best_complete_beam_score: f32 = 0.0;
    let total_num_start_positions = block_length - primer_length + 1;
    let mut results_probs = vec![0.0f32; total_num_start_positions];


    let mut subblock_end = subblock_length;
    // Iterate over subblocks
    for subblock_idx in 0..num_subblocks {

        //println!("subblock_end {} block_length/a {}", subblock_end, block_length / a);
        if subblock_end >= block_length / a {
            break;
        }
        let mut offset_beams: Vec<OffsetBeamState_>;

        // -------------- STAGE 1: SUBSAMPLE ------------------
        if subsample > 1 {
            // initialize offset_beams with empty lists
            offset_beams = (0..shift)
                .map(|start_offset| {
                    let mut beam = Vec::new();
                    OffsetBeamState_ {
                        offset: start_offset,
                        beam,
                        beam_size: beam_size_per_offset,
                        total_prob_mass: 0.0,
                        active: true,
                        current_time_idx: 0,
                    }
                })
                .collect();

            let (combined_beam, score_computations) = compute_combined_beam(
                subsample,
                5,
                &network_output,
                subblock_idx * subblock_length,
                subblock_end,
                &primer_sequence,
                best_complete_beam_score,
                collapse_repeats,
            )?;
            total_score_computations += score_computations;

            if combined_beam.is_empty() {
                break;
            }

            // find out how the probability is concentrated in combined_beam
            /*let total_prob: f32 = combined_beam.iter().map(|b| b.probability()).sum();
            let mut cumulative_prob = 0.0;
            let mut effective_beam_size = combined_beam.len();

            for (idx, b) in combined_beam.iter().enumerate() {
                cumulative_prob += b.probability();
                if cumulative_prob >= concentration_threshold * total_prob {
                    effective_beam_size = idx + 1;
                    break;
                }
            }
            // truncate combined_beam based on effective_beam_size if need be
            if effective_beam_size < combined_beam.len() / 2 {
                combined_beam.truncate((effective_beam_size * 4 / 3).max(combined_beam.len() / 2));
            }*/

            // ---- assign remaining beams (combined_beams) to offset_beams ----
            for beam in combined_beam {
                let offset = beam.start_pos % shift;
                offset_beams[offset].beam.push(beam);
                offset_beams[offset].total_prob_mass += beam.probability();
            }

            for i in 0..shift {
                if offset_beams[i].beam.is_empty() {
                    offset_beams[i].active = false;
                }
            }
            // -------------------------------------------------------------------

            // _____ !!!! UPDATE: The following makes things worse!!!!!! ______
            // ------- new way to assign remaining beams (combined_beams) to offset_beams -------
            // aims to distribute beams into minimal number of buckets
            /*let mut sorted_beams = combined_beam;

            // Greedy assignment: assign each beam to the first valid bucket
            for beam in sorted_beams {
                let mut assigned = false;

                // Try each existing bucket in offset_beams
                for bucket in offset_beams.iter_mut() {
                    // Check if this bucket can accept the beam
                    if bucket.beam.len() < bucket.beam_size {
                        // Check if all beams in this bucket are at least shift samples apart
                        let can_fit = bucket.beam.iter().all(|existing| {
                            (existing.start_pos as i32 - beam.start_pos as i32).abs() >= shift as i32
                        });

                        if can_fit {
                            let prob = beam.probability();
                            bucket.beam.push(beam);
                            bucket.total_prob_mass += prob;
                            assigned = true;
                            break;
                        }
                    }
                }

                // If couldn't assign, create a new bucket
                if !assigned {
                    let prob = beam.probability();
                    let new_bucket = OffsetBeamState_ {
                        offset: offset_beams.len(),
                        beam: vec![beam],
                        beam_size: beam_size_per_offset,
                        total_prob_mass: prob,
                        active: true,
                        current_time_idx: 0,
                    };
                    offset_beams.push(new_bucket);
                }
            }*/
            // --------------------------------------------------------------------
        }
        else {
            // assign beam in offset_beams the old-fashioned way
            offset_beams = (0..shift)
                .map(|start_offset| {
                    let mut pos = start_offset + subblock_idx * subblock_length;
                    let mut end_pos_at_offset = pos;
                    while end_pos_at_offset + shift < subblock_end {
                        end_pos_at_offset += shift;
                    }

                    let mut beam = Vec::new();
                    let mut temp_pos = pos;

                    while temp_pos <= end_pos_at_offset {
                        beam.push(PrimerSearchPoint_ {
                            node: (temp_pos as i32) * 1000,
                            gap_prob: 1.0,
                            label_prob: 0.0,
                            sequence_length: 0,
                            start_pos: temp_pos,
                            cur_time_idx: temp_pos,
                        });
                        temp_pos += shift;
                    }

                    let is_empty = beam.is_empty();
                    let init_prob_mass = if is_empty { 0.0 } else { beam_size_per_offset as f32 };

                    OffsetBeamState_ {
                        offset: start_offset,
                        beam,
                        beam_size: beam_size_per_offset,
                        total_prob_mass: init_prob_mass, // just needs to be equal at start
                        active: !is_empty,
                        current_time_idx: 0,
                    }
                })
                .collect();
        }
        // =====================================================================================


        // -------------- STAGE 2: STANDARD BEAM PROPAGATION AFTER SUBSAMPLING ------------------
        // process each beam list in offset beams: propagate till max_sample_depth
        let mut step = 0;
        let mut next_beam = Vec::new();
        let mut max_total_prob_mass = 0.0f32;
        while true {

            //  termination conditions: if all beams'lists are inactive
            if offset_beams.iter().all(|s| !s.active) {
                break;
            }

            // ------------------- beam killer -------------------------
            /*if step > 0 && step % truncation_interval_offsets == 0{
                max_total_prob_mass = offset_beams
                    .iter()
                    .filter(|s| s.active)
                    .map(|s| s.total_prob_mass)
                    .fold(max_total_prob_mass, f32::max);
                let prob_threshold = relative_prob_threshold * max_total_prob_mass;
                for j in 0..shift {
                    if offset_beams[j].active && offset_beams[j].total_prob_mass > 0.0 && offset_beams[j].total_prob_mass < prob_threshold {
                        offset_beams[j].active = false;
                        offset_beams[j].beam.clear();
                    }
                }
                if offset_beams.iter().all(|s| !s.active) {
                    break;
                }
            }*/
            // --------------------------------------------------------

            // pick beam list (among all active beams) with highest total probability
            let offset_state =  offset_beams
                .iter_mut()
                .filter(|state| state.active)
                .max_by(|a, b| a.total_prob_mass.partial_cmp(&b.total_prob_mass).unwrap()).unwrap();


            // -------- prepare for beam expansion (beam list with highest prob. mass) --------
            next_beam.clear();
            for &search_point in &offset_state.beam {
                let PrimerSearchPoint_ {
                    node,
                    gap_prob,
                    label_prob,
                    sequence_length,
                    start_pos,
                    cur_time_idx,
                } = search_point;

                // Early pruning: skip beams below the max probability at this depth
                if gap_prob + label_prob < best_complete_beam_score {
                    continue;
                }

                let sample_pos = cur_time_idx;
                // Skip if we're beyond the network output
                if sample_pos >= block_length || sequence_length >= target_length || sample_pos - start_pos >= max_sample_depth {
                    continue;
                }

                let pr = network_output.row(sample_pos);

                // Blank/gap transition
                if pr[0] > beam_cut_threshold && sequence_length > 0 {
                    total_score_computations += 1;
                    next_beam.push(PrimerSearchPoint_ {
                        node: node,
                        gap_prob: (label_prob + gap_prob) * pr[0],
                        label_prob: 0.0,
                        sequence_length,
                        start_pos,
                        cur_time_idx: sample_pos + 1,
                    });
                }

                // Current base dwells
                if sequence_length > 0 {
                    let current_base_idx = primer_sequence[sequence_length - 1];
                    let prob_current_base = pr[current_base_idx + 1];
                    if prob_current_base >= beam_cut_threshold {
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node: node,
                            gap_prob: 0.0,
                            label_prob: label_prob * prob_current_base,
                            sequence_length,
                            start_pos,
                            cur_time_idx: sample_pos + 1,
                        });
                    }
                }

                // Extension transition
                if sequence_length < target_length {
                    let next_base_idx = primer_sequence[sequence_length];
                    let prob_next_base = pr[next_base_idx + 1];
                    let current_base_idx = if sequence_length > 0 {
                        Some(primer_sequence[sequence_length - 1])
                    } else {
                        None
                    };

                    if prob_next_base < beam_cut_threshold {
                        continue;
                    }

                    if Some(next_base_idx) == current_base_idx {
                        if gap_prob > 0.0 {
                            total_score_computations += 1;
                            next_beam.push(PrimerSearchPoint_ {
                                node: if collapse_repeats { node } else { node + 1 },
                                gap_prob: 0.0,
                                label_prob: gap_prob * prob_next_base,
                                sequence_length: if collapse_repeats { sequence_length } else { sequence_length + 1 },
                                start_pos,
                                cur_time_idx: sample_pos + 1,
                            });
                            if !collapse_repeats && sequence_length + 1 == target_length && gap_prob * prob_next_base > best_complete_beam_score {
                                best_complete_beam_score = gap_prob * prob_next_base;
                            }
                        }
                    } else {
                        total_score_computations += 1;
                        next_beam.push(PrimerSearchPoint_ {
                            node: node + 1,
                            gap_prob: 0.0,
                            label_prob: (label_prob + gap_prob) * prob_next_base,
                            sequence_length: sequence_length + 1,
                            start_pos,
                            cur_time_idx: sample_pos + 1,
                        });
                        if sequence_length + 1 == target_length && (label_prob + gap_prob) * prob_next_base > best_complete_beam_score {
                            best_complete_beam_score = (label_prob + gap_prob) * prob_next_base;
                        }
                    }
                }
            }

            if offset_state.active && next_beam.is_empty() {
                offset_state.beam.clear();
                offset_state.active = false;
                continue;
            }

            std::mem::swap(&mut offset_state.beam, &mut next_beam);

            // Merge identical paths
            offset_state.beam.sort_by_key(|x| x.node);
            let mut last_key = DELETE_MARKER;
            let mut last_key_pos = 0;
            for i in 0..offset_state.beam.len() {
                let beam_item = offset_state.beam[i];
                if beam_item.node == last_key {
                    offset_state.beam[last_key_pos].label_prob += beam_item.label_prob;
                    offset_state.beam[last_key_pos].gap_prob += beam_item.gap_prob;
                    offset_state.beam[i].node = DELETE_MARKER;
                } else {
                    last_key_pos = i;
                    last_key = beam_item.node;
                }

            }
            offset_state.beam.retain(|x| x.node != DELETE_MARKER);
            let mut has_nans = false;

            // Sort by probability and prune
            offset_state.beam.sort_unstable_by(|a, b| {
                (b.probability())
                    .partial_cmp(&(a.probability()))
                    .unwrap_or_else(|| {
                        has_nans = true;
                        std::cmp::Ordering::Equal
                    })
            });

            if has_nans {
                return Err(PrimerSearchError::IncomparableValues);
            }

            if offset_state.active && offset_state.beam.is_empty() {
                offset_state.active = false;
                continue;
            }

            // Adaptive beam size reduction
            /*if step > 0 && step % truncation_interval == 0 {
                let total_prob: f32 = offset_state.beam.iter().map(|b| b.probability()).sum();
                let mut cumulative_prob = 0.0;
                let mut effective_beam_size = offset_state.beam.len();

                for (idx, b) in offset_state.beam.iter().enumerate() {
                    cumulative_prob += b.probability();
                    if cumulative_prob >= concentration_threshold * total_prob {
                        effective_beam_size = idx + 1;
                        break;
                    }
                }

                if effective_beam_size < offset_state.beam_size / 2
                    && (effective_beam_size * 4 / 3).max(min_beams) < offset_state.beam_size
                {
                    offset_state.beam_size = (effective_beam_size * 4 / 3).max(min_beams);
                }
            }
            offset_state.beam.truncate(offset_state.beam_size);*/

            // Update results
            offset_state.total_prob_mass = 0.0;
            for b in offset_state.beam.iter() {
                let prob = b.probability();
                offset_state.total_prob_mass += prob;
                if b.sequence_length == target_length {
                    if prob > results_probs[b.start_pos] {
                        results_probs[b.start_pos] = prob;
                    }
                    if prob > best_complete_beam_score {
                        best_complete_beam_score = prob;
                    }
                }
            }

            // ============== beam list expansion completed (might be useless) =====================
            offset_state.current_time_idx += 1;
            if offset_state.current_time_idx >= max_sample_depth {
                offset_state.active = false;
                offset_state.beam.clear();
            }

            step += 1;
        }

        // cleaning up
        for offset_state in offset_beams.iter_mut() {
            if !offset_state.beam.is_empty() {
                offset_state.beam.clear();
            }
        }

        // update subblock_idx
        subblock_end = (subblock_end + subblock_length).min(block_length / a).min(total_num_start_positions);
    }

    Ok((results_probs, total_score_computations))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    fn init_logger() {
        let _ = env_logger::builder()
            .is_test(true) // makes output play nice with `cargo test`
            .try_init();
    }
    #[test]
    fn test_primer_beam_search_basic() {
        init_logger();
        let max_initial_beams = 3;

        // Create a simple test case
        let block_length = 10;

        let alphabet = vec![String::from("N"), String::from("A"), String::from("C"), String::from("G"), String::from("T")];

        let network_output = array![
            [0.4f32, 0.6, 0.0, 0.0, 0.0],
            [0.0f32, 0.2, 0.8, 0.0, 0.0],
            [0.0f32, 0.0, 1.0, 0.0, 0.0],
            [0.0f32, 0.0, 1.0, 0.0, 0.0],
            [0.0f32, 0.0, 0.0, 1.0, 0.0],
            [0.0f32, 0.5, 0.0, 0.0, 0.5],
            [0.0f32, 0.0, 1.0, 0.0, 0.0],
            [0.0f32, 0.0, 0.0, 1.0, 0.0],
            [0.0f32, 0.0, 0.0, 0.0, 1.0],
        ];
        let block_length = network_output.nrows();

        let primer_sequence = vec![0, 1, 2, 3]; // ACCCGT
        let primer_length = primer_sequence.len();

        let shift = (block_length - primer_length + 1) / max_initial_beams;

        // IDEALLY ENSURE THAT SHIFT MUST BE GREATER THAN 250

        let beam_size = 10;
        let beam_cut_threshold = 0.00001;
        let max_sample_depth = 6;
        let subsample = 1;

        let (result, total_score_computations) = primer_beam_search_brute(
            &network_output,
            5,
            0.0,
            &primer_sequence,
            max_sample_depth,
            shift,
            subsample,
        ).unwrap();

        //assert!(result.is_ok());
        let probs = result;//.unwrap();
        assert_eq!(probs.len(), block_length - primer_sequence.len() + 1);
        let correct_probability_scores = vec![0.3, 0.1, 0.0, 0.0, 0.0, 0.5];
        assert_eq!(probs, correct_probability_scores);
    }
}
