#[macro_use(s)]
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;

use std::fmt;
use std::collections::HashMap;
mod duplex;
mod search;
mod tree;
mod vec2d;
mod code_aware_beam_search;
mod primer_search;

#[cfg(feature = "fastexp")]
mod fastexp;

#[cfg(feature = "python")]
use {
    ndarray::Array2,
    numpy::PyArray1,
    numpy::PyArray2,
    numpy::PyArray3,
    pyo3::types::{PyDict, PySequence},
    pyo3::exceptions::{PyRuntimeError, PyValueError},
    pyo3::prelude::*,
    pyo3::wrap_pyfunction,
};
use crate::code_aware_beam_search::ConvCodeConfig;


#[cfg(feature = "wasm")]
extern crate serde_derive;
#[cfg(feature = "wasm")]
extern crate serde_json;
#[cfg(feature = "wasm")]
extern crate wasm_bindgen;
#[cfg(feature = "wasm")]
use {js_sys::Error, ndarray::ArrayView, serde_json::json, wasm_bindgen::prelude::*};

#[derive(Clone, Copy, Debug)]
pub enum SearchError {
    RanOutOfBeam,
    IncomparableValues,
    InvalidEnvelope,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::RanOutOfBeam => {
                write!(f, "Ran out of search space (beam_cut_threshold too high)")
            }
            SearchError::IncomparableValues => {
                write!(f, "Failed to compare values (NaNs in input?)")
            }
            // TODO: document envelope constraints
            SearchError::InvalidEnvelope => write!(f, "Invalid envelope values"),
        }
    }
}

impl std::error::Error for SearchError {}

// wasm
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn js_beam_search(
    network_output: &JsValue,
    alphabet: &JsValue,
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    shape: &JsValue,
) -> Result<JsValue, JsValue> {
    let shape: Vec<usize> = shape.into_serde().unwrap();
    let alphabet: Vec<String> = alphabet.into_serde().unwrap();
    let max_beam_cut = 1.0 / (alphabet.len() as f32);
    let network_output: Vec<f32> = network_output.into_serde().unwrap();
    let network_output = ArrayView::from_shape((shape[0], shape[1]), &network_output).unwrap();

    if beam_size == 0 {
        Error::new("beam_size cannot be 0");
        return Ok(JsValue::from_str("Error"));
    } else if beam_cut_threshold < -0.0 {
        Error::new("beam_cut_threshold must be at least 0.0");
        return Ok(JsValue::from_str("Error"));
    } else if beam_cut_threshold >= max_beam_cut {
        Error::new(&format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut.to_string()
        ));
        return Ok(JsValue::from_str("Error"));
    } else {
        let (seq, starts, total_comp) = search::beam_search(
            &network_output,
            &alphabet,
            beam_size,
            beam_cut_threshold,
            collapse_repeats,
        )
        .unwrap();

        let beam = json!({ "seq": seq, "starts": starts }).to_string();
        return Ok(JsValue::from_str(&beam));
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn js_viterbi_search(
    network_output: &JsValue,
    alphabet: &JsValue,
    qstring: bool,
    qscale: f32,
    qbias: f32,
    collapse_repeats: bool,
    shape: &JsValue,
) -> Result<JsValue, JsValue> {
    let shape: Vec<usize> = shape.into_serde().unwrap();
    let alphabet: Vec<String> = alphabet.into_serde().unwrap();
    let network_output: Vec<f32> = network_output.into_serde().unwrap();
    let network_output = ArrayView::from_shape((shape[0], shape[1]), &network_output).unwrap();

    if alphabet.is_empty() {
        Error::new("Empty alphabet given");
        return Ok(JsValue::from_str("Error"));
    } else if alphabet.len() != network_output.shape()[1] {
        Error::new("alphabet size does not match probability matrix dimensions");
        return Ok(JsValue::from_str("Error"));
    } else {
        let (seq, starts) = search::viterbi_search(
            &network_output,
            &alphabet,
            qstring,
            qscale,
            qbias,
            collapse_repeats,
        )
        .unwrap();

        let beam = json!({ "seq": seq, "starts": starts }).to_string();
        return Ok(JsValue::from_str(&beam));
    }
}

/// Python
#[cfg(feature = "python")]
fn seq_to_vec(seq: &PySequence) -> PyResult<Vec<String>> {
    Ok(seq.to_tuple()?.iter().map(|x| x.to_string()).collect())
}

/// Perform a Viterbi search decode on an RNN output.
///
/// This is the simplest possible approach to labelling - assign the highest probability label at
/// each time point.
///
/// See the module-level documentation for general requirements on `network_output` and `alphabet`.
///
/// Args:
///     network_output (numpy.ndarray): The 2D array output of the neural network.
///     alphabet (sequence): The labels (including the blank label, which must be first) in the
///         order given on the inner axis of `network_output`.
///     qstring (bool): If `True` a quality string will be appended to the returned sequence
///         which contains an ascii encoded phred quality score for each predicted label.
///     qscale (float): scale factor for the phred quality scores.
///     qbias (float): bias value for the phred quality scores.
///
/// Returns:
///     tuple of (str, numpy.ndarray): The decoded sequence and an array of the final
///         timepoints of each label (as indices into the outer axis of `network_output`).
///
/// Raises:
///     PyValueError: The constraints on the arguments have not been met.
#[cfg(feature = "python")]
#[pyfunction(
    qstring = false,
    qscale = "1.0",
    qbias = "0.0",
    collapse_repeats = true
)]
#[pyo3(
    text_signature = "(network_output, alphabet, qstring=False, qscale=1.0, qbias=0.0, collapse_repeats=True)"
)]
fn viterbi_search(
    py: Python,
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    qstring: bool,
    qscale: f32,
    qbias: f32,
    collapse_repeats: bool,
) -> PyResult<(String, Vec<usize>)> {
    let alphabet = seq_to_vec(alphabet)?;
    if alphabet.is_empty() {
        Err(PyValueError::new_err("Empty alphabet given"))
    } else if alphabet.len() != network_output.shape()[1] {
        Err(PyValueError::new_err(
            "alphabet size does not match probability matrix dimensions",
        ))
    } else {
        unsafe {
            let network_output = network_output.as_array();
            py.allow_threads(|| {
                search::viterbi_search(
                    &network_output,
                    &alphabet,
                    qstring,
                    qscale,
                    qbias,
                    collapse_repeats,
                )
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

#[cfg(feature = "python")]
#[pyfunction(qstring = false, qscale = "1.0", qbias = "0.0")]
#[pyo3(text_signature = "(network_output, init_state, alphabet)")]
fn crf_greedy_search(
    py: Python,
    network_output: &PyArray3<f32>,
    init_state: &PyArray1<f32>,
    alphabet: &PySequence,
    qstring: bool,
    qscale: f32,
    qbias: f32,
) -> PyResult<(String, Vec<usize>)> {
    let alphabet = seq_to_vec(alphabet)?;
    if alphabet.is_empty() {
        Err(PyValueError::new_err("Empty alphabet given"))
    } else if network_output.shape()[2] != alphabet.len() {
        Err(PyValueError::new_err(
            "alphabet size does not match probability matrix dimensions",
        ))
    } else {
        unsafe {
            let network_output = network_output.as_array();
            let init_state = init_state.as_array();
            py.allow_threads(|| {
                search::crf_greedy_search(
                    &network_output,
                    &init_state,
                    &alphabet,
                    qstring,
                    qscale,
                    qbias,
                )
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

#[cfg(feature = "python")]
#[pyfunction(beam_size = "5", beam_cut_threshold = "0.0")]
#[pyo3(text_signature = "(network_output, init_state, alphabet, beam_size, beam_cut_threshold)")]
fn crf_beam_search(
    py: Python,
    network_output: &PyArray3<f32>,
    init_state: &PyArray1<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> PyResult<(String, Vec<usize>)> {
    let alphabet = seq_to_vec(alphabet)?;
    if alphabet.is_empty() {
        Err(PyValueError::new_err("Empty alphabet given"))
    } else if network_output.shape()[2] != alphabet.len() {
        Err(PyValueError::new_err(
            "alphabet size does not match probability matrix dimensions",
        ))
    } else {
        unsafe {
            let network_output = network_output.as_array();
            let init_state = init_state.as_array();
            py.allow_threads(|| {
                search::crf_beam_search(
                    &network_output,
                    &init_state,
                    &alphabet,
                    beam_size,
                    beam_cut_threshold,
                )
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

/// Perform a CTC beam search decode on an RNN output.
///
/// This function does a beam search variant of the prefix search decoding mentioned (and described
/// in fairly vague terms) in the original CTC paper (Graves et al, 2006, section 3.2).
///
/// The paper mentioned above provides recursive equations that give an efficient way to find the
/// probability for a specific labelling. A tree of possible labelling suffixes, together with
/// their probabilities, can be built up by starting at one end and trying every possible label at
/// each stage. The "beam" part of the search is how we keep the search space managable - at each
/// step, we ignore all but the most-probable tree leaves (like searching with a torch beam). This
/// means we may not actually find the most likely labelling, but it often works very well.
///
/// See the module-level documentation for general requirements on `network_output` and `alphabet`.
///
/// Args:
///     network_output (numpy.ndarray): The 2D array output of the neural network.
///     alphabet (sequence): The labels (including the blank label, which must be first) in the
///         order given on the inner axis of `network_output`.
///     beam_size (int): How many search points should be kept at each step. Higher numbers are
///         less likely to discard the true labelling, but also make it slower and more memory
///         intensive. Must be at least 1.
///     beam_cut_threshold (float): Ignore any entries in `network_output` below this value. Must
///         be at least 0.0, and less than ``1/len(alphabet)``.
///
/// Returns:
///     tuple of (str, numpy.ndarray): The decoded sequence and an array of the final
///         timepoints of each label (as indices into the outer axis of `network_output`).
///
/// Raises:
///     PyValueError: The constraints on the arguments have not been met.
#[cfg(feature = "python")]
#[pyfunction(beam_size = "5", beam_cut_threshold = "0.0", collapse_repeats = true)]
#[pyo3(
    text_signature = "(network_output, alphabet, beam_size=5, beam_cut_threshold=0.0, collapse_repeats=True)"
)]
fn beam_search(
    py: Python,
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
) -> PyResult<(String, Vec<usize>, u64)> {
    let alphabet = seq_to_vec(alphabet)?;
    let max_beam_cut = 1.0 / (alphabet.len() as f32);
    if alphabet.len() != network_output.shape()[1] {
        Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output.shape()[1]
        )))
    } else if beam_size == 0 {
        Err(PyValueError::new_err("beam_size cannot be 0"))
    } else if beam_cut_threshold < -0.0 {
        Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ))
    } else if beam_cut_threshold >= max_beam_cut {
        Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )))
    } else {
        unsafe {
            let network_output = network_output.as_array();
            py.allow_threads(|| {
                search::beam_search(
                    &network_output,
                    &alphabet,
                    beam_size,
                    beam_cut_threshold,
                    collapse_repeats,
                )
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0",
    collapse_repeats = true
)]
#[pyo3(
    text_signature = "(network_output, alphabet, beam_size=5, beam_cut_threshold=0.0, collapse_repeats=True, forward_primer, reverse_primer, offset_sequence, conv_config)"
)]
fn convolutional_beam_search(
    py: Python,
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer_str: String,
    reverse_primer_str: String,
    offset_sequence_str: String,
    conv_config: &PyDict,
) -> PyResult<(String, Vec<usize>, u64)> {
    // Convert alphabet
    let alphabet = seq_to_vec(alphabet)?;

    // Convert primer sequences
    let forward_primer = dna_str_to_vec(&forward_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let reverse_primer = dna_str_to_vec(&reverse_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let offset_sequence = dna_str_to_vec(&offset_sequence_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert conv_config from PyDict to ConvCodeConfig
    let conv_config = pydict_to_conv_config(conv_config)?;

    // Validation
    let max_beam_cut = 1.0 / (alphabet.len() as f32);

    if alphabet.len() != network_output.shape()[1] {
        return Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output.shape()[1]
        )));
    }

    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }

    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }

    if beam_cut_threshold >= max_beam_cut {
        return Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )));
    }

    // Validate primer values are within alphabet range
    let max_alphabet_idx = alphabet.len() - 1;
    for (i, &val) in forward_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "forward_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    for (i, &val) in reverse_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "reverse_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    for (i, &val) in offset_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "offset_sequence[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    // Call the actual convolutional beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            code_aware_beam_search::convolutional_beam_search(
                &network_output,
                &alphabet,
                beam_size,
                beam_cut_threshold,
                collapse_repeats,
                &forward_primer,
                &reverse_primer,
                &offset_sequence,
                &conv_config,
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}

// fn for vanilla beam search log
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0",
    collapse_repeats = true
)]
#[pyo3(
    text_signature = "(network_output, alphabet, beam_size=5, payload_length, beam_cut_threshold=0.0, collapse_repeats=True, forward_primer, reverse_primer)"
)]
fn vanilla_beam_search_log(
    py: Python,
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    payload_length: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer_str: String,
    reverse_primer_str: String,
) -> PyResult<(String, Vec<usize>, f32, u64)> {
    // Convert alphabet
    let alphabet = seq_to_vec(alphabet)?;

    // Convert primer sequences
    let forward_primer = dna_str_to_vec(&forward_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let reverse_primer = dna_str_to_vec(&reverse_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Validation
    let max_beam_cut = 1.0 / (alphabet.len() as f32);

    if alphabet.len() != network_output.shape()[1] {
        return Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output.shape()[1]
        )));
    }

    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }

    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }

    if beam_cut_threshold >= max_beam_cut {
        return Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )));
    }

    // Validate primer values are within alphabet range
    let max_alphabet_idx = alphabet.len() - 1;
    for (i, &val) in forward_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "forward_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    for (i, &val) in reverse_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "reverse_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    // Call the actual vanilla beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            code_aware_beam_search::vanilla_beam_search_log(
                &network_output,
                &alphabet,
                beam_size,
                payload_length,
                beam_cut_threshold,
                collapse_repeats,
                &forward_primer,
                &reverse_primer,
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}


// fn for convolutional beam search log
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0",
    collapse_repeats = true
)]
#[pyo3(
    text_signature = "(network_output, alphabet, beam_size=5, beam_cut_threshold=0.0, collapse_repeats=True, forward_primer, reverse_primer, offset_sequence, conv_config)"
)]
fn convolutional_beam_search_log(
    py: Python,
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer_str: String,
    reverse_primer_str: String,
    offset_sequence_str: String,
    conv_config: &PyDict,
) -> PyResult<(String, Vec<usize>, f32, u64)> {
    // Convert alphabet
    let alphabet = seq_to_vec(alphabet)?;

    // Convert primer sequences
    let forward_primer = dna_str_to_vec(&forward_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let reverse_primer = dna_str_to_vec(&reverse_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let offset_sequence = dna_str_to_vec(&offset_sequence_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert conv_config from PyDict to ConvCodeConfig
    let conv_config = pydict_to_conv_config(conv_config)?;

    // Validation
    let max_beam_cut = 1.0 / (alphabet.len() as f32);

    if alphabet.len() != network_output.shape()[1] {
        return Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output.shape()[1]
        )));
    }

    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }

    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }

    if beam_cut_threshold >= max_beam_cut {
        return Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )));
    }

    // Validate primer values are within alphabet range
    let max_alphabet_idx = alphabet.len() - 1;
    for (i, &val) in forward_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "forward_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    for (i, &val) in reverse_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "reverse_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    for (i, &val) in offset_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "offset_sequence[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }

    // Call the actual convolutional beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            code_aware_beam_search::convolutional_beam_search_log(
                &network_output,
                &alphabet,
                beam_size,
                beam_cut_threshold,
                collapse_repeats,
                &forward_primer,
                &reverse_primer,
                &offset_sequence,
                &conv_config,
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}

// fn for marker convolutional beam search log
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0",
    collapse_repeats = true,
    marker_interval = "4"
)]
#[pyo3(
    text_signature = "(network_output, alphabet, beam_size=5, beam_cut_threshold=0.0, collapse_repeats=True, forward_primer, reverse_primer, offset_sequence, conv_config, marker_interval=4, marker_sequence)"
)]
fn marker_convolutional_beam_search_log(
    py: Python,
    network_output: &PyArray2<f32>,
    alphabet: &PySequence,
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
    forward_primer_str: String,
    reverse_primer_str: String,
    offset_sequence_str: String,
    conv_config: &PyDict,
    marker_interval: usize,
    marker_sequence_str: String,
) -> PyResult<(String, Vec<usize>, f32, u64)> {
    // Convert alphabet
    let alphabet = seq_to_vec(alphabet)?;

    // Convert primer sequences
    let forward_primer = dna_str_to_vec(&forward_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let reverse_primer = dna_str_to_vec(&reverse_primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let offset_sequence = dna_str_to_vec(&offset_sequence_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let marker_sequence = dna_str_to_vec(&marker_sequence_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Convert conv_config from PyDict to ConvCodeConfig
    let conv_config = pydict_to_conv_config(conv_config)?;

    // Validation
    let max_beam_cut = 1.0 / (alphabet.len() as f32);

    if alphabet.len() != network_output.shape()[1] {
        return Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output.shape()[1]
        )));
    }
    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }
    if marker_interval <= 0 {
        return Err(PyValueError::new_err("marker interval must be positive"));
    }
    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }
    if beam_cut_threshold >= max_beam_cut {
        return Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )));
    }
    // Validate primer values are within alphabet range
    let max_alphabet_idx = alphabet.len() - 1;
    for (i, &val) in forward_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "forward_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }
    for (i, &val) in reverse_primer.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "reverse_primer[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }
    for (i, &val) in offset_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "offset_sequence[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }
    for (i, &val) in marker_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "marker_sequence[{}] = {} exceeds alphabet size {}",
                i, val, alphabet.len()
            )));
        }
    }
    // Call the actual convolutional beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            code_aware_beam_search::marker_convolutional_beam_search_log(
                &network_output,
                &alphabet,
                beam_size,
                beam_cut_threshold,
                collapse_repeats,
                &forward_primer,
                &reverse_primer,
                &offset_sequence,
                &conv_config,
                marker_interval,
                &marker_sequence,
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}


// fn for primer beam search_brute
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0"
)]
#[pyo3(
    text_signature = "(network_output, beam_size=5, beam_cut_threshold=0.0, primer_sequence, max_sample_depth, shift)"
)]
fn primer_beam_search_brute(
    py: Python,
    network_output: &PyArray2<f32>,
    beam_size: usize,
    beam_cut_threshold: f32,
    primer_str: String,
    max_sample_depth: usize,
    shift: usize,
    subsample: usize,
) -> PyResult<(Vec<f32>, u64)> {
    // Convert primer sequence
    let primer_sequence = dna_str_to_vec(&primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Validation
    if network_output.shape()[1] != 5{
        return Err(PyValueError::new_err(format!(
            "Probability matrix must have inner dimension 5, instead has {}",
            network_output.shape()[1]
        )));
    }

    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }

    if subsample <= 0 {
        return Err(PyValueError::new_err("subsample must be greater than or equal to 1"));
    }

    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }

    // Validate primer values are within alphabet range
    let max_alphabet_idx = 3;
    for (i, &val) in primer_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "primer_sequence[{}] = {} exceeds alphabet size 3",
                i, val
            )));
        }
    }
    // Call the actual convolutional beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            primer_search::primer_beam_search_brute(
                &network_output,
                beam_size,
                beam_cut_threshold,
                &primer_sequence,
                max_sample_depth,
                shift,
                subsample
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}


// fn for primer beam search (with tricks)
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0"
)]
#[pyo3(
    text_signature = "(network_output, beam_size=5, beam_cut_threshold=0.0, primer_sequence, max_sample_depth, shift)"
)]
fn primer_beam_search_opt(
    py: Python,
    network_output: &PyArray2<f32>,
    beam_size: usize,
    fraction_to_examine: f32,
    beam_cut_threshold: f32,
    concentration_threshold: f32,
    primer_str: String,
    max_sample_depth: usize,
    shift: usize,
    subsample: usize,
) -> PyResult<(Vec<f32>, u64)> {
    // Convert primer sequence
    let primer_sequence = dna_str_to_vec(&primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Validation
    if network_output.shape()[1] != 5{
        return Err(PyValueError::new_err(format!(
            "Probability matrix must have inner dimension 5, instead has {}",
            network_output.shape()[1]
        )));
    }

    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }
    if fraction_to_examine < 0.0 || fraction_to_examine > 1.0 {
        return Err(PyValueError::new_err("fraction_to_examine must be between 0.0 and 1.0"));
    }
    if subsample <= 0 {
        return Err(PyValueError::new_err("subsample must be greater than or equal to 1"));
    }

    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }
    if concentration_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "concentration_threshold must be at least 0.0",
        ));
    }

    // Validate primer values are within alphabet range
    let max_alphabet_idx = 3;
    for (i, &val) in primer_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "primer_sequence[{}] = {} exceeds alphabet size 3",
                i, val
            )));
        }
    }
    // Call the actual convolutional beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            primer_search::primer_beam_search_opt(
                &network_output,
                beam_size,
                fraction_to_examine,
                beam_cut_threshold,
                concentration_threshold,
                &primer_sequence,
                max_sample_depth,
                shift,
                subsample
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}

// fn for primer beam search (smart subsampler)
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0"
)]
#[pyo3(
    text_signature = "(network_output, beam_size=5, beam_cut_threshold=0.0, primer_sequence, max_sample_depth, shift)"
)]
fn primer_beam_search_ss(
    py: Python,
    network_output: &PyArray2<f32>,
    beam_size: usize,
    min_beam_size: usize,
    beam_cut_threshold: f32,
    concentration_threshold: f32,
    relative_prob_threshold: f32,
    primer_str: String,
    max_sample_depth: usize,
    truncation_int1: usize,
    truncation_int2: usize,
    shift: usize,
    subsample: usize,
) -> PyResult<(Vec<f32>, u64)> {
    // Convert primer sequence
    let primer_sequence = dna_str_to_vec(&primer_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Validation
    if network_output.shape()[1] != 5{
        return Err(PyValueError::new_err(format!(
            "Probability matrix must have inner dimension 5, instead has {}",
            network_output.shape()[1]
        )));
    }

    if beam_size == 0 {
        return Err(PyValueError::new_err("beam_size cannot be 0"));
    }
    if min_beam_size == 0 {
        return Err(PyValueError::new_err("min_beam_size cannot be 0"));
    }
    if min_beam_size > beam_size {
        return Err(PyValueError::new_err("min_beam_size should be less than or equal to beam_size"));
    }
    if subsample <= 0 {
        return Err(PyValueError::new_err("subsample must be greater than or equal to 1"));
    }

    if beam_cut_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ));
    }
    if concentration_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "concentration_threshold must be at least 0.0",
        ));
    }
    if relative_prob_threshold < 0.0 {
        return Err(PyValueError::new_err(
            "relative_prob_threshold must be at least 0.0",
        ));
    }

    // Validate primer values are within alphabet range
    let max_alphabet_idx = 3;
    for (i, &val) in primer_sequence.iter().enumerate() {
        if val > max_alphabet_idx {
            return Err(PyValueError::new_err(format!(
                "primer_sequence[{}] = {} exceeds alphabet size 3",
                i, val
            )));
        }
    }
    // Call the actual convolutional beam search
    unsafe {
        let network_output = network_output.as_array();
        py.allow_threads(|| {
            primer_search::primer_beam_search_ss(
                &network_output,
                beam_size,
                min_beam_size,
                beam_cut_threshold,
                concentration_threshold,
                relative_prob_threshold,
                &primer_sequence,
                max_sample_depth,
                truncation_int1,
                truncation_int2,
                shift,
                subsample
            )
        })
    }
    .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
}


#[cfg(feature = "python")]
// Helper function to convert PySequence to Vec<usize>
fn seq_to_usize_vec(seq: &PySequence) -> PyResult<Vec<usize>> {
    let mut result = Vec::new();
    for item in seq.iter()? {
        let item = item?;
        let usize_item: usize = item.extract()?;
        result.push(usize_item);
    }
    Ok(result)
}

fn dna_str_to_vec(seq: &str) -> Result<Vec<usize>, String> {
    let mut vec = Vec::with_capacity(seq.len());

    for c in seq.chars() {
        let val = match c {
            'A' | 'a' => 0,
            'C' | 'c' => 1,
            'G' | 'g' => 2,
            'T' | 't' => 3,
            _ => return Err(format!("Invalid character: {}", c)),
        };
        vec.push(val);
    }

    Ok(vec)
}


#[cfg(feature = "python")]
// Helper function to convert PyDict to ConvCodeConfig
fn pydict_to_conv_config(dict: &PyDict) -> PyResult<ConvCodeConfig> {
    let k: usize = dict.get_item("K")
        .ok_or_else(|| PyValueError::new_err("Missing 'K' in conv_config"))?
        .extract()?;

    let b: usize = dict.get_item("b")
        .ok_or_else(|| PyValueError::new_err("Missing 'b' in conv_config"))?
        .extract()?;

    let c: usize = dict.get_item("c")
        .ok_or_else(|| PyValueError::new_err("Missing 'c' in conv_config"))?
        .extract()?;

    let df: usize = dict.get_item("df")
        .ok_or_else(|| PyValueError::new_err("Missing 'df' in conv_config"))?
        .extract()?;

    let ext: usize = dict.get_item("ext")
        .ok_or_else(|| PyValueError::new_err("Missing 'ext' in conv_config"))?
        .extract()?;

    let m: usize = dict.get_item("m")
        .ok_or_else(|| PyValueError::new_err("Missing 'm' in conv_config"))?
        .extract()?;

    let q: usize = dict.get_item("q")
        .ok_or_else(|| PyValueError::new_err("Missing 'q' in conv_config"))?
        .extract()?;

    let num_term_code_symbols: usize = dict.get_item("num_term_code_symbols")
        .ok_or_else(|| PyValueError::new_err("Missing 'num_term_code_symbols' in conv_config"))?
        .extract()?;

    // Convert num_edges
    let num_edges_py: &PySequence = dict.get_item("numEdges")
        .ok_or_else(|| PyValueError::new_err("Missing 'numEdges' in conv_config"))?
        .downcast()?;
    let num_edges = seq_to_usize_vec(num_edges_py)?;

    // Convert allowed_edges - this is more complex
    let allowed_edges_py: &PySequence = dict.get_item("allowedEdges")
        .ok_or_else(|| PyValueError::new_err("Missing 'allowedEdges' in conv_config"))?
        .downcast()?;
    let allowed_edges = convert_allowed_edges(allowed_edges_py)?;

    // Convert allowed_edges_term
    let allowed_edges_term_py: &PySequence = dict.get_item("allowedEdges_term")
        .ok_or_else(|| PyValueError::new_err("Missing 'allowedEdges_term' in conv_config"))?
        .downcast()?;
    let allowed_edges_term = convert_allowed_edges(allowed_edges_term_py)?;

    Ok(ConvCodeConfig {
        k,
        allowed_edges,
        allowed_edges_term,
        b,
        c,
        df,
        ext,
        m,
        num_edges,
        num_term_code_symbols,
        q,
    })
}
#[cfg(feature = "python")]
// Helper function to convert Python allowed_edges structure
fn convert_allowed_edges(seq: &PySequence) -> PyResult<Vec<HashMap<String, Vec<Vec<usize>>>>> {
    let mut result = Vec::new();

    for item in seq.iter()? {
        let item = item?;
        let dict: &PyDict = item.downcast()?;
        let mut edge_map = HashMap::new();

        for (key, value) in dict {
            let key_str: String = key.extract()?;
            let value_seq: &PySequence = value.downcast()?;

            let mut edge_list = Vec::new();
            for edge_item in value_seq.iter()? {
                let edge_item = edge_item?;
                let edge_seq: &PySequence = edge_item.downcast()?;
                let edge_vec = seq_to_usize_vec(edge_seq)?;
                edge_list.push(edge_vec);
            }

            edge_map.insert(key_str, edge_list);
        }

        result.push(edge_map);
    }

    Ok(result)
}



/// Perform a CTC beam search decode on two RNN outputs that describe the same sequence.
///
/// This is a variation of `beam_search` that attempts to find a common labelling for two RNN
/// outputs. This could be the same network run over two different samplings of the same sequence,
/// or two different networks run over the same input, for example.
///
/// It is an implementation of the algorithm developed by Silvestre-Ryan and Holmes
/// (https://doi.org/10.1101/2020.02.25.956771).
///
/// If no envelope is provided, a default one will be used. For now, that will just search the
/// whole of `network_output_2` at every step, but in future it may calculate a more constrained
/// envelope. For consistent results, you should provide an envelope.
///
/// Args:
///     network_output_1 (numpy.ndarray): The 2D array output of the first neural network.
///     network_output_2 (numpy.ndarray): The 2D array output of the second neural network. Note
///         that while the inner axis size must match that of `network_output_1`, the outer axis
///         can be a different size.
///     alphabet (str): The labels (including the blank label, which must be first) in the order
///         given on the inner axis of `network_output_1` and `network_output_2`.
///     envelope (numpy.ndarray, optional): An Nx2 array, where N is the outer axis length of
///         `network_output_1`. For each row of `network_output_1`, this gives the starting and
///         ending rows of `network_output_2` to consider for alignment.
///     beam_size (int): How many suffix_tree should be kept at each step. Higher numbers are less
///         likely to discard the true labelling, but also make it slower and more memory
///         intensive. Must be at least 1.
///     beam_cut_threshold (float): Ignore any entries in `network_output` below this value. Must
///         be at least 0.0, and less than ``1/len(alphabet)``.
///
/// Returns:
///     str: The decoded sequence.
///
/// Raises:
///     PyValueError: The constraints on the arguments have not been met.
#[cfg(feature = "python")]
#[pyfunction(
    beam_size = "5",
    beam_cut_threshold = "0.0",
    envelope = "None",
    collapse_repeats = true
)]
#[pyo3(
    text_signature = "(network_output_1, network_output_2, alphabet, envelope=None, beam_size=5, beam_cut_threshold=0.0, collapse_repeats=True)"
)]
fn beam_search_duplex(
    py: Python,
    network_output_1: &PyArray2<f32>,
    network_output_2: &PyArray2<f32>,
    alphabet: &PySequence,
    envelope: Option<&PyArray2<usize>>,
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
) -> PyResult<String> {
    let alphabet = seq_to_vec(alphabet)?;
    let max_beam_cut = 1.0 / (alphabet.len() as f32);
    if network_output_1.shape()[1] != network_output_2.shape()[1] {
        Err(PyValueError::new_err(
            "inner axes of the network outputs do not match",
        ))
    } else if alphabet.len() != network_output_1.shape()[1] {
        Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output_1.shape()[1]
        )))
    } else if beam_size == 0 {
        Err(PyValueError::new_err("beam_size cannot be 0"))
    } else if beam_cut_threshold < -0.0 {
        Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ))
    } else if beam_cut_threshold >= max_beam_cut {
        Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )))
    } else {
        if let Some(env) = envelope {
            if env.shape()[0] != network_output_1.shape()[0] {
                return Err(PyValueError::new_err(
                    "the lengths of network_output_1 and envelope do not match",
                ));
            } else if env.shape()[1] != 2 {
                return Err(PyValueError::new_err(
                    "the inner axis of envelope must have size 2",
                ));
            }
        }
        // if we need to construct an envelope, this holds it in scope while we're using it
        let default_envelope;
        unsafe {
            let envelope_view = match envelope {
                Some(env) => env.as_array(),
                None => {
                    default_envelope =
                        Array2::from_shape_fn((network_output_1.shape()[0], 2), |p| match p {
                            (_, 0) => 0,
                            (_, _) => network_output_2.shape()[0],
                        });
                    default_envelope.view()
                }
            };

            let network_output_1 = network_output_1.as_array();
            let network_output_2 = network_output_2.as_array();

            py.allow_threads(|| {
                duplex::beam_search(
                    &network_output_1,
                    &network_output_2,
                    &alphabet,
                    &envelope_view,
                    beam_size,
                    beam_cut_threshold,
                    collapse_repeats,
                )
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

#[cfg(feature = "python")]
#[pyfunction(beam_size = "5", beam_cut_threshold = "0.0", envelope = "None")]
#[pyo3(
    text_signature = "(network_output_1, init_state_1, network_output_2, init_state_2, alphabet, envelope=None, beam_size=5, beam_cut_threshold=0.0)"
)]
fn crf_beam_search_duplex(
    py: Python,
    network_output_1: &PyArray3<f32>,
    init_state_1: &PyArray1<f32>,
    network_output_2: &PyArray3<f32>,
    init_state_2: &PyArray1<f32>,
    alphabet: &PySequence,
    envelope: Option<&PyArray2<usize>>,
    beam_size: usize,
    beam_cut_threshold: f32,
) -> PyResult<String> {
    let alphabet = seq_to_vec(alphabet)?;
    let max_beam_cut = 1.0 / (alphabet.len() as f32);

    if network_output_1.shape()[2] != network_output_2.shape()[2] {
        Err(PyValueError::new_err(
            "inner axes of the network outputs do not match",
        ))
    } else if alphabet.len() != network_output_1.shape()[2] {
        Err(PyValueError::new_err(format!(
            "alphabet size {} does not match probability matrix inner dimension {}",
            alphabet.len(),
            network_output_1.shape()[1]
        )))
    } else if beam_size == 0 {
        Err(PyValueError::new_err("beam_size cannot be 0"))
    } else if beam_cut_threshold < -0.0 {
        Err(PyValueError::new_err(
            "beam_cut_threshold must be at least 0.0",
        ))
    } else if beam_cut_threshold >= max_beam_cut {
        Err(PyValueError::new_err(format!(
            "beam_cut_threshold cannot be more than {}",
            max_beam_cut
        )))
    } else {
        if let Some(env) = envelope {
            if env.shape()[0] != network_output_1.shape()[0] {
                return Err(PyValueError::new_err(
                    "the lengths of network_output_1 and envelope do not match",
                ));
            } else if env.shape()[1] != 2 {
                return Err(PyValueError::new_err(
                    "the inner axis of envelope must have size 2",
                ));
            }
        }
        // if we need to construct an envelope, this holds it in scope while we're using it
        let default_envelope;

        unsafe {
            let envelope_view = match envelope {
                Some(env) => env.as_array(),
                None => {
                    default_envelope =
                        Array2::from_shape_fn((network_output_1.shape()[0], 2), |p| match p {
                            (_, 0) => 0,
                            (_, _) => network_output_2.shape()[0],
                        });
                    default_envelope.view()
                }
            };

            let network_output_1 = network_output_1.as_array();
            let init_state_1 = init_state_1.as_array();
            let network_output_2 = network_output_2.as_array();
            let init_state_2 = init_state_2.as_array();

            py.allow_threads(|| {
                duplex::crf_beam_search(
                    &network_output_1,
                    &init_state_1,
                    &network_output_2,
                    &init_state_2,
                    &alphabet,
                    &envelope_view,
                    beam_size,
                    beam_cut_threshold,
                )
            })
        }
        .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
    }
}

/// Methods for labelling RNN results using CTC decoding.
///
/// The methods in this module implement the last step of labelling input data. In the case of
/// nanopore sequencing data, we're taking the electrical current samples, and labelling them with
/// what we think the DNA/RNA base is at any given time.
///
/// CTC decoding (and hence the funtions in this module) takes as input the result of a neural
/// network that has figured out, for each sample and each label, the probability that the sample
/// corresponds to that label. The network also outputs a probability for the data point
/// corresponding to an extra "blank" label (a sort of "none of the above" option). This is
/// represented as a 2D matrix of size ``N x (L+1)``, where ``N`` is the number of samples and
/// ``L`` is the number of labels we're interested in (the ``+1`` is to account for the blank
/// label).
///
/// A _path_ through the matrix is an assignment of a label or blank to each sample. The
/// probability that the path is correct is the product of the selected entries in the matrix. Each
/// path produces a labelling: first collapse all duplicate labels or blanks, then remove the
/// remaining blanks - AAAGGbGGbbbC would become AGbGbC, and then AGGC. The probability that the
/// labelling is correct is the sum of the probabilities of the paths that produce it. We want the
/// most likely labelling.
///
/// This problem is, in general, intractable. This module provides heuristic functions that attempt
/// to find the most likely labelling (but may produce a suboptimal labelling).
///
/// All functions take outputs from one or more neural networks, plus an alphabet to use for the
/// labelling.
///
/// The network outpus are 2D arrays produced by a softmax layer of a neural network, with values
/// between 0.0 and 1.0 representing probabilities. The outer axis (rows) is time, and the inner
/// axis (columns) is labels. The first entry on the label axis is assumed to be the blank label.
/// It's also worth noting that the values in each row should sum to 1.0.
///
/// The alphabet can be a str or any sequence of str (eg: a list or tuple of str). Each element (or
/// character in the case of str) provides the labelling for one element of the inner axis of the
/// network output(s) - therefore, len(alphabet) must be the size of that inner axis. Using a list
/// or tuple allows multi-character labels to be specified. Note that the first label is not
/// actually used by any of the functions in this module, so the value does not matter.

#[cfg(feature = "python")]
#[pymodule]
fn fast_ctc_decode(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(convolutional_beam_search))?;
    m.add_wrapped(wrap_pyfunction!(vanilla_beam_search_log))?;
    m.add_wrapped(wrap_pyfunction!(convolutional_beam_search_log))?;
    m.add_wrapped(wrap_pyfunction!(marker_convolutional_beam_search_log))?;
    m.add_wrapped(wrap_pyfunction!(primer_beam_search_brute))?;
    m.add_wrapped(wrap_pyfunction!(primer_beam_search_opt))?;
    m.add_wrapped(wrap_pyfunction!(primer_beam_search_ss))?;
    m.add_wrapped(wrap_pyfunction!(beam_search))?;
    m.add_wrapped(wrap_pyfunction!(beam_search_duplex))?;
    m.add_wrapped(wrap_pyfunction!(viterbi_search))?;
    m.add_wrapped(wrap_pyfunction!(crf_greedy_search))?;
    m.add_wrapped(wrap_pyfunction!(crf_beam_search))?;
    m.add_wrapped(wrap_pyfunction!(crf_beam_search_duplex))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
