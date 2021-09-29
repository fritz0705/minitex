//! This module provides a fairly low-level implementation of the Knuth--Plass line breaking
//! algorithm, which is incorporated in the traditional TeX engines.
//!
//! Beginners should use the 'simple' function [layout].

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FitnessClass {
    /// Undefined fitness class, compatible to any fitness class.
    Undefined,
    Tight,
    Normal,
    Loose,
    VeryLoose,
}

impl FitnessClass {
    #[inline]
    fn from_adj_ratio(adj_ratio: f64) -> FitnessClass {
        if adj_ratio >= 1.0 {
            FitnessClass::VeryLoose
        } else if adj_ratio >= 0.5 {
            FitnessClass::Loose
        } else if adj_ratio >= -0.5 {
            FitnessClass::Normal
        } else {
            FitnessClass::Tight
        }
    }

    #[inline]
    fn is_compatible(self, other: FitnessClass) -> bool {
        match self {
            FitnessClass::Undefined => true,
            FitnessClass::Tight => {
                if let FitnessClass::Normal = other {
                    true
                } else {
                    false
                }
            }
            FitnessClass::Normal => {
                if let FitnessClass::VeryLoose = other {
                    false
                } else {
                    true
                }
            }
            FitnessClass::Loose => {
                if let FitnessClass::Tight = other {
                    false
                } else {
                    true
                }
            }
            FitnessClass::VeryLoose => {
                if let FitnessClass::Loose = other {
                    true
                } else {
                    false
                }
            }
        }
    }
}

/// A `Node` represents an unbreakable portion to the line-breaking algorithm
/// implemented in this module.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Node<D> {
    /// A `Box` represents some material with size given in attribute
    /// `.size` and user-defined data in attribute `.data`.
    Box { size: f64, data: D },
    /// A `Penalty` represents a position in a `Node` list that may be a
    /// valid breakpoint. The actual semantics are defined by the `.costs`
    /// attribute:
    ///
    /// If `.costs` is +inf, the node marks a point where a breakpoint is
    /// absolutely not acceptable; as a result, a breakpoint won't be set.
    /// If `.costs` is -inf, the node marks a point where a breakpoint is
    /// mandatory. This is typically used at the end of a node list.
    /// Otherwise the node marks a possible breakpoint with given `.costs`;
    /// those costs are taken into account when searching the best possible
    /// breakpoint in the following sense: A breakpoint is more feasible
    /// if the costs are lower, e.g. costs of 0 are more feasible than costs of 1.
    ///
    /// The `.flag` attribute controls two things: First it is used to
    /// prevent two consecutive breakpoints that are induced by a penalty
    /// with set flag. Second, it marks whether the penalty node is
    /// discardable.
    ///
    /// The `.size` attribute holds the size of material that is to be
    /// typeset. Its semantics are two-fold: If `.size` is positive or
    /// zero, material of `.size` is to be typeset when a break occurs at
    /// the penalty node; otherwise,
    ///
    /// Summarising, penalty nodes are magic: They can act as breakpoints
    /// and as typesettable material and their meaning depends heavily
    /// on their attribute values.
    Penalty {
        size: f64,
        data: D,
        costs: f64,
        flag: bool,
    },
    /// A `Glue` represents whitespace in a `Node` list.
    /// Its natural size is defined by the attribute `.size`,
    /// but is also stretchable and shrinkable by the amounts given in
    /// `.stretch` and `.shrink`, respectively.
    Glue {
        size: f64,
        shrink: f64,
        stretch: f64,
    },
}

impl<D> Node<D> {
    /// Create new glue node
    pub fn new_glue(size: f64, shrink: f64, stretch: f64) -> Node<D> {
        Node::Glue {
            size,
            shrink,
            stretch,
        }
    }

    /// Create a node which induces a forced break.
    pub fn new_forced_break(size: f64, data: D) -> Node<D> {
        Node::Penalty {
            size: size,
            data: data,
            costs: -f64::INFINITY,
            flag: true,
        }
    }

    /// Create a node which is forbidden to be broken.
    pub fn new_no_break(size: f64, data: D) -> Node<D> {
        Node::Penalty {
            size: size,
            data: data,
            costs: f64::INFINITY,
            flag: false,
        }
    }

    /// Creates an empty penalty, which is breakable without further costs.
    pub fn new_empty_penalty(data: D) -> Node<D> {
        Node::Penalty {
            size: 0.0,
            data: data,
            costs: 0.0,
            flag: false,
        }
    }

    /// Creates a new glue node, which stretches infinitely.
    pub fn new_hfill(fils: f64) -> Node<D> {
        Node::Glue {
            size: fils,
            stretch: f64::INFINITY,
            shrink: 0.0,
        }
    }

    /// Constructs nodes to emulate a discretionary hyphen, as known from TeX.
    ///
    /// A discretionary hyphen is an extended variant of a normal hyphen: It is associated with three
    /// contents: A content which shall be typeset before a line-break (called prebreak), a content
    /// which shall be typeset after a line-break at the begin of the next line (called postbreak),
    /// and a content which shall be typeset if there is no line-break at that point (called
    /// nobreak).
    ///
    /// This function constructs exactly three penalty nodes with appropriate costs. The first
    /// penalty node is to be placed directly into the nodes list, without preceeding glue. The
    /// second penalty node is to be placed after an optional glue, which might be used to
    /// stretch/shrink if there is no line-break at that point. The third penalty node is to be
    /// placed immediately after the second penalty node.
    ///
    /// The first returned node is also called a _prebreak node_, the second one is called a
    /// _nobreak node_ and the third one is called a _postbreak node_. It is notable that only the
    /// prebreak node and the nobreak node are discardable, while the postbreak node is not
    /// discardable.
    ///
    /// # Example
    ///
    /// To correctly break 'backen', which should break as 'bak-ken' in old German orthography, we could
    /// write the following:
    ///
    /// ```
    /// let mut nodes: Vec<Node<D>> = ...;
    /// // Assume that `nodes` contains a box for 'ba'
    /// nodes.append(Node::new_discretionary(80.0, true, 2, "k-", 0, "", 1, "c"));
    /// // And append a box for 'ken' to `nodes`.
    ///
    /// ```
    pub fn new_discretionary(
        costs: f64,
        flag: bool,
        prebreak_size: f64,
        prebreak: D,
        postbreak_size: f64,
        postbreak: D,
        nobreak_size: f64,
        nobreak: D,
    ) -> [Node<D>; 3] {
        return [
            Node::Penalty {
                size: prebreak_size,
                data: prebreak,
                costs,
                flag,
            },
            Node::Penalty {
                size: -nobreak_size,
                data: nobreak,
                costs: f64::INFINITY,
                flag: false,
            },
            Node::Penalty {
                size: postbreak_size,
                data: postbreak,
                costs: f64::INFINITY,
                flag: true,
            },
        ];
    }

    /// A node is a legal breakpoint if it is a penalty.
    pub fn is_legal_breakpoint(&self) -> bool {
        self.is_penalty()
    }

    /// A `Node` value is discardable if it is a glue, or a penalty that
    /// is not a forced breakpoint and not a forbidden breakpoint.
    ///
    /// Discardable nodes are to be discarded after a breakpoint.
    #[inline]
    pub fn is_discardable(&self) -> bool {
        self.is_glue()
            || (self.is_penalty()
                && !self.is_forced_break()
                && (!self.is_forbidden_break() || !self.is_flagged()))
        /*
        if self.is_glue() {
            true
        } else if self.is_penalty() {
            if self.is_forced_break() {
                false
            } else if self.is_forbidden_break() && self.is_flagged() {
                false
            } else {
                true
            }
        } else {
            false
        }
        */
    }

    /// A `Node` value is a forced break if it is a penalty node with `.costs == -inf`.
    #[inline]
    pub fn is_forced_break(&self) -> bool {
        match self {
            Node::Penalty { costs, .. } => costs.is_infinite() && *costs < 0.0,
            _ => false,
        }
    }

    /// A node is a forbidden break if it is a penalty node with `.costs == +inf`.
    #[inline]
    pub fn is_forbidden_break(&self) -> bool {
        match self {
            Node::Penalty { costs, .. } => costs.is_infinite() && *costs > 0.0,
            _ => true,
        }
    }

    pub fn is_flagged(&self) -> bool {
        match self {
            Node::Penalty { flag, .. } => *flag,
            _ => false,
        }
    }

    pub fn is_penalty(&self) -> bool {
        if let Node::Penalty { .. } = self {
            true
        } else {
            false
        }
    }

    pub fn is_glue(&self) -> bool {
        if let Node::Glue { .. } = self {
            true
        } else {
            false
        }
    }

    pub fn is_box(&self) -> bool {
        if let Node::Box { .. } = self {
            true
        } else {
            false
        }
    }

    /// A `Node` is inline, i.e. typesettable without restrictions, if it
    /// is either a box or a penalty, which is non-breakable and not
    /// flagged.
    pub fn is_inline(&self) -> bool {
        self.is_box() || (self.is_penalty() && self.is_forbidden_break() && !self.is_flagged())
    }

    /// The size of a node if it is typeset inside of a line.
    ///
    /// # Rules
    ///
    /// * For a `Node::Box`, the `inline_size` is `Node::Box.size`.
    /// * For a `Node::Glue`, the `inline_size` is `Node::Glue.size`.
    /// * For a `Node::Penalty`, the `inline_size` is `-Node::Penalty.size`
    ///   if `.size < 0.0`, otherwise `0.0`. This is useful in combination
    ///   with `.costs = +inf` and another preceeding penalty node.
    #[inline]
    pub fn inline_size(&self) -> f64 {
        match self {
            Node::Box { size, data: _ } => *size,
            Node::Glue { size, .. } => *size,
            Node::Penalty { size, .. } => {
                if *size < 0.0 {
                    -*size
                } else {
                    0.0
                }
            }
        }
    }

    /// The size of a node before a break under the assumption, that it induces
    /// a break.
    ///
    /// # Rules
    ///
    /// * For a `Node::Penalty`, the `prebreak_size` is `Node::Penalty.size`
    ///   if `.size > 0.0`, otherwise `0.0`.
    /// * For all other nodes, the `prebreak_size` is `0.0`.
    pub fn prebreak_size(&self) -> f64 {
        match self {
            Node::Penalty { size, .. } => {
                if *size > 0.0 {
                    *size
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// The size of a node if it stays at the beginning of a node list.
    pub fn postbreak_size(&self) -> f64 {
        match self {
            Node::Penalty { size, costs, .. } => {
                if *costs > 0.0 && costs.is_infinite() && *size > 0.0 {
                    *size
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// The stretchability of a `Node` inside a paragraph.
    #[inline]
    pub fn inline_stretch(&self) -> f64 {
        match self {
            Node::Glue { stretch, .. } => *stretch,
            _ => 0.0,
        }
    }

    /// The shrinkability of a `Node` inside a paragraph.
    #[inline]
    pub fn inline_shrink(&self) -> f64 {
        match self {
            Node::Glue { shrink, .. } => *shrink,
            _ => 0.0,
        }
    }

    /// Size of the node after applying the adjustment ratio.
    pub fn adjusted_size(&self, adj_ratio: f64) -> f64 {
        if adj_ratio > 0.0 {
            self.inline_size() + adj_ratio * self.inline_stretch()
        } else {
            self.inline_size() + adj_ratio * self.inline_shrink()
        }
    }

    /// Costs of a break of a node. Unpacks the costs from a `Node::Penalty`
    /// value, otherwise returns `f64::INFINITY` to indicate that a break is
    /// not acceptable.
    pub fn break_costs(&self) -> f64 {
        match self {
            Node::Penalty { costs, .. } => *costs,
            _ => f64::INFINITY,
        }
    }

    /// Strips discardable nodes from a slice of nodes.
    pub fn strip_discardable(nodes: &[Node<D>]) -> (usize, &[Node<D>]) {
        let mut nodes = nodes;
        let mut stripped = 0;
        while nodes.len() > 0 && nodes[0].is_discardable() {
            nodes = &nodes[1..];
            stripped += 1;
        }
        (stripped, nodes)
    }
}

impl<D: Copy> Node<D> {
    /// Linearizes a node list.
    ///
    /// A node list is _linear_ iff it consists of boxes and glues.
    pub fn linearize(nodes: &[Node<D>]) -> Vec<Node<D>> {
        let mut linear_nodes: Vec<Node<D>> = Vec::new();

        if let Some(&Node::Penalty { data, size, .. }) = nodes.first() {
            linear_nodes.push(Node::Box { data, size });
        }

        for node in nodes {
            match node {
                &Node::Box { .. } | &Node::Glue { .. } => linear_nodes.push(*node),
                &Node::Penalty { data, size, .. } if node.is_inline() => {
                    linear_nodes.push(Node::Box { data, size })
                }
                _ => (),
            }
        }

        if let Some(&Node::Penalty {
            size,
            data,
            flag: true,
            ..
        }) = nodes.last()
        {
            linear_nodes.push(Node::Box { data, size });
        }

        linear_nodes
    }
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct BreakPoint {
    pub offset: usize,
    pub previous_offset: usize,
    pub demerits: f64,
    pub adj_ratio: f64,
    pub line_no: usize,
    pub fitness_class: FitnessClass,
    pub flagged: bool,
    pub forced: bool,
}

impl BreakPoint {
    /// The badness of the breakpoint.
    pub fn badness(&self) -> f64 {
        if self.adj_ratio < -1.0 {
            f64::INFINITY
        } else {
            100.0 * self.adj_ratio.abs().powi(3)
        }
    }

    fn key(&self) -> (usize, usize, FitnessClass) {
        (self.offset, self.line_no, self.fitness_class)
    }

    /// Method which compares (offset, -demerits) of two breakpoints.
    pub fn cmp_offset_demerits(&self, other: &BreakPoint) -> Ordering {
        self.offset.cmp(&other.offset).then(
            self.demerits
                .partial_cmp(&other.demerits)
                .unwrap()
                .reverse(),
        )
    }

    pub fn join(&self, rhs: &Self) -> Self{
        BreakPoint {
            offset: self.offset + rhs.offset,
            demerits: self.demerits + rhs.demerits,
            adj_ratio: rhs.adj_ratio,
            line_no: self.line_no + rhs.line_no,
            fitness_class: rhs.fitness_class,
            previous_offset: self.offset,
            flagged: rhs.flagged,
            forced: rhs.forced,
        }
    }

    /// If a breakpoint has adjustment ratio lower than 1, it will result in a overfull line.
    pub fn is_overfull(&self) -> bool {
        self.adj_ratio < -1.0
    }
}

/// Line-breaking parameters.
#[derive(PartialEq, Clone)]
pub struct Parameters {
    pub desired_lengths: Vec<f64>,
    pub default_length: f64,
    pub flag_demerits: f64,
    pub fitness_demerits: f64,
    pub tolerance: f64,
    pub max_demerits: f64,
    pub line_penalty: f64,
    pub looseness: isize,
}

impl Parameters {
    /// Return `Parameters` value with `default_length` and default plain
    /// TeX parameters.
    pub fn new(default_length: f64) -> Parameters {
        Parameters {
            desired_lengths: vec![],
            default_length,
            flag_demerits: 10_000.0,
            fitness_demerits: 10_000.0,
            tolerance: 100.0,
            max_demerits: f64::INFINITY,
            line_penalty: 10.0,
            looseness: 0,
        }
    }

    /// Return the line length for a given line number.
    pub fn line_length(&self, line_no: usize) -> f64 {
        match self.desired_lengths.get(line_no) {
            Some(n) => *n,
            _ => self.default_length,
        }
    }
}

/// Find subsequent acceptable breakpoints for a nodes slice.
pub fn subseq_breakpoints<D>(
    nodes: &[Node<D>],
    desired_length: f64, // TODO Bad API design???
    previous_fitness_class: FitnessClass,
    previous_flagged: bool,
    parameters: &Parameters,
) -> Vec<BreakPoint> {
    let mut ret = Vec::new();

    let mut sum_shrink = 0.0;
    let mut sum_stretch = 0.0;
    let mut sum_length = if let Some(node) = nodes.first() {
        node.postbreak_size()
    } else {
        0.0
    };

    for (offset, node) in nodes.iter().enumerate() {
        sum_length += node.inline_size();
        sum_shrink += node.inline_shrink();
        sum_stretch += node.inline_stretch();

        if !node.is_legal_breakpoint() || node.is_forbidden_break() {
            continue;
        }

        // Determine various parameters
        let force_break = node.is_forced_break();
        let actual_length = sum_length - node.inline_size() + node.prebreak_size();

        let adj_ratio = if actual_length < desired_length {
            (desired_length - actual_length) / sum_stretch
        } else if actual_length > desired_length {
            (desired_length - actual_length) / sum_shrink
        } else {
            0.0
        };

        let fitness_class = FitnessClass::from_adj_ratio(adj_ratio);

        let break_costs = node.break_costs();
        let demerits = if break_costs >= 0.0 {
            (parameters.line_penalty + 100.0 * adj_ratio.abs().powi(3) + break_costs).powi(2)
        } else if !break_costs.is_infinite() {
            (parameters.line_penalty + 100.0 * adj_ratio.abs().powi(3)).powi(2)
                - break_costs.powi(2)
        } else {
            (parameters.line_penalty + 100.0 * adj_ratio.abs().powi(3)).powi(2)
        } + if !fitness_class.is_compatible(previous_fitness_class) {
            parameters.fitness_demerits
        } else {
            0.0
        } + if node.is_flagged() && previous_flagged {
            parameters.flag_demerits
        } else {
            0.0
        };

        // Construct BreakPoint value
        let breakpoint = BreakPoint {
            offset: offset,
            previous_offset: 0,
            demerits: demerits,
            adj_ratio: adj_ratio,
            line_no: 1,
            fitness_class: fitness_class,
            flagged: node.is_flagged(),
            forced: force_break,
        };

        if force_break {
            // The break is forced. Thus we will return it and exit the loop.
            ret.push(breakpoint);
            break;
        } else if adj_ratio >= -1.0 && breakpoint.badness() <= parameters.tolerance {
            // The break is feasible.
            ret.push(breakpoint)
        }
    }

    ret
}

#[inline]
fn select_optimal_breakpoint(
    breakpoints: &mut Vec<BreakPoint>,
    parameters: &Parameters,
) -> BreakPoint {
    let mut forced_breaks: Vec<BreakPoint> = Vec::new();

    // Sort breakpoint list by offsets and demerits
    breakpoints.sort_by(BreakPoint::cmp_offset_demerits);
    // Remove all forced breaks from `breakpoints` and record them in `forced_breaks`.
    while let Some(breakpoint) = breakpoints.last() {
        if breakpoint.forced {
            forced_breaks.push(*breakpoint);
            breakpoints.pop();
        } else {
            break;
        }
    }
    // Now `breakpoints` should contain no further forced breaks.

    if forced_breaks.is_empty() {
        // But if `breakpoints` vector does not contain any forced breaks, panic.
        panic!("No forced breakpoint found!");
    }

    // The optimal line count is the line number of the best break minus 1.
    let optimal_line_count = forced_breaks[0].line_no - 1;
    // Desired line count, at least 1.
    let desired_line_count = (optimal_line_count as isize + parameters.looseness).max(1);

    // Sort forced breaks by difference between actual line number and desired line count.
    forced_breaks.sort_by_key(|b| (b.line_no as isize - 1 - desired_line_count).abs());

    // Now forced_breaks[0] is the best forced break w.r.t. the looseness.
    forced_breaks[0]
}

/// Given acceptable breakpoints, construct an optimal breakpoint sequence.
///
/// # Arguments
///
/// * `breakpoints` – A vector that contains `BreakPoint` values.  The vector
///   must contain at least one forced `BreakPoint`.  Usually such breakpoint
///   declares the end of a paragraph to be laid out.
/// * `parameters` – A `Parameters` value defining the typesetting parameters.
///    This function honours the attributes `.tolerance` and `.looseness`.
pub fn build_lines(mut breakpoints: Vec<BreakPoint>, parameters: &Parameters) -> Vec<BreakPoint> {
    // First we choose an optimal final breakpoint w.r.t. the looseness and tolerance
    // determined by `parameters`.
    let optimal_break = select_optimal_breakpoint(&mut breakpoints, parameters);

    let mut lines: Vec<BreakPoint> = vec![optimal_break];

    while let Some(breakpoint) = breakpoints.pop() {
        let last_breakpoint = lines.last().unwrap();
        if last_breakpoint.line_no - 1 == breakpoint.line_no
            && last_breakpoint.previous_offset == breakpoint.offset
        {
            lines.push(breakpoint);
        }
    }

    // The lines are in wrong order. Therefore, reverse the vector of lines.
    lines.reverse();
    lines
}

#[inline]
fn replace_if_better(vec: &mut Vec<BreakPoint>, breakpoint: &BreakPoint) -> bool {
    // Perform binary search with reversed key order relation.
    match vec.binary_search_by(|b| b.key().cmp(&breakpoint.key()).reverse()) {
        Ok(offset) => {
            if vec[offset].demerits > breakpoint.demerits {
                // Replace breakpoint in vec, since it is better than the old breakpoint
                vec[offset] = *breakpoint;
                true
            } else {
                false
            }
        }
        Err(offset) => {
            // Insert breakpoint, since it is not contained in the vector.
            vec.insert(offset, *breakpoint);
            true
        }
    }
}

/// Determine a vector of feasible breakpoints to layout an paragraph up to the first forced
/// line break.
///
/// # Arguments
///
/// * `nodes` – A slice of `Node` values. Must contain a forced line break somewhere.
/// * `parameters` – A `Parameters` value, which contains typesetting parameters.
///
/// # Return value
///
/// This function returns a vector of feasible BreakPoint objects.
pub fn determine_breakpoints<D>(nodes: &[Node<D>], parameters: &Parameters) -> Vec<BreakPoint> {
    // We start with some active breaks, which are derived by running subseq_breakpoints
    // with the beginning of the paragraph.
    let mut active_breaks = subseq_breakpoints(
        nodes,
        parameters.line_length(0),
        FitnessClass::Undefined,
        false,
        &parameters,
    );
    // Further we initialize this vector with a capacity, since it will hold many breakpoints
    // at the end.
    let mut passive_breaks = Vec::with_capacity(nodes.len());
    // Specify max demerits in this context to allow adaptive control
    let mut max_demerits = parameters.max_demerits;
    // TODO Adaptive control for tolerance -- We could try to keep the tolerance low by
    // increasing and backtracing if there is no feasible forced break.

    // Up to this point, the list of active breakpoints `active_break` contains
    // breakpoints, which are forced and are, therefore, no
    // longer interesting for the algorithm. We have to move them to `passive_breaks`.
    while let Some(&active_break) = active_breaks.last() {
        if active_break.forced {
            passive_breaks.push(active_breaks.pop().unwrap());
        } else {
            break;
        }
    }

    // Still, active_breaks is unsorted, but we need a sorted vector for binary searches.
    active_breaks.sort_by(|a, b| a.key().cmp(&b.key()).reverse());

    // Main loop: We take some active breakpoint from the vector (think of it as a queue)
    // and try to find the next feasible breakpoints after that active breakpoint.
    while let Some(active_break) = active_breaks.pop() {
        // Prepare the slice of nodes after the breakpoint...
        let remaining_nodes = &nodes[active_break.offset..];
        // As it holds discardable nodes at the beginning, we have to strip them before
        // continuing. We also remember the number of stripped nodes to adjust the
        // new breakpoints afterwards.
        let (n_stripped, remaining_nodes) = Node::strip_discardable(remaining_nodes);

        // Use the subseq_breakpoints function to find feasible breakpoints after the current
        // breakpoint. We provide some information about the current breakpoint to the
        // algorithm in subseq_breakpoints.
        let breakpoints = subseq_breakpoints(
            remaining_nodes,
            parameters.line_length(active_break.line_no),
            active_break.fitness_class,
            active_break.flagged,
            &parameters,
        );

        // We only make the current breakpoint passive, if it is feasible in the following
        // sense: It either yields a forced breakpoint that is feasible, or there is some
        // breakpoint that satisfies the adaptive max demerits control condition.
        let mut is_feasible = false;
        for breakpoint in breakpoints {
            // Here `breakpoint` is a breakpoint after the current, active breakpoint, which
            // is feasible for `subseq_breakpoints`. Since we used a modified slice to execute
            // subseq_breakpoints, we have to adjust the offsets, line numbers and demerits of
            // `breakpoint` to reflect the actual values. This is done by a call to join, which
            // 'joins' two breakpoints.
            let mut breakpoint = active_break.join(&breakpoint);
            // Unfortunately the preceeding operation does not account for the stripped,
            // discardable nodes from above. Therefore, we have to adjust the offset of
            // `breakpoint`.
            breakpoint.offset += n_stripped;

            if breakpoint.forced {
                // If the breakpoint is forced, it seems to be feasible (at least for a last
                // resort to an overfull line). Since a forced breakpoint is a 'leaf node'
                // for the algorithm, we directly feed it into the list of passive breaks,
                // and won't repeat the whole procedure for active breakpoints for this one.
                replace_if_better(&mut passive_breaks, &breakpoint);
                // If the looseness is zero, we use adaptive max demerits control, meaning that
                // we reduce the max_demerits variable if we encounter a forced breakpoint
                // with lower demerits.
                if parameters.looseness == 0 {
                    max_demerits = breakpoint.demerits.min(max_demerits);
                }
                if breakpoint.badness() <= parameters.tolerance && breakpoint.demerits <= max_demerits{
                    is_feasible = true;
                }
            } else if breakpoint.demerits <= max_demerits {
                // If the breakpoint satisfies the (adaptive) demerits maximum condition,
                // proceed with the following test: If there is already a passive breakpoint
                // with same fitness class, line number and offset, and lower demerits, do not
                // add this breakpoint to the list of active breakpoints: It is useless to do
                // this whole procedure for the breakpoint, which is nonetheless too bad
                // compared to a present one.
                if let Ok(offset) =
                    passive_breaks.binary_search_by(|b| b.key().cmp(&breakpoint.key()).reverse())
                {
                    if passive_breaks[offset].demerits <= breakpoint.demerits {
                        continue;
                    }
                }
                // But otherwise, we try to insert the breakpoint to the list of active breaks.
                if replace_if_better(&mut active_breaks, &breakpoint) {
                    is_feasible = true;
                }
            }
        }
        // And when the current, active breakpoint is feasible, which means that there is a
        // succeeding breakpoint which is feasible, we make it passive.
        if is_feasible {
            // Actually we maintain the lists `passive_breaks` and `active_breaks` with the
            // helper function replace_if_better. It only inserts a breakpoint into the list
            // if there is no other breakpoint with same 'key', i.e. fitness class, line number
            // and offset, or if its total demerits are a-priori minimal.
            replace_if_better(&mut passive_breaks, &active_break);
        }
    }
    passive_breaks
}

/// Layout a slice of nodes to lines in an optimal way. Returns a vector of (stripped) lines.
pub fn layout<D: Copy>(
    nodes: &[Node<D>],
    parameters: &Parameters,
) -> Vec<(BreakPoint, Vec<Node<D>>)> {
    let mut lines = Vec::new();
    let breakings = determine_breakpoints(&nodes, &parameters);
    let breakings = build_lines(breakings, &parameters);
    for breaking in breakings {
        let line_nodes = &nodes[breaking.previous_offset..breaking.offset + 1];
        let (_, line_nodes) = Node::strip_discardable(line_nodes);

        lines.push((breaking, line_nodes.to_vec()));
    }
    lines
}
