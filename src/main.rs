use std::io::BufRead;
use std::time::Instant;

use minitex::linebreak::*;

// Fonts
use font_kit::source::SystemSource;
use skribo::{FontCollection, FontFamily, TextStyle};

fn build_graph<D: std::fmt::Debug + Copy>(nodes: &[Node<D>],
                                          parameters: &Parameters) {
    let mut breakpoints = determine_breakpoints(nodes, &parameters);
    breakpoints.sort_by_key(|b| b.offset);
    println!("digraph {{");

    let mut last_box = None;
    for (offset, node) in nodes.iter().enumerate() {
        let breakpoint = if let Some(bp) = breakpoints.last() {
            if bp.offset == offset {
                breakpoints.pop()
            } else {
                None
            }
        } else {
            None
        };
        match node {
            Node::Box{ data, .. } => println!("n{} [label={:?}];", offset, data),
            _ => ()
        }
        if offset == 0 {
            continue;
        }
        println!("n{} -> n{};", offset - 1, offset);
        if let Node::Box{ .. } = node {
            last_box = Some(node);
        }
    }

    println!("}}");
}

fn main() {
    let stdin = std::io::stdin();
    let mut args = std::env::args();
    args.next();

    let mut tolerance = 100.0;
    let mut looseness = 0;
    let mut font_name = "DejaVu Sans".to_string();
    let mut hyphenate = true;
    while let Some(arg) = args.next() {
        if arg == "-l" {
            looseness = args.next().unwrap().parse().unwrap();
        } else if arg == "-t" {
            tolerance = args.next().unwrap().parse().unwrap();
        } else if arg == "-f" {
            font_name = args.next().unwrap();
        } else if arg == "-nohyphen" {
            hyphenate = false;
        }
    }

    let mut font_collection = FontCollection::new();
    let font_source = SystemSource::new();
    let font = font_source
        .select_by_postscript_name(&font_name)
        .unwrap()
        .load()
        .unwrap();
    font_collection.add_family(FontFamily::new_from_font(font));
    let text_style = TextStyle { size: 32.0 };

    let space_len = skribo::layout(&text_style, &font_collection, " ")
        .advance
        .x() as f64;

    let hyphen_len = skribo::layout(&text_style, &font_collection, "\u{2010}")
        .advance
        .x() as f64;

    let reader = std::io::BufReader::new(stdin);
    for line in reader.lines() {
        let line = line.unwrap();
        let words = line.split(" ");
        let mut nodes = Vec::new();

        for word in words {
            for syllable in word.split("\u{00ad}") {
                // Layout syllable
                let syllable_size = skribo::layout(&text_style, &font_collection, &syllable)
                    .advance
                    .x();
                nodes.push(Node::Box {
                    size: syllable_size as f64,
                    data: syllable,
                });
                if hyphenate {
                    nodes.extend(&Node::new_discretionary(
                        40.0, true, hyphen_len, "\u{2010}", 0.0, "", 0.0, "",
                    ));
                }
            }
            // Remove last hyphen penalty
            if hyphenate {
                nodes.pop();
                nodes.pop();
                nodes.pop();
            }
            nodes.push(Node::new_empty_penalty("\n"));
            nodes.push(Node::new_glue(
                space_len as f64,
                space_len as f64 / 3.0,
                space_len as f64 / 2.0,
            ));
        }

        // Remove last glue and empty penalty
        nodes.pop();
        nodes.pop();
        // Add hfill and forced break
        //nodes.push(Node::new_no_break(0.0, ""));
        nodes.push(Node::new_hfill(0.0));
        nodes.push(Node::new_forced_break(0.0, ""));

        let mut parameters = Parameters::new(60.0 * 32.0);
        parameters.looseness = looseness;
        parameters.tolerance = tolerance;

        //build_graph(&nodes, &parameters);
        //return;

        let now = Instant::now();
        let lines = layout(&nodes, &parameters);
        let elapsed = now.elapsed();
        for (breaking, line_nodes) in &lines {
            print!("[n={:<5}]", breaking.line_no);
            print!("[r={:+.2}]", breaking.adj_ratio);
            print!("[b={:+.2}]", breaking.badness());
            print!("[f={:?}] ", breaking.fitness_class);
            let linear_nodes = Node::linearize(line_nodes);
            for node in linear_nodes {
                if let Node::Box { data, .. } = node {
                    print!("{}", data);
                } else if let Node::Glue { .. } = node {
                    let glue_size = node.adjusted_size(breaking.adj_ratio) / space_len;
                    let glue_size = glue_size.round() as usize;
                    let glue_size = if glue_size == 0 { 1 } else { glue_size };
                    print!("{}", " ".repeat(glue_size));
                }
            }
            println!("");
        }
        println!("Total demerits: {}", lines.last().unwrap().0.demerits);
        println!(
            "Total badness: {}",
            lines.iter().map(|b| b.0.badness()).sum::<f64>()
        );
        println!("Layouting took {:?}", elapsed);
    }
}
