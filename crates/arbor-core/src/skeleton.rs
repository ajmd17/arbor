use glam::Vec3;
use serde::Serialize;

#[derive(Clone, Debug, Default)]
pub struct SkeletonNode {
    pub parent: Option<u32>,
    pub children: Vec<u32>,
    pub position: Vec3,
    pub level: u8,
    pub path: u64,
    pub vigor: f32,
    pub stem_fraction: f32,
    pub radius: f32,
    /// Identifies the continuous run of segments this node belongs to. Nodes that
    /// share a `stem` form one unbroken tube; a change of `stem` between a node and
    /// its parent marks a real junction (branch or fork), not a mesh seam.
    pub stem: u32,
    /// Whether this node belongs to a stem the tree has lost.
    ///
    /// Dead wood still stands on a mature tree: it carries no leaves, and it ends in a
    /// break rather than tapering to a living tip. Set for a whole stem at once, since
    /// a branch dies as a unit.
    pub dead: bool,
    /// Whether this node has already broken off and is no longer part of the tree.
    ///
    /// Dead wood does not stand for ever: the thin, exposed end of a dead branch snaps
    /// and falls, leaving a stub. Broken nodes are skipped by everything downstream, so
    /// they are absent rather than merely dead.
    pub broken: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub nodes: Vec<SkeletonNode>,
}

impl Skeleton {
    #[allow(clippy::too_many_arguments)]
    pub fn push_node(
        &mut self,
        parent: Option<u32>,
        position: Vec3,
        level: u8,
        path: u64,
        vigor: f32,
        stem_fraction: f32,
        stem: u32,
    ) -> u32 {
        let index = self.nodes.len() as u32;
        self.nodes.push(SkeletonNode {
            parent,
            children: Vec::new(),
            position,
            level,
            path,
            vigor,
            stem_fraction,
            radius: 0.0,
            stem,
            dead: false,
            broken: false,
        });
        if let Some(p) = parent {
            self.nodes[p as usize].children.push(index);
        }
        index
    }

    pub fn segments(&self) -> impl Iterator<Item = (&SkeletonNode, &SkeletonNode)> {
        self.nodes
            .iter()
            .filter_map(|node| {
                node.parent
                    .map(|p| (&self.nodes[p as usize], node))
            })
    }

    /// Nodes grouped into continuous stems, ordered by the index of each stem's
    /// first node so the result is deterministic.
    pub fn stem_runs(&self) -> Vec<Vec<u32>> {
        let mut runs: Vec<Vec<u32>> = Vec::new();
        let mut slot: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            // Broken wood is gone, not merely dead, so nothing downstream ever sees it.
            if node.broken {
                continue;
            }
            let entry = *slot.entry(node.stem).or_insert_with(|| {
                runs.push(Vec::new());
                runs.len() - 1
            });
            runs[entry].push(i as u32);
        }
        runs.retain(|r| !r.is_empty());
        runs
    }

    /// How far the leader grew, measured along it from the ground. The trunk shares a
    /// stem with the root node, so this is that stem's polyline.
    pub fn leader_length(&self) -> f32 {
        let Some(root) = self.nodes.first() else {
            return 0.0;
        };
        let mut length = 0.0;
        let mut prev = root.position;
        for node in self.nodes.iter().skip(1).filter(|n| n.stem == root.stem) {
            length += (node.position - prev).length();
            prev = node.position;
        }
        length
    }

    pub fn stats(&self) -> SkeletonStats {
        let mut stats = SkeletonStats {
            node_count: self.nodes.len(),
            segment_count: self.nodes.len().saturating_sub(1),
            ..Default::default()
        };
        let mut stems = std::collections::HashSet::new();
        for node in &self.nodes {
            stats.height = stats.height.max(node.position.y);
            stats.max_radius = stats.max_radius.max(node.radius);
            let level = (node.level as usize).min(7);
            stats.per_level[level] += 1;
            stems.insert(node.stem);
            if node.parent.is_some() && node.children.is_empty() {
                stats.tip_count += 1;
            }
        }
        stats.stem_count = stems.len();
        stats
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SkeletonStats {
    pub node_count: usize,
    pub segment_count: usize,
    pub stem_count: usize,
    pub height: f32,
    pub max_radius: f32,
    pub tip_count: usize,
    pub per_level: [u32; 8],
}
