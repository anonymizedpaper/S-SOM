from vtkmodules import vtkFiltersExtraction
import heapq
import numpy as np
from sklearn.preprocessing import normalize
import pyvista as pv
from collections import defaultdict


class SphereSOM3D:
    def __init__(self, mesh: pv.PolyData, radius: float = 0.1):
        self.mesh = mesh
        self.radius = radius
        self.mesh.points = normalize(mesh.points)  # nodes start on the unit sphere
        # The lattice topology is FIXED during training, so the point-point
        # adjacency and every n-ring neighborhood are constant. Cache both:
        # pyvista's point_neighbors() was being called once per node per epoch.
        self._adj = None          # lazily-built adjacency list
        self._ring_cache = {}     # memoized n-ring sets, keyed by (mu_idx, n)

    # def _initialize_nodes(self, n_dim: int = 3):
    #     print("Initializing SOM nodes...")
    #     #self.weights = self.mesh.point_normals/5
    #     # Node positions live on the unit sphere; (re)initialize them from the mesh points.
    #     #self.mesh.points = normalize(self.mesh.points)
    def _adjacency(self):
        """Build & cache the fixed point-point adjacency list once."""
        if self._adj is None:
            self._adj = [set(int(j) for j in self.mesh.point_neighbors(i))
                         for i in range(self.mesh.n_points)]
        return self._adj

    def get_n_ring_neighbors(self, mu_idx, n):
        key = (int(mu_idx), int(n))
        cached = self._ring_cache.get(key)
        if cached is not None:
            return cached
        adj = self._adjacency()
        visited = set()
        current_ring = {int(mu_idx)}
        all_neighbors = {int(mu_idx)}   # include the BMU (center) itself
        for _ in range(n):
            next_ring = set()
            for vid in current_ring:
                next_ring.update(adj[vid])
            next_ring -= visited
            all_neighbors.update(next_ring)
            visited.update(current_ring)
            current_ring = next_ring
        self._ring_cache[key] = all_neighbors  # read-only by callers
        return all_neighbors


    def train(self, data: np.ndarray, n_epochs: int = 1000, n_rings: int = 0, init_neuron_size: float = 2,  lr0: float = 0.1, lr_end: float = 0.01, verbose: bool = True):
        """Train the spherical SOM.

        Neighborhood: an n-ring window on the FIXED mesh topology around the BMU.
        The window starts at `n_rings` rings and shrinks linearly to a single ring
        over training (coarse ordering first, fine-tuning last). Ring distance is
        the mesh graph distance, so it stays meaningful as the nodes drift.
        Influence within the window is Gaussian over ring distance.

        n_rings = 0  -> get_n_ring_neighbors returns just the BMU itself (no neighborhood, equivalent to vanilla k-means)
        The learning rate decays exponentially from lr0 to lr_end.
        """
        self.mesh.points = normalize(self.mesh.points)*init_neuron_size
        points = self.mesh.points


        # n-ring window around the BMU.

        if verbose:
            print(f"Training SOM with {n_epochs} epochs, initial radius {n_rings} rings, "
                  f"learning rate {lr0} decaying to {lr_end}...")

        for epoch in range(n_epochs):
            frac = epoch / n_epochs
            lr    = lr0 * (lr_end / lr0) ** frac

            x = data[np.random.randint(0, len(data))]   # data is assumed already unit-normalized
            bmu_idx = np.argmin(np.linalg.norm(points - x, axis=1))
            bmu_pos = points[bmu_idx]
            searched_nodes = self.get_n_ring_neighbors(bmu_idx, n_rings)
            for i in searched_nodes:
                dist = np.linalg.norm(points[i] - bmu_pos)
                if dist <= self.radius:
                    influence = np.exp(-dist**2 / (2 * self.radius**2))
                    points[i] += lr * influence * (x - points[i])
        self.mesh.points = points

    def predict(self, data):
        # Euclidean nearest node (BMU), consistent with training.
        # ||d - p||^2 = ||d||^2 - 2 d.p + ||p||^2  (the ||d||^2 term is constant per row)
        data = np.asarray(data)
        sq_dists = (np.sum(data**2, axis=1)[:, None]
                    - 2.0 * (data @ self.mesh.points.T)
                    + np.sum(self.mesh.points**2, axis=1)[None, :])
        return np.argmin(sq_dists, axis=1)

    def get_weights(self) -> np.ndarray:
        return self.mesh.points

    def get_mesh(self) -> pv.PolyData:
        return self.mesh
    

def load_obj_with_face_normals(obj_path):
    mesh = pv.read(obj_path)    # Load the mesh
    # Per-face normals only; the 3D pipeline does not use face centers, so we skip
    # the cell_centers() pass (it is only needed by the older 2D pipelines).
    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    return mesh

def smooth_normals(mesh, face_adjacency, n_iter=5, feature_angle_deg=30.0):
    """Feature-preserving normal-field smoothing (denoises the SOM input).

    Iteratively replaces each face normal with the average of itself and its
    1-ring face neighbors *whose normal differs by less than feature_angle_deg*.
    Vertices are NOT moved, so sharp edges/creases stay crisp while per-facet
    jitter (the cause of salt-and-pepper labels) is removed.

    Updates mesh.cell_data['Normals'] in place and returns the smoothed array.
    Set n_iter=0 to disable.
    """
    normals = normalize(mesh.cell_data['Normals'].copy())
    if n_iter <= 0:
        return normals
    cos_thresh = np.cos(np.deg2rad(feature_angle_deg))
    n_faces = mesh.n_cells

    # Build a padded (n_faces, max_deg) neighbor-index matrix once, plus a validity
    # mask. Padding slots point at index 0 and are masked out, so they contribute 0.
    nbr_lists = [list(face_adjacency.get(f, ())) for f in range(n_faces)]
    max_deg = max((len(nb) for nb in nbr_lists), default=0)
    if max_deg == 0:
        return normals
    nbr_idx = np.zeros((n_faces, max_deg), dtype=np.intp)
    nbr_mask = np.zeros((n_faces, max_deg), dtype=bool)
    for f, nb in enumerate(nbr_lists):
        if nb:
            nbr_idx[f, :len(nb)] = nb
            nbr_mask[f, :len(nb)] = True

    for _ in range(n_iter):
        nbr_normals = normals[nbr_idx]                          # (F, K, 3)
        sims = np.einsum('fkd,fd->fk', nbr_normals, normals)    # (F, K) cos with center
        keep = nbr_mask & (sims >= cos_thresh)                  # same facet/feature only
        summed = np.einsum('fk,fkd->fd', keep, nbr_normals)     # sum of kept neighbors
        normals = normalize(normals + summed)                   # include self, renormalize
    mesh.cell_data['Normals'] = normals
    return normals

def build_face_adjacency(mesh):
    """Returns a dict: face index -> set of edge-adjacent face indices.

    Vectorized for triangle meshes: build the 3 undirected edges of every face as
    sorted vertex pairs, sort them, and any two consecutive identical edges are a
    shared-edge face pair (manifold assumption: each interior edge joins 2 faces).
    """
    faces = mesh.faces.reshape((-1, 4))[:, 1:]            # (F, 3) vertex ids (triangles)
    F = len(faces)

    # 3F undirected edges as canonical (min, max) vertex pairs + the owning face id.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    face_of_edge = np.tile(np.arange(F), 3)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e = edges[order]
    fo = face_of_edge[order]
    # consecutive rows with the same (v0, v1) share that edge -> adjacent faces
    same = np.where((e[1:, 0] == e[:-1, 0]) & (e[1:, 1] == e[:-1, 1]))[0]
    a = fo[same]
    b = fo[same + 1]

    adjacency = defaultdict(set)
    for x, y in zip(a.tolist(), b.tolist()):
        adjacency[x].add(y)
        adjacency[y].add(x)
    return adjacency

def remap_labels(labels, face_adjacency=None, mesh=None):
    """Remaps region labels to 0-based integers.

    When face_adjacency is provided (or auto_compute_adjacency=True with a mesh),
    IDs are assigned so that adjacent regions receive values as far apart as
    possible (better colormap contrast).  Falls back to sequential remapping
    when no adjacency information is available.
    """
    unique_labels = np.unique(labels)
    n = len(unique_labels)
    old_to_idx = {label: i for i, label in enumerate(unique_labels)}

    if face_adjacency is None and mesh is not None:
        face_adjacency = build_face_adjacency(mesh)

    if face_adjacency is None or n <= 2:
        return np.vectorize(old_to_idx.get)(labels), n

    # Build region adjacency graph
    idx_labels = np.vectorize(old_to_idx.get)(labels)
    region_adj = [set() for _ in range(n)]
    for face, neighbors in face_adjacency.items():
        r = int(idx_labels[face])
        for nb in neighbors:
            nb_r = int(idx_labels[nb])
            if nb_r != r:
                region_adj[r].add(nb_r)
                region_adj[nb_r].add(r)

    # Greedy max-spread: assign IDs so adjacent regions are far apart in value.
    # Process most-connected regions first so they have the most freedom.
    order = sorted(range(n), key=lambda r: -len(region_adj[r]))
    new_id = np.full(n, -1, dtype=np.int64)
    available = list(range(n))

    for region in order:
        neighbor_ids = [new_id[nb] for nb in region_adj[region] if new_id[nb] >= 0]
        if not neighbor_ids:
            pick = available[len(available) // 2]   # seed near the middle
        else:
            pick = max(available, key=lambda x: min(abs(x - nid) for nid in neighbor_ids))
        new_id[region] = pick
        available.remove(pick)

    final_map = {old: int(new_id[old_to_idx[old]]) for old in unique_labels}
    return np.vectorize(final_map.get)(labels), n

def compute_region_adjacency(face_adjacency, face_labels):     # Build a reverse map: face_id → region_id
    region_adjacency = defaultdict(set)
    # For each face, look at its neighbors
    for face, neighbors in face_adjacency.items():
        region = face_labels[face]
        if region is None:
            continue  # skip if face isn't part of any region

        for neighbor in neighbors:
            neighbor_region = face_labels[neighbor]
            if neighbor_region is None or neighbor_region == region:
                continue  # skip if same region or unknown
            region_adjacency[region].add(neighbor_region)
    return dict(region_adjacency)

def compute_region_to_faces(region_labels):
    region_to_faces = defaultdict(list)
    for face, region in enumerate(region_labels):
        if region != -1:  # Ignore unassigned faces
            region_to_faces[region].append(face)
    region_to_faces = dict(sorted(region_to_faces.items()))
    return region_to_faces  

def separate_disconnected_components(mesh, face_adjacency, raw_cluster_labels):
    unique_classes = np.unique(raw_cluster_labels)
    class_regions = {}
    visited = set()
    region_labels = -np.ones(mesh.n_cells, dtype=int) # Initialize region labels to -1
    region_id = 0

    for cls in unique_classes:
        class_faces = np.where(raw_cluster_labels == cls)[0]
        class_face_set = set(class_faces)

        regions = []
        for face in class_faces:
            if face in visited:
                continue
            # BFS to find all connected faces of same class
            queue = [face]
            region = []
            while queue: 
                current = queue.pop()
                if current in visited or current not in class_face_set:
                    continue
                visited.add(current)
                region.append(current)
                queue.extend(face_adjacency[current])
            regions.append(region)
            for f in region:
                region_labels[f] = region_id
            region_id += 1
        class_regions[cls] = regions

    region_labels_count = len(np.unique(region_labels))
    #print("After separate_disconnected_components, There are {} regions having data".format(region_labels_count))

    return region_labels

def compute_face_areas(mesh):
    """Returns per-face area array computed from triangle geometry."""
    return mesh.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']

def merge_zero_area_regions(mesh, region_labels, face_adjacency, area_tol=1e-12):
    """Merge regions with total area <= area_tol into an adjacent real region.

    Each pass merges ONE zero-area region into its largest-area neighbor (found
    via region adjacency, else by an edge scan), then recomputes so the maps
    never go stale. Only neighbors with area > area_tol are valid targets, so an
    isolated blob of mutually adjacent zero-area regions can never thrash by
    swapping labels; such regions are left as-is with a warning.
    """
    merged_labels = region_labels.copy()
    face_areas = compute_face_areas(mesh)

    while True:
        region_to_faces = compute_region_to_faces(merged_labels)
        region_adjacency = compute_region_adjacency(face_adjacency, merged_labels)
        region_areas = {
            rid: float(np.sum(face_areas[faces]))
            for rid, faces in region_to_faces.items()
        }
        zero_area = [rid for rid, area in region_areas.items() if area <= area_tol]
        if not zero_area:
            break

        merged_any = False
        for rid in zero_area:
            faces = region_to_faces[rid]
            neighbors = list(region_adjacency.get(rid, set()))
            if not neighbors:
                for f in faces:
                    for nb_face in face_adjacency.get(f, ()):
                        nb_rid = int(merged_labels[nb_face])
                        if nb_rid != rid:
                            neighbors.append(nb_rid)
                neighbors = list(set(neighbors))

            # Only merge into a real (non-zero-area) region: merging a zero-area
            # region into another zero-area one makes no progress and can loop
            # forever by swapping labels.
            real = [n for n in neighbors if region_areas.get(n, 0.0) > area_tol]
            if not real:
                continue

            target = max(real, key=lambda n: region_areas[n])
            for f in faces:
                merged_labels[f] = target
            print(f"Merged zero-area region {rid} -> {target} "
                  f"(area={region_areas[rid]:.2e}, {len(faces)} faces)")
            merged_any = True
            break   # recompute the maps before handling the next zero-area region

        if not merged_any:
            print(f"Warning: {len(zero_area)} zero-area region(s) have no "
                  f"non-zero-area neighbor; leaving them as-is")
            break

    return merged_labels

def _build_border_buckets(mesh, region_labels):
    """Precompute per region-pair border dihedral-angle sums and counts.

    Returns (border_angle_sum, border_count), both dicts keyed by the canonical
    region pair (min_rid, max_rid). For each interior edge whose two faces lie in
    different regions, accumulate its dihedral angle arccos(n1·n2) into the bucket.

    Face normals are fixed for the whole merge, so each edge's angle is computed
    exactly once here; merging only re-buckets these sums (never recomputes them).
    The smoothness of a pair is then 1 - log1p(sum/count)/log1p(pi) — identical to
    _smoothness_term applied to that pair's border edges, but O(1) per query.
    """
    faces = mesh.faces.reshape((-1, 4))[:, 1:]           # (F, 3) triangle vertex ids
    F = len(faces)

    # Same canonical-edge construction as build_face_adjacency: any two faces that
    # share an undirected (min, max) vertex edge are edge-adjacent (manifold mesh).
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges.sort(axis=1)
    face_of_edge = np.tile(np.arange(F), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e = edges[order]
    fo = face_of_edge[order]
    same = np.where((e[1:, 0] == e[:-1, 0]) & (e[1:, 1] == e[:-1, 1]))[0]
    a = fo[same]
    b = fo[same + 1]

    normals = np.asarray(mesh.cell_data['Normals'], dtype=np.float64)
    dots = np.einsum('ij,ij->i', normals[a], normals[b])
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    la = region_labels[a]
    lb = region_labels[b]
    # cross-region edges only; skip unassigned (-1) faces, matching region_to_faces.
    border = (la != lb) & (la != -1) & (lb != -1)

    border_angle_sum: dict = defaultdict(float)
    border_count: dict = defaultdict(int)
    for ra, rb, ang in zip(la[border].tolist(), lb[border].tolist(), angles[border].tolist()):
        key = (ra, rb) if ra < rb else (rb, ra)
        border_angle_sum[key] += ang
        border_count[key] += 1
    return border_angle_sum, border_count


def compute_region_noise(weighted_normal_sum, area):
    """Area-weighted spherical variance of a region's face normals, in [0, 1].

    weighted_normal_sum = Σ aᵢ·nᵢ  (area-weighted resultant of the unit face normals)
    area                = Σ aᵢ
    Returns 1 - ‖resultant‖ / area: 0 when all normals are identical (perfectly
    flat), approaching 1 as the normals disperse (noisy / curved region).
    """
    if area < 1e-12:
        return 0.0
    return float(1.0 - np.linalg.norm(weighted_normal_sum) / area)

def compute_noise_term(wsum1, wsum2, theta0_deg=15.0, sharpness=0.3):
    """Orientation-coherence gate N ∈ [0, 1] for merging two regions.

    Penalises merging regions whose *dominant orientations* differ. A region's
    dominant orientation is the direction of its area-weighted normal resultant
    (weighted_normal_sum). Let θ be the angle between the two resultants:

        N = 1 / (1 + exp(sharpness · (θ_deg − theta0_deg)))

      θ ≪ theta0_deg  → N ≈ 1   (aligned, reward the merge)
      θ  = theta0_deg → N = 0.5  (neutral)
      θ ≫ theta0_deg  → N ≈ 0   (different orientation → strongly penalise)

    Unlike the old "change-in-noise" formulation, this depends only on the angle
    between the regions' mean normals, so two flat facets meeting at a crease are
    penalised regardless of how internally flat each one is.

    theta0_deg : half-way threshold in degrees (default 15°).
    sharpness  : sigmoid steepness per degree (default 0.3 → N≈0 by ~30–40°).
                 With these defaults: 0°→0.99, 15°→0.5, 30°→0.011, 42°→0.0003.
    Returns 0.5 (neutral) if either region has no net orientation (degenerate).
    """
    n1 = float(np.linalg.norm(wsum1))
    n2 = float(np.linalg.norm(wsum2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.5
    cos_theta = float(np.dot(wsum1, wsum2)) / (n1 * n2)
    theta_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return float(1.0 / (1.0 + np.exp(sharpness * (theta_deg - theta0_deg))))


def _balance_gate(area1, area2, total_area, k=1.0, w_rel=0.5, w_abs=0.5):
    """Size gate R = w_rel·R_rel + w_abs·R_abs, a weighted blend of two terms in [0,1].

    R_rel  *relative* imbalance between the two regions:
        r = A_big / A_small ≥ 1;  R_rel = (r^k - 1)/(r^k + 1).
        Equal regions (r=1) → 0; strongly imbalanced → 1.

    R_abs *absolute* smallness of the smaller region vs the whole mesh:
        u = total_area / A_small ≥ 1;  R_abs = (u^k - 1)/(u^k + 1).
        Smaller region tiny vs mesh (ratio = A_small/total → 0) → R_abs → 1;
        smaller region ≈ whole mesh (ratio → 1) → R_abs → 0.

    With w_rel + w_abs = 1 the result stays in [0, 1]. k=0 disables both gates (R=1).
    Set w_rel=1, w_abs=0 to recover the old pure-imbalance behaviour.
    """
    if k == 0:
        return 1.0
    a_big = max(area1, area2)
    a_small = max(min(area1, area2), 1e-12) # to avoid division by zero

    u = total_area / a_small
    uk = u ** k
    r_abs = (uk - 1.0) / (uk + 1.0)

    if w_abs == 1:
        return r_abs
    else:
        r = a_big / a_small
        rk = r ** k
        r_rel = (rk - 1.0) / (rk + 1.0)
        return float(w_rel * r_rel + w_abs * r_abs)

def merge_region_based_on_power(mesh, region_labels, face_adjacency,
                                alpha=1/4, beta=3/4,
                                balance_k= 0.5,
                                power_thr=0.15, max_merges=1000,
                                target_n_regs=None, visualize=False, verbose=True):
    """Greedy merge using multiplicative power P = (α·S + β·N) · R.
    S - Boundary Smoothness  : 1 - log-mapped average dihedral angle of border edges → [0,1].
        Flat boundary (angle≈0) → S≈1; sharp fold → S≈0.  

    N - Orientation gate : sigmoid of the angle θ between the two regions' mean
        normals. Aligned (θ small) → N→1; different orientation (θ large) → N→0;
        neutral at θ = theta0_deg. See compute_noise_term for the threshold/steepness.

    R - Size gate: weighted blend R = w_rel·R_rel + w_abs·R_abs, each in [0,1].
        R_rel = (r^balance_k - 1)/(r^balance_k + 1),  r = A_big/A_small ≥ 1
            → 0 for equal-sized regions, → 1 for strongly imbalanced pairs.
        R_abs = (u^balance_k - 1)/(u^balance_k + 1),  u = total_area/A_small ≥ 1
            → 1 when the smaller region is tiny vs the whole mesh (cleanup of fragments), shrinking toward 0 as it grows to a large fraction.
        balance_k controls steepness: 0 disables the gate (R=1 always).

    The multiplicative structure favors absorbing a small region into a large
    neighbor: P stays low for similar-sized pairs even when S and N are high.

    Args:
        alpha, beta   : weights for S and N (should sum to 1).
        balance_k     : sharpness of the balance gate (default 1.0).
        noise_k       : sigmoid steepness for the noise term (default 5.0).
        balance_w_rel, balance_w_abs : weights for the relative-imbalance and
            absolute-smallness gate terms (default 0.5/0.5; should sum to 1).
            Use 1.0/0.0 to recover the old pure-imbalance gate.
        power_thr     : stop when best remaining power < this value. Smaller value means more merging.
        max_merges    : hard cap on merge iterations.
        target_n_regs: stop when region count reaches this value.
        visualize     : show a PyVista window for each merge step.
    """
    balance_w_rel=0
    balance_w_abs=1

    
    merged_labels = region_labels.copy()
    normals_data = mesh.cell_data['Normals']
    face_areas = compute_face_areas(mesh)

    region_to_faces = {rid: list(faces)  for rid, faces in compute_region_to_faces(merged_labels).items()}
    region_adjacency = {r: set(nbrs)     for r, nbrs in compute_region_adjacency(face_adjacency, merged_labels).items()}
    region_areas = {rid: float(np.sum(face_areas[faces]))   for rid, faces in region_to_faces.items()}
    total_area = float(np.sum(face_areas))   # constant; denominator of the absolute size gate
    avg_normals = {
        rid: normalize(np.mean(normals_data[faces], axis=0, keepdims=True))[0]
        for rid, faces in region_to_faces.items()
    }
    weighted_normal_sum = {
        rid: (face_areas[faces][:, None] * normals_data[faces]).sum(axis=0)
        for rid, faces in region_to_faces.items()
    }
    noise = {rid: compute_region_noise(weighted_normal_sum[rid], region_areas[rid])
             for rid in region_to_faces}

    # Incremental border bookkeeping: per region-pair sum/count of border-edge
    # dihedral angles, built once and updated in O(deg) per merge (see section 6).
    border_angle_sum, border_count = _build_border_buckets(mesh, merged_labels)
    _LOG1P_PI = float(np.log1p(np.pi))

    def smoothness_of(r1, r2):
        pair = (r1, r2) if r1 < r2 else (r2, r1)
        c = border_count.get(pair, 0)
        if not c:
            return 0.0
        avg_angle = border_angle_sum[pair] / c
        return float(1.0 - np.log1p(avg_angle) / _LOG1P_PI)

    def power_of(r1, r2):
        s = smoothness_of(r1, r2)
        n = compute_noise_term(weighted_normal_sum[r1], weighted_normal_sum[r2])
        r = _balance_gate(region_areas[r1], region_areas[r2], total_area,
                          balance_k, balance_w_rel, balance_w_abs)
        return (alpha * s + beta * n) * r, s, n, r

    heap: list = []
    pair_version: dict = {}     # pair -> latest push version; absent => invalidated. Only the latest version of a pair is "fresh".
    push_counter = [0]          # list-wrapped so the closure can mutate it

    def push_pair(r1, r2):
        pair = (min(r1, r2), max(r1, r2))
        push_counter[0] += 1
        v = push_counter[0]
        pair_version[pair] = v
        p, s, n_val, r_val = power_of(*pair)
        heapq.heappush(heap, (-p, v, pair, s, n_val, r_val))

    seen: set = set()
    for r1, neighbors in region_adjacency.items():
        for r2 in neighbors:
            pair = (min(r1, r2), max(r1, r2))
            if pair in seen:
                continue
            seen.add(pair)
            push_pair(r1, r2)

    for merge_num in range(max_merges):
        n_regions = len(region_to_faces)
        if target_n_regs is not None and n_regions <= target_n_regs:
            if verbose:
                print(f"Reached target of {target_n_regs} regions after {merge_num} merges.")
            break

        best_power, best_pair = None, None
        best_s = best_n = best_r = 0.0
        while heap: #Pop Best Pair
            neg_p, v, pair, s, n_val, r_val = heapq.heappop(heap)
            if pair_version.get(pair) == v:    # latest version => fresh; stale and invalidated entries are skipped
                best_power = -neg_p
                best_pair = pair
                best_s, best_n, best_r = s, n_val, r_val
                break

        if best_pair is None or best_power < power_thr:
            if verbose:
                label = f"{best_power:.4f}" if best_power is not None else "N/A"
                print(f"Stopping after {merge_num} merges: max power = {label}")
            break

        r1, r2 = best_pair
        pair_version.pop(best_pair, None)   # this pair is consumed

        p_fresh, s, n_val, r_val = best_power, best_s, best_n, best_r
        merged_id = r1 if region_areas[r1] < region_areas[r2] else r2
        target_id = r2 if merged_id == r1 else r1

        area_r1, area_r2 = region_areas[r1], region_areas[r2]
        ratio = max(area_r1, area_r2) / max(min(area_r1, area_r2), 1e-12)
        angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(avg_normals[r1], avg_normals[r2]), -1.0, 1.0))))
        if verbose:
            print(f"Merge {merge_num + 1}: {r1}+{r2}  "
                  f"areas=({area_r1:.2f},{area_r2:.2f})  ratio={ratio:.2f}  "
                  f"angle={angle_deg:.1f}°  "
                  f"P={p_fresh:.4f} (αS={alpha*s:.3f} N={n_val:.3f} R={r_val:.3f})  "
                  f"{merged_id}→{target_id}  {n_regions}→{n_regions-1}")

        if visualize:
            highlight = np.zeros(len(merged_labels), dtype=np.int32)
            for f in region_to_faces[merged_id]:
                highlight[f] = 1
            for f in region_to_faces[target_id]:
                highlight[f] = 2
            mesh_vis = mesh.copy()
            mesh_vis.cell_data['highlight'] = highlight
            pl = pv.Plotter(title=f"Merge step {merge_num + 1}")
            pl.add_mesh(mesh_vis, scalars='highlight',
                        cmap=['#bbbbbb', '#e63232', '#3264e6'],
                        clim=[0, 2], show_scalar_bar=False,
                        show_edges=True, edge_opacity=0.15)
            pl.add_text(
                f"Step {merge_num+1}  |  {merged_id}→{target_id}  |  "
                f"P={p_fresh:.4f}  αS={alpha*s:.3f}  βN={beta*n_val:.3f}  R={r_val:.3f}  |  "
                f"{n_regions}→{n_regions-1} regions",
                position='upper_left', font_size=9, color='white')
            pl.show()

        for f in region_to_faces[merged_id]:
            merged_labels[f] = target_id

        # 1. Merge face lists; delete merged_id from dict
        combined_faces = region_to_faces[target_id] + region_to_faces.pop(merged_id)
        region_to_faces[target_id] = combined_faces
        # 2. Add areas; delete merged_id's area entry
        region_areas[target_id] += region_areas.pop(merged_id)
        # 3. Recompute average normal from the new full face list
        avg_normals[target_id] = normalize(
            np.mean(normals_data[combined_faces], axis=0, keepdims=True))[0]
        # 4. Update weighted normal sum and noise for the merged region
        weighted_normal_sum[target_id] = (
            weighted_normal_sum[target_id] + weighted_normal_sum.pop(merged_id))
        noise[target_id] = compute_region_noise(weighted_normal_sum[target_id], region_areas[target_id])

        #5 After a merge, any heap entry involving merged_id or target_id is now outdated and must be invalidated.
        for r in region_adjacency.get(merged_id, set()):
            pair_version.pop((min(merged_id, r), max(merged_id, r)), None)
        for r in region_adjacency.get(target_id, set()):
            pair_version.pop((min(target_id, r), max(target_id, r)), None)
        #6 Update adjacency: rewires the region adjacency graph after merged_id is absorbed into target_id.
        merged_nbrs = region_adjacency.pop(merged_id, set())  # remove merged_id from adjacency keys
        target_nbrs = region_adjacency.get(target_id, set())  # neighbors of the region that survives
        new_nbrs = (merged_nbrs | target_nbrs) - {merged_id, target_id}  # neighbors of the merged region

        # 6b. Re-bucket border edges: the merged↔target seam becomes interior (drop it),
        # and each (merged_id, r) border folds into (target_id, r). O(deg) — no rescanning.
        seam = (min(merged_id, target_id), max(merged_id, target_id))
        border_angle_sum.pop(seam, None)
        border_count.pop(seam, None)
        for r in merged_nbrs:
            if r == target_id:
                continue
            src = (min(merged_id, r), max(merged_id, r))
            dst = (min(target_id, r), max(target_id, r))
            c = border_count.pop(src, 0)
            if c:
                border_angle_sum[dst] = border_angle_sum.get(dst, 0.0) + border_angle_sum.pop(src, 0.0)
                border_count[dst] = border_count.get(dst, 0) + c

        region_adjacency[target_id] = new_nbrs    # update adjacency of the target region
        for r in new_nbrs:  # update adjacency of the neighbors
            region_adjacency[r].discard(merged_id)
            region_adjacency[r].add(target_id)

        #7 update heap
        for r in new_nbrs:
            push_pair(target_id, r)

    if verbose:
        print(f"After merge_region_based_on_power: {len(region_to_faces)} regions remain.")
    return merged_labels

