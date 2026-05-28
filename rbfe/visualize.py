"""Interactive HTML network visualization with aligned 2D molecular depictions."""

import json
import sys
from pathlib import Path

from .network import build_perturbation_network, discover_ligands
from .parameterize import _infer_net_charge


def _sim_to_hex(sim: float) -> str:
    """Tanimoto → hex color: red (0.45) → yellow (0.65) → green (0.90+)."""
    t = max(0.0, min(1.0, (sim - 0.45) / 0.50))
    if t < 0.5:
        f = t * 2.0
        r, g, b = 210, int(40 + f * 175), 40
    else:
        f = (t - 0.5) * 2.0
        r, g, b = int(210 * (1 - f) + 40 * f), 215, 40
    return f"#{r:02x}{g:02x}{b:02x}"


def _set_2d_coords(mol, ref_mol=None) -> None:
    """Compute 2D coordinates for mol, optionally aligning to ref_mol via MCS."""
    from rdkit.Chem import rdDepictor, rdFMCS
    from rdkit.Geometry.rdGeometry import Point2D

    if ref_mol is None or not ref_mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
        return

    mcs = rdFMCS.FindMCS(
        [ref_mol, mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        timeout=3,
    )
    if mcs.numAtoms < 4:
        rdDepictor.Compute2DCoords(mol)
        return

    from rdkit import Chem
    mcs_mol   = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    mol_match = mol.GetSubstructMatch(mcs_mol)
    if not ref_match or not mol_match:
        rdDepictor.Compute2DCoords(mol)
        return

    conf = ref_mol.GetConformer()
    coord_map = {
        mol_i: Point2D(conf.GetAtomPosition(ref_i).x,
                       conf.GetAtomPosition(ref_i).y)
        for ref_i, mol_i in zip(ref_match, mol_match)
    }
    rdDepictor.Compute2DCoords(mol, coordMap=coord_map)


def _mol_to_svg(mol, width: int = 160, height: int = 120) -> str:
    """Render mol to SVG. Caller must set 2D coordinates first."""
    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = False
    try:
        opts.clearBackground = False
    except AttributeError:
        pass
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    idx = svg.find("<svg")
    return svg[idx:] if idx != -1 else svg


def _html_tree_layout(
    root:    str,
    adj:     dict[str, list[tuple[str, float]]],
    n_heavy: dict[str, int],
) -> dict[str, tuple[float, int]]:
    """Return {node: (float_x, int_depth)} grid positions for a top-down tree."""
    def _width(node: str, parent: str | None) -> int:
        kids = [nb for nb, _ in adj[node] if nb != parent]
        return max(1, sum(_width(k, node) for k in kids)) if kids else 1

    positions: dict[str, tuple[float, int]] = {}

    def _place(node: str, parent: str | None, x_left: float, depth: int) -> float:
        kids = sorted(
            [nb for nb, _ in adj[node] if nb != parent],
            key=lambda k: n_heavy[k],
        )
        if not kids:
            positions[node] = (x_left + 0.5, depth)
            return x_left + 1.0
        x = x_left
        for kid in kids:
            x = _place(kid, node, x, depth + 1)
        positions[node] = (
            (positions[kids[0]][0] + positions[kids[-1]][0]) / 2.0,
            depth,
        )
        return x

    _place(root, None, 0.0, 0)
    return positions


def visualize_network(
    network_path:   Path = Path("network.json"),
    output_path:    Path = Path("network.html"),
    structures_dir: Path | None = None,
    cfg:            dict | None = None,
) -> None:
    """
    Generate a self-contained interactive HTML tree visualization of the MST.

    Reads network.json produced by 'network'.  If network.json is absent but
    structures_dir + cfg are supplied the network is built on-the-fly.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        sys.exit("rdkit required:  conda install -c conda-forge rdkit")

    if network_path.exists():
        print(f"\n  Reading {network_path}...")
        network = json.loads(network_path.read_text())
    elif structures_dir is not None and cfg is not None:
        print("\n  network.json not found — building network from structures/...")
        protein_stem = Path(cfg.get("protein", {}).get("pdb", "protein.pdb")).stem
        pdbs         = discover_ligands(structures_dir.resolve(), protein_stem)
        network      = build_perturbation_network(pdbs)
    else:
        sys.exit(
            f"{network_path} not found.\n"
            "Run 'network' first, or pass --structures-dir to build on-the-fly."
        )

    n_heavy: dict[str, int] = {lig["name"]: lig["n_heavy"] for lig in network["ligands"]}

    MOL_W, MOL_H = 160, 120
    print("  Loading molecules...")
    mols: dict[str, object] = {}
    for lig in network["ligands"]:
        pdb   = Path(lig["pdb"])
        mol_h = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
        if mol_h is None:
            sys.exit(f"RDKit could not parse {pdb}")
        charge = _infer_net_charge(pdb)
        rdDetermineBonds.DetermineBonds(mol_h, charge=charge)
        Chem.SanitizeMol(mol_h)
        mols[lig["name"]] = Chem.RemoveHs(mol_h)

    root_tmp = min(network["ligands"], key=lambda x: x["n_heavy"])["name"]
    print("  Computing aligned 2D layouts...")
    _set_2d_coords(mols[root_tmp])
    for name, mol in mols.items():
        if name != root_tmp:
            _set_2d_coords(mol, ref_mol=mols[root_tmp])

    print("  Rendering molecule depictions...")
    mol_svgs: dict[str, str] = {
        name: _mol_to_svg(mol, MOL_W, MOL_H) for name, mol in mols.items()
    }

    adj: dict[str, list[tuple[str, float]]] = {
        lig["name"]: [] for lig in network["ligands"]
    }
    for edge in network["edges"]:
        adj[edge["old"]].append((edge["new"], edge["similarity"]))
        adj[edge["new"]].append((edge["old"], edge["similarity"]))

    root = min(network["ligands"], key=lambda x: x["n_heavy"])["name"]

    NODE_W, NODE_H = 174, 162
    COL_W,  ROW_H  = 204, 240
    MARGIN         = 80

    grid = _html_tree_layout(root, adj, n_heavy)

    def to_px(gx: float, gy: int) -> tuple[float, float]:
        return (MARGIN + gx * COL_W - NODE_W / 2, MARGIN + gy * ROW_H)

    px_pos: dict[str, tuple[float, float]] = {
        name: to_px(gx, gy) for name, (gx, gy) in grid.items()
    }
    canvas_w = int(max(x + NODE_W for x, _ in px_pos.values()) + MARGIN)
    canvas_h = int(max(y + NODE_H for _, y in px_pos.values()) + MARGIN)

    parent: dict[str, str | None] = {root: None}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nb, _ in adj[node]:
            if nb not in parent:
                parent[nb] = node
                queue.append(nb)

    arrow_defs = ""
    edge_elems = ""
    for k, edge in enumerate(network["edges"]):
        old_n, new_n = edge["old"], edge["new"]
        sim   = edge["similarity"]
        color = _sim_to_hex(sim)
        sw    = round(1.5 + sim * 2.5, 1)

        src, dst = (old_n, new_n) if parent.get(new_n) == old_n else (new_n, old_n)

        sx, sy = px_pos[src];  sx += NODE_W / 2;  sy += NODE_H
        dx, dy = px_pos[dst];  dx += NODE_W / 2

        gap = dy - sy
        c1x, c1y = sx, sy + gap * 0.45
        c2x, c2y = dx, dy - gap * 0.45
        mx,  my  = (sx + dx) / 2, (sy + dy) / 2

        arrow_defs += (
            f'\n    <marker id="a{k}" markerWidth="9" markerHeight="7" '
            f'refX="8" refY="3.5" orient="auto">'
            f'<polygon points="0 0, 9 3.5, 0 7" fill="{color}"/></marker>'
        )
        edge_elems += f"""
  <g class="eg" data-tip="sim = {sim:.3f}  |  {src} → {dst}">
    <path d="M{sx:.1f},{sy:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {dx:.1f},{dy:.1f}"
          stroke="transparent" stroke-width="14" fill="none"/>
    <path d="M{sx:.1f},{sy:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {dx:.1f},{dy:.1f}"
          stroke="{color}" stroke-width="{sw}" fill="none" opacity="0.80"
          marker-end="url(#a{k})"/>
    <text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" dy="-5"
          style="font-size:11px;font-weight:700;fill:{color};pointer-events:none;">{sim:.3f}</text>
  </g>"""

    node_divs = ""
    for lig in network["ligands"]:
        name = lig["name"]
        x, y = px_pos[name]
        node_divs += (
            f'\n  <div class="node" style="left:{x:.1f}px;top:{y:.1f}px;'
            f'width:{NODE_W}px;">'
            f'\n    {mol_svgs[name]}'
            f'\n    <div class="nl">{name}</div>'
            f'\n    <div class="ns">{lig["n_heavy"]} heavy atoms'
            f' · {Path(lig["pdb"]).name}</div>'
            f'\n  </div>'
        )

    n_lig   = len(network["ligands"])
    n_edges = len(network["edges"])
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RBFE Network</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#1a1a2e;font-family:'Segoe UI',Helvetica,sans-serif;overflow:hidden;color:#e0e0e0}}
#bar{{position:fixed;top:0;left:0;right:0;height:46px;background:#16213e;display:flex;
  align-items:center;padding:0 20px;gap:14px;z-index:100;
  box-shadow:0 2px 10px rgba(0,0,0,.6);font-size:13px}}
#bar b{{color:#a8d8ff}}
#bar .hint{{margin-left:auto;color:#666;font-size:11px}}
#leg{{position:fixed;bottom:16px;right:16px;background:rgba(22,33,62,.92);
  border:1px solid #333;border-radius:6px;padding:10px 14px;z-index:100;font-size:11px;color:#aaa}}
#leg h4{{font-size:12px;color:#ccc;margin-bottom:6px}}
.lr{{display:flex;align-items:center;gap:8px;margin-top:4px}}
#vp{{position:fixed;top:46px;left:0;right:0;bottom:0;overflow:hidden;cursor:grab}}
#vp.drag{{cursor:grabbing}}
#cv{{position:absolute;transform-origin:0 0}}
#esvg{{position:absolute;top:0;left:0;overflow:visible;pointer-events:none}}
.eg{{pointer-events:all;cursor:default}}
.eg:hover path:last-of-type{{opacity:1!important;stroke-width:5!important}}
.node{{position:absolute;background:#fff;border-radius:8px;border:2px solid #ccc;
  cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.4);
  transition:border-color .15s,box-shadow .15s,transform .1s;overflow:hidden;user-select:none}}
.node:hover{{border-color:#4a9eff;z-index:20;
  box-shadow:0 6px 20px rgba(74,158,255,.5);transform:translateY(-2px)}}
.node svg{{display:block;background:#fff}}
.nl{{text-align:center;font-size:12px;font-weight:700;color:#222;
  padding:3px 4px 1px;background:#f0f4ff;border-top:1px solid #dde}}
.ns{{text-align:center;font-size:10px;color:#888;padding:1px 4px 4px;background:#f0f4ff}}
#tt{{position:fixed;background:rgba(10,10,30,.92);color:#fff;padding:5px 10px;
  border-radius:4px;font-size:12px;pointer-events:none;opacity:0;
  transition:opacity .1s;z-index:200;white-space:nowrap;border:1px solid #444}}
</style>
</head>
<body>
<div id="bar">
  <b>RBFE Perturbation Network</b>
  <span style="color:#444">|</span>
  <span><b>{n_lig}</b> ligands · <b>{n_edges}</b> edges (MST)</span>
  <span style="color:#444">|</span>
  <span>Root: <b>{root}</b> ({n_heavy[root]} heavy atoms)</span>
  <span class="hint">Scroll = zoom &nbsp;·&nbsp; Drag = pan &nbsp;·&nbsp; Hover edge = similarity</span>
</div>
<div id="leg">
  <h4>Tanimoto similarity</h4>
  <div class="lr">
    <div style="width:70px;height:6px;background:linear-gradient(to right,#d22828,#d2a828,#28d228);border-radius:3px"></div>
    <span>low → high</span>
  </div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.50)};border-radius:2px"></div><span>0.50</span></div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.65)};border-radius:2px"></div><span>0.65</span></div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.80)};border-radius:2px"></div><span>0.80</span></div>
  <div class="lr"><div style="width:14px;height:4px;background:{_sim_to_hex(0.92)};border-radius:2px"></div><span>0.90+</span></div>
</div>
<div id="vp">
  <div id="cv" style="width:{canvas_w}px;height:{canvas_h}px">
    <svg id="esvg" width="{canvas_w}" height="{canvas_h}">
      <defs>{arrow_defs}
      </defs>
      {edge_elems}
    </svg>
    {node_divs}
  </div>
</div>
<div id="tt"></div>
<script>
const vp=document.getElementById('vp'),cv=document.getElementById('cv');
let sc=1,tx=0,ty=0,drag=false,lx=0,ly=0;
function upd(){{cv.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{sc}})`;}}
vp.addEventListener('wheel',e=>{{
  e.preventDefault();
  const d=e.deltaY>0?.9:1.1,r=vp.getBoundingClientRect(),
        mx=e.clientX-r.left,my=e.clientY-r.top;
  tx=mx-(mx-tx)*d; ty=my-(my-ty)*d;
  sc=Math.max(.15,Math.min(4,sc*d)); upd();
}},{{passive:false}});
vp.addEventListener('mousedown',e=>{{if(e.target.closest('.node'))return;drag=true;lx=e.clientX;ly=e.clientY;vp.classList.add('drag');}});
document.addEventListener('mousemove',e=>{{if(!drag)return;tx+=e.clientX-lx;ty+=e.clientY-ly;lx=e.clientX;ly=e.clientY;upd();}});
document.addEventListener('mouseup',()=>{{drag=false;vp.classList.remove('drag');}});
const tt=document.getElementById('tt');
document.querySelectorAll('.eg').forEach(el=>{{
  el.addEventListener('mouseenter',()=>{{tt.textContent=el.dataset.tip;tt.style.opacity=1;}});
  el.addEventListener('mousemove',e=>{{tt.style.left=(e.clientX+14)+'px';tt.style.top=(e.clientY-32)+'px';}});
  el.addEventListener('mouseleave',()=>{{tt.style.opacity=0;}});
}});
const vw=vp.clientWidth,vh=vp.clientHeight,cw={canvas_w},ch={canvas_h};
sc=Math.min(.92,vw/cw,vh/ch); tx=(vw-cw*sc)/2; ty=(vh-ch*sc)/2; upd();
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"\n  Visualization written → {output_path}")
    print(f"  Open with: open {output_path}")
