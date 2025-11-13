#!/usr/bin/env python3
import os, sys, argparse, json
import numpy as np
import torch

# ------------------------
#   k-space utilities
# ------------------------
def build_kbuffers(G, BoxSize, device):
    kx = 2*np.pi*np.fft.fftfreq(G, d=BoxSize/G)
    ky = 2*np.pi*np.fft.fftfreq(G, d=BoxSize/G)
    kz = 2*np.pi*np.fft.rfftfreq(G, d=BoxSize/G)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    KN2 = KX**2 + KY**2 + KZ**2
    KN2[KN2 == 0.0] = 1.0
    KX = torch.from_numpy(KX).to(torch.complex64).to(device)
    KY = torch.from_numpy(KY).to(torch.complex64).to(device)
    KZ = torch.from_numpy(KZ).to(torch.complex64).to(device)
    KN2 = torch.from_numpy(KN2).to(torch.complex64).to(device)
    return KX, KY, KZ, KN2

@torch.no_grad()
def vf_from_delta(delta_1dhw, KX, KY, KZ, KN2, G):
    """
    delta_1dhw: torch.float32, shape (1, D, H, W) on device
    returns: np.ndarray (6, G, G, G)
    """
    # Keep REAL for rfftn:
    x = delta_1dhw.unsqueeze(0)                   # (1,1,D,H,W), float32

    # Real-to-complex FFT
    Xk = torch.fft.rfftn(x, dim=(-3, -2, -1))    # (1,1,G,G,G//2+1), complex

    # Physics in k-space
    fac = (-1j) / KN2
    ux_k = fac * KX * Xk
    uy_k = fac * KY * Xk
    uz_k = fac * KZ * Xk

    duydx_k = 1j * KX * uy_k
    duzdx_k = 1j * KX * uz_k
    duzdy_k = 1j * KY * uz_k

    def inv(z):  # -> (1,1,G,G,G) real
        return torch.fft.irfftn(z, s=(G, G, G), dim=(-3, -2, -1)).real

    ux  = inv(ux_k);  uy  = inv(uy_k);  uz  = inv(uz_k)
    dxy = inv(duydx_k); dxz = inv(duzdx_k); dyz = inv(duzdy_k)

    out = torch.cat([ux, uy, uz, dxy, dxz, dyz], dim=1)[0]  # (6,G,G,G)
    return out.to(torch.float32).cpu().numpy()


def parse_id_list(start, end, exclude):
    ids = list(range(start, end+1))
    excl_set = set(exclude or [])
    return [i for i in ids if i not in excl_set]

# ------------------------
#         main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0_dir", required=True,
                    help="Directory containing halo_lh_{ID}.npy")
    ap.add_argument("--out_dir_files", default="",
                    help="If set, write per-file {ID}_vf.npy into this dir")
    ap.add_argument("--out_memmap", default="",
                    help="If set, write a single memmap stack .npy here")
    ap.add_argument("--start", type=int, required=True, help="first ID (inclusive)")
    ap.add_argument("--end",   type=int, required=True, help="last ID (inclusive)")
    ap.add_argument("--exclude_ids", type=int, nargs="*", default=[],
                    help="IDs to skip (e.g., --exclude_ids 9)")
    ap.add_argument("--grid", type=int, default=128, help="grid size (e.g., 128)")
    ap.add_argument("--boxsize", type=float, default=1000.0, help="box size (Mpc/h)")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ids = parse_id_list(args.start, args.end, args.exclude_ids)
    if not ids:
        print("No IDs to process; check your --start/--end/--exclude_ids.")
        sys.exit(0)

    dev = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    G = args.grid
    KX, KY, KZ, KN2 = build_kbuffers(G, args.boxsize, dev)

    if args.out_dir_files:
        os.makedirs(args.out_dir_files, exist_ok=True)

    # If building a single memmap stack
    if args.out_memmap:
        N = len(ids)
        vf_shape = (N, 6, G, G, G)
        print(f"[memmap] creating stack {args.out_memmap} with shape {vf_shape}")
        vf_stack = np.memmap(args.out_memmap, dtype="float32", mode="w+", shape=vf_shape)
    else:
        vf_stack = None

    # Keep a simple index map for the memmap order
    index_map = []
    for mm_idx, sim_id in enumerate(ids):
        z0_path = os.path.join(args.z0_dir, f"halo_lh_{sim_id:04d}.npy")
        if not os.path.exists(z0_path):
            print(f"[warn] missing {z0_path}, skipping")
            continue

        arr = np.load(z0_path)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        assert arr.shape == (G, G, G), f"{z0_path}: expected {(G,G,G)}, got {arr.shape}"

        x = torch.from_numpy(arr.astype(np.float32)).to(dev).unsqueeze(0)  # (1,G,G,G)
        vf = vf_from_delta(x, KX, KY, KZ, KN2, G)  # (6,G,G,G)

        # per-file save
        if args.out_dir_files:
            out_f = os.path.join(args.out_dir_files, f"{sim_id}_vf.npy")
            if args.overwrite or (not os.path.exists(out_f)):
                np.save(out_f, vf.astype(np.float32))
                print(f"[file] wrote {out_f}")
            else:
                print(f"[file] skip exists {out_f}")

        # memmap write
        if vf_stack is not None:
            vf_stack[mm_idx, ...] = vf
            index_map.append(int(sim_id))
            print(f"[memmap] wrote id {sim_id} at index {mm_idx}")

    # Flush memmap and save a small index json so you know which ID is at which row
    if vf_stack is not None:
        vf_stack.flush()
        meta = {
            "shape": list(vf_stack.shape),
            "dtype": "float32",
            "grid": G,
            "boxsize": args.boxsize,
            "ids": index_map
        }
        meta_path = os.path.splitext(args.out_memmap)[0] + "_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[memmap] done. meta saved to {meta_path}")

if __name__ == "__main__":
    main()
