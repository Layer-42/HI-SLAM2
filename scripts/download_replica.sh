#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "Downloading Replica"

wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip -q Replica.zip
rm -f Replica.zip

cd Replica

echo "Downloading culled meshes"

wget -c https://cvg-data.inf.ethz.ch/nice-slam/cull_replica_mesh.zip
unzip -q cull_replica_mesh.zip
rm -f cull_replica_mesh.zip

rm -rf gt_mesh_culled
mv cull_replica_mesh gt_mesh_culled

mkdir -p gt_mesh
mv ./*.ply gt_mesh/ 2>/dev/null || true

echo "Replica ready at ${DATA_DIR}/Replica"