# Unity import check

Turns the pipeline's `<asset>_LOD0/1/2.glb` files into Unity `LODGroup`
prefabs and checks that none of them needs a manual fix (missing materials,
wrong scale, lost triangles).

## Steps (about 30 minutes, most of it installing Unity)

1. Install **Unity Hub** and **Unity 6 LTS** (free Personal licence).
2. Create a new project with the **3D (URP)** template.
3. **Window → Package Manager → + → Add package by name…** →
   `com.unity.cloud.gltfast` → Add.
4. Create the folder `Assets/PipelineLODs/` and copy every `*_LOD*.glb`
   from the pipeline's output folder into it. Wait for the import to finish.
5. Copy `unity/Editor/LodSetImporter.cs` to `Assets/Editor/LodSetImporter.cs`.
6. Menu **Tools → 3D Pipeline → Build LOD Prefabs**. The Console prints how
   many prefabs it built and how many problems it found.
7. Cross-check against the pipeline:
   ```bash
   python unity/compare_with_pipeline.py \
     <UnityProject>/Assets/PipelineLODs/Prefabs/unity_import_report.csv \
     results/pipeline_report.json
   ```
8. Drag two or three prefabs (a car first) into the scene, move the camera
   away, and watch LOD0 → LOD1 → LOD2 switch. Take a screenshot with the
   LODGroup inspector visible and save it as `docs/unity_lodgroup.png`.

## What to record

Copy the summary lines printed in step 7 (imported, triangles match, no
missing material, same size) into the main README's results section.
