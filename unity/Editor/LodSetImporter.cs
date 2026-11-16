// LodSetImporter.cs
// Unity Editor tool: turns the pipeline's <asset>_LOD0/1/2.glb files into
// ready-to-use LODGroup prefabs and checks that nothing needs a manual fix.
//
// Requires the glTFast package (com.unity.cloud.gltfast), which imports .glb
// files placed under Assets/ as GameObject assets.
//
// Menu: Tools > 3D Pipeline > Build LOD Prefabs
// Input : every *_LOD0.glb under Assets/PipelineLODs/ (with its LOD1/LOD2 siblings)
// Output: Assets/PipelineLODs/Prefabs/<asset>.prefab with a LODGroup, and
//         Assets/PipelineLODs/Prefabs/unity_import_report.csv

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class LodSetImporter
{
    const string InputDir = "Assets/PipelineLODs";
    const string OutputDir = "Assets/PipelineLODs/Prefabs";

    // Screen-height fraction at which each LOD stops being used.
    static readonly float[] Transitions = { 0.50f, 0.20f, 0.03f };

    [MenuItem("Tools/3D Pipeline/Build LOD Prefabs")]
    public static void BuildAll()
    {
        if (!AssetDatabase.IsValidFolder(OutputDir))
            AssetDatabase.CreateFolder(InputDir, "Prefabs");

        var report = new StringBuilder(
            "asset,lod,imported,meshes,triangles,missing_materials,bounds_size,bounds_diff_vs_lod0\n");
        int sets = 0, problems = 0;

        string[] lod0Paths = Directory.GetFiles(InputDir, "*_LOD0.glb", SearchOption.AllDirectories);
        foreach (string lod0Path in lod0Paths.OrderBy(p => p))
        {
            string asset = Path.GetFileName(lod0Path).Replace("_LOD0.glb", "");
            string dir = Path.GetDirectoryName(lod0Path).Replace('\\', '/');

            var root = new GameObject(asset);
            var lods = new List<LOD>();
            Bounds? lod0Bounds = null;

            for (int i = 0; i < 3; i++)
            {
                string path = $"{dir}/{asset}_LOD{i}.glb";
                var source = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (source == null)
                {
                    report.AppendLine($"{asset},LOD{i},no,0,0,0,0,");
                    problems++;
                    continue;
                }

                var child = (GameObject)PrefabUtility.InstantiatePrefab(source);
                child.name = $"{asset}_LOD{i}";
                child.transform.SetParent(root.transform, false);

                Renderer[] renderers = child.GetComponentsInChildren<Renderer>();
                int meshes = 0, tris = 0, missing = 0;
                foreach (var mf in child.GetComponentsInChildren<MeshFilter>())
                {
                    if (mf.sharedMesh == null) continue;
                    meshes++;
                    tris += CountTriangles(mf.sharedMesh);
                }
                foreach (var r in renderers)
                    missing += r.sharedMaterials.Count(m => m == null);

                Bounds b = Combined(renderers);
                if (i == 0) lod0Bounds = b;
                float diff = lod0Bounds.HasValue
                    ? (b.size - lod0Bounds.Value.size).magnitude / Mathf.Max(lod0Bounds.Value.size.magnitude, 1e-6f)
                    : 0f;
                if (missing > 0 || diff > 0.01f) problems++;

                report.AppendLine(string.Join(",", asset, $"LOD{i}", "yes", meshes, tris, missing,
                    b.size.magnitude.ToString("F4"), diff.ToString("F4")));
                lods.Add(new LOD(Transitions[i], renderers));
            }

            var group = root.AddComponent<LODGroup>();
            group.SetLODs(lods.ToArray());
            group.RecalculateBounds();

            PrefabUtility.SaveAsPrefabAsset(root, $"{OutputDir}/{asset}.prefab");
            Object.DestroyImmediate(root);
            sets++;
        }

        File.WriteAllText($"{OutputDir}/unity_import_report.csv", report.ToString());
        AssetDatabase.Refresh();
        Debug.Log($"[LodSetImporter] Built {sets} LOD prefabs; {problems} problem(s). " +
                  $"Report: {OutputDir}/unity_import_report.csv");
    }

    static int CountTriangles(Mesh mesh)
    {
        int n = 0;
        for (int s = 0; s < mesh.subMeshCount; s++)
            if (mesh.GetTopology(s) == MeshTopology.Triangles)
                n += (int)mesh.GetIndexCount(s) / 3;
        return n;
    }

    static Bounds Combined(Renderer[] renderers)
    {
        if (renderers.Length == 0) return new Bounds(Vector3.zero, Vector3.zero);
        Bounds b = renderers[0].bounds;
        foreach (var r in renderers.Skip(1)) b.Encapsulate(r.bounds);
        return b;
    }
}
