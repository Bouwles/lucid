using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Lucid.Recorder.EditorTools
{
    /// <summary>
    /// Window > LUCID > Determinism Test. Replays a recorded episode (same seed, same actions)
    /// through the LucidRecorder in the open scene, then reports the first step where the state
    /// differs and, optionally, the first frame whose pixels differ by more than a tolerance.
    /// Can also compare two existing episode folders, for example a replay recorded in a build.
    /// </summary>
    public sealed class DeterminismTest : EditorWindow
    {
        const string PendingKey = "Lucid.Determinism.Pending";
        const string CompareKey = "Lucid.Determinism.CompareOnEditMode";
        const string OriginalKey = "Lucid.Determinism.Original";
        const string ReplayKey = "Lucid.Determinism.Replay";
        const string ToleranceKey = "Lucid.Determinism.Tolerance";
        const string FramesKey = "Lucid.Determinism.CompareFrames";
        const string ReportKey = "Lucid.Determinism.Report";

        string _original = "";
        string _replay = "";
        Vector2 _scroll;

        [MenuItem("Window/LUCID/Determinism Test")]
        static void Open() => GetWindow<DeterminismTest>("LUCID Determinism");

        [InitializeOnLoadMethod]
        static void RegisterHooks()
        {
            LucidRecorder.EditorStarting -= OnRecorderStarting;
            LucidRecorder.EditorStarting += OnRecorderStarting;
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
        }

        void OnEnable()
        {
            _original = SessionState.GetString(OriginalKey, "");
            _replay = SessionState.GetString(ReplayKey, "");
        }

        void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            EditorGUILayout.HelpBox(
                "Replays an episode with its seed and action list through the LucidRecorder in the open scene, " +
                "then compares the replay with the original. The replay is written to a temporary folder.",
                MessageType.Info);

            FolderField("Original episode", ref _original);
            SessionState.SetString(OriginalKey, _original);
            int tolerance = EditorGUILayout.IntSlider(
                new GUIContent("Pixel tolerance", "Largest allowed difference per colour channel. 0 means pixel-identical."),
                SessionState.GetInt(ToleranceKey, 0), 0, 64);
            SessionState.SetInt(ToleranceKey, tolerance);
            bool frames = EditorGUILayout.Toggle("Compare frames", SessionState.GetBool(FramesKey, true));
            SessionState.SetBool(FramesKey, frames);

            using (new EditorGUI.DisabledScope(EditorApplication.isPlaying || !Directory.Exists(_original)))
            {
                if (GUILayout.Button("Replay in play mode and compare", GUILayout.Height(28)))
                {
                    SessionState.SetString(ReportKey, "Running the replay...");
                    SessionState.SetBool(PendingKey, true);
                    EditorApplication.isPlaying = true;
                }
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Compare two existing episodes", EditorStyles.boldLabel);
            FolderField("Replayed episode", ref _replay);
            SessionState.SetString(ReplayKey, _replay);
            using (new EditorGUI.DisabledScope(!Directory.Exists(_original) || !Directory.Exists(_replay)))
            {
                if (GUILayout.Button("Compare")) SessionState.SetString(ReportKey, Compare(_original, _replay, tolerance, frames));
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Result", EditorStyles.boldLabel);
            EditorGUILayout.TextArea(SessionState.GetString(ReportKey, "No test run yet."), GUILayout.ExpandHeight(true));
            EditorGUILayout.EndScrollView();
        }

        static void FolderField(string label, ref string value)
        {
            EditorGUILayout.BeginHorizontal();
            value = EditorGUILayout.TextField(label, value);
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                string picked = EditorUtility.OpenFolderPanel(label, value, "");
                if (!string.IsNullOrEmpty(picked)) value = picked;
                GUIUtility.ExitGUI();
            }
            EditorGUILayout.EndHorizontal();
        }

        static void OnRecorderStarting(LucidRecorder recorder)
        {
            if (!SessionState.GetBool(PendingKey, false)) return;
            SessionState.SetBool(PendingKey, false);

            var replay = recorder.gameObject.AddComponent<ReplayActionSource>();
            try
            {
                replay.Load(SessionState.GetString(OriginalKey, ""));
            }
            catch (Exception e)
            {
                SessionState.SetString(ReportKey, "Could not load the original episode: " + e.Message);
                EditorApplication.isPlaying = false;
                return;
            }

            recorder.ActionSourceComponent = replay;
            recorder.OutputFolderOverride = Path.Combine(Path.GetTempPath(), "lucid_determinism");
            recorder.MaxEpisodesOverride = 1;
            recorder.SessionFinished += OnReplayFinished;
            if (!recorder.StartRecording())
            {
                recorder.SessionFinished -= OnReplayFinished;
                SessionState.SetString(ReportKey, "The recorder did not start. See the Console.");
                EditorApplication.isPlaying = false;
            }
        }

        static void OnReplayFinished(string sessionFolder)
        {
            SessionState.SetString(ReplayKey, Path.Combine(sessionFolder, "episodes", "ep_000000"));
            SessionState.SetBool(CompareKey, true);
            EditorApplication.isPlaying = false;
        }

        static void OnPlayModeChanged(PlayModeStateChange change)
        {
            if (change != PlayModeStateChange.EnteredEditMode) return;

            if (SessionState.GetBool(PendingKey, false))
            {
                SessionState.SetBool(PendingKey, false);
                SessionState.SetString(ReportKey, "No LucidRecorder started in the open scene, so nothing was replayed.");
            }
            else if (SessionState.GetBool(CompareKey, false))
            {
                SessionState.SetBool(CompareKey, false);
                string report = Compare(SessionState.GetString(OriginalKey, ""), SessionState.GetString(ReplayKey, ""),
                    SessionState.GetInt(ToleranceKey, 0), SessionState.GetBool(FramesKey, true));
                SessionState.SetString(ReportKey, report);
                Debug.Log("[LUCID] Determinism test\n" + report);
            }

            foreach (DeterminismTest window in Resources.FindObjectsOfTypeAll<DeterminismTest>())
            {
                window.OnEnable();
                window.Repaint();
            }
        }

        /// <summary>Compares two episode folders and returns a readable report.</summary>
        public static string Compare(string originalEpisode, string replayEpisode, int tolerance, bool compareFrames)
        {
            string[] a, b;
            try
            {
                a = ReadSteps(originalEpisode);
                b = ReadSteps(replayEpisode);
            }
            catch (Exception e)
            {
                return "Could not read steps.jsonl: " + e.Message;
            }

            var report = new StringBuilder();
            report.AppendLine($"Original: {originalEpisode} ({a.Length} steps)");
            report.AppendLine($"Replay:   {replayEpisode} ({b.Length} steps)");

            int n = Math.Min(a.Length, b.Length);
            for (int i = 0; i < n; i++)
            {
                JsonScan.TryGetLong(a[i], "buttons", out long buttonsA);
                JsonScan.TryGetLong(b[i], "buttons", out long buttonsB);
                if (buttonsA != buttonsB)
                {
                    // Expected after the game diverges (dead steps force buttons to 0); before that it means the replay itself is wrong.
                    report.AppendLine($"Note: buttons differ from step {i} ({buttonsA} vs {buttonsB}).");
                    break;
                }
            }

            int firstStateDiff = -1;
            string stateA = null, stateB = null;
            for (int i = 0; i < n; i++)
            {
                // Both files come from the same writer, so identical state gives identical text.
                JsonScan.TryGetRaw(a[i], "state", out stateA);
                JsonScan.TryGetRaw(b[i], "state", out stateB);
                if (!string.Equals(stateA, stateB, StringComparison.Ordinal))
                {
                    firstStateDiff = i;
                    break;
                }
            }

            if (firstStateDiff >= 0)
            {
                report.AppendLine($"FAIL state: first difference at step {firstStateDiff}.");
                report.AppendLine($"  original: {stateA}");
                report.AppendLine($"  replay:   {stateB}");
            }
            else if (a.Length != b.Length)
            {
                report.AppendLine($"FAIL state: identical for the first {n} steps, but the episode lengths differ.");
            }
            else
            {
                report.AppendLine($"PASS state: identical for all {n} steps.");
            }

            if (compareFrames) CompareFrames(originalEpisode, replayEpisode, n, tolerance, report);
            return report.ToString();
        }

        static string[] ReadSteps(string episode)
        {
            return File.ReadAllLines(Path.Combine(episode, "steps.jsonl"))
                .Where(line => !string.IsNullOrWhiteSpace(line))
                .ToArray();
        }

        static void CompareFrames(string a, string b, int n, int tolerance, StringBuilder report)
        {
            var texA = new Texture2D(2, 2, TextureFormat.RGB24, false);
            var texB = new Texture2D(2, 2, TextureFormat.RGB24, false);
            int worst = 0;
            try
            {
                for (int i = 0; i < n; i++)
                {
                    if (i % 25 == 0 && EditorUtility.DisplayCancelableProgressBar(
                            "LUCID determinism test", $"Comparing frame {i} of {n}", (float)i / n))
                    {
                        report.AppendLine($"Frame comparison cancelled at frame {i}.");
                        return;
                    }

                    string name = i.ToString("D6") + ".png";
                    string pathA = Path.Combine(a, "frames", name);
                    string pathB = Path.Combine(b, "frames", name);
                    if (!File.Exists(pathA) || !File.Exists(pathB))
                    {
                        report.AppendLine($"FAIL frames: frame {i} is missing from the {(File.Exists(pathA) ? "replay" : "original")}.");
                        return;
                    }

                    texA.LoadImage(File.ReadAllBytes(pathA));
                    texB.LoadImage(File.ReadAllBytes(pathB));
                    if (texA.width != texB.width || texA.height != texB.height)
                    {
                        report.AppendLine($"FAIL frames: frame {i} has different sizes.");
                        return;
                    }

                    Color32[] pa = texA.GetPixels32();
                    Color32[] pb = texB.GetPixels32();
                    int maxDiff = 0;
                    int over = 0;
                    for (int p = 0; p < pa.Length; p++)
                    {
                        int d = Math.Max(Math.Abs(pa[p].r - pb[p].r),
                            Math.Max(Math.Abs(pa[p].g - pb[p].g), Math.Abs(pa[p].b - pb[p].b)));
                        if (d > maxDiff) maxDiff = d;
                        if (d > tolerance) over++;
                    }
                    worst = Math.Max(worst, maxDiff);
                    if (over > 0)
                    {
                        report.AppendLine($"FAIL frames: frame {i} differs. {over} of {pa.Length} pixels exceed " +
                                          $"tolerance {tolerance}; largest channel difference {maxDiff}.");
                        return;
                    }
                }
                report.AppendLine($"PASS frames: all {n} frames within tolerance {tolerance} (largest channel difference {worst}).");
            }
            finally
            {
                EditorUtility.ClearProgressBar();
                Object.DestroyImmediate(texA);
                Object.DestroyImmediate(texB);
            }
        }
    }
}
