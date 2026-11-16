using System;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Lucid.Recorder.EditorTools
{
    /// <summary>Window > LUCID > Recorder: set up, start and stop recording, and watch live counts.</summary>
    public sealed class LucidRecorderWindow : EditorWindow
    {
        const string StartOnPlayKey = "Lucid.Recorder.StartOnPlay";

        LucidRecorder _recorder;
        SerializedObject _serialized;
        UnityEditor.Editor _configEditor;
        Vector2 _scroll;

        [MenuItem("Window/LUCID/Recorder")]
        static void Open() => GetWindow<LucidRecorderWindow>("LUCID Recorder");

        [InitializeOnLoadMethod]
        static void RegisterHooks()
        {
            LucidRecorder.EditorStarting -= StartIfRequested;
            LucidRecorder.EditorStarting += StartIfRequested;
            EditorApplication.playModeStateChanged -= OnPlayModeChanged;
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
        }

        static void StartIfRequested(LucidRecorder recorder)
        {
            if (!SessionState.GetBool(StartOnPlayKey, false)) return;
            SessionState.SetBool(StartOnPlayKey, false);
            recorder.StartRecording();
        }

        // Stop before the scene unloads, so files are flushed while every object still exists.
        static void OnPlayModeChanged(PlayModeStateChange change)
        {
            if (change != PlayModeStateChange.ExitingPlayMode) return;
            foreach (LucidRecorder r in FindObjectsOfType<LucidRecorder>()) r.StopRecording();
        }

        void OnInspectorUpdate()
        {
            if (EditorApplication.isPlaying) Repaint();
        }

        void OnDisable()
        {
            if (_configEditor != null) DestroyImmediate(_configEditor);
        }

        void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);
            DrawRecorderPicker();
            if (_recorder != null)
            {
                DrawSetup();
                EditorGUILayout.Space();
                DrawControls();
                EditorGUILayout.Space();
                DrawCounts();
            }
            EditorGUILayout.EndScrollView();
        }

        void DrawRecorderPicker()
        {
            if (_recorder == null) _recorder = FindObjectOfType<LucidRecorder>();
            var picked = (LucidRecorder)EditorGUILayout.ObjectField("Recorder", _recorder, typeof(LucidRecorder), true);
            if (picked != _recorder) _recorder = picked;
            if (_recorder != null) return;

            EditorGUILayout.HelpBox("No LucidRecorder in the open scene. Add one to your recording scene.", MessageType.Info);
            if (!EditorApplication.isPlaying && GUILayout.Button("Create LucidRecorder in scene"))
            {
                var go = new GameObject("LUCID Recorder");
                Undo.RegisterCreatedObjectUndo(go, "Create LUCID Recorder");
                _recorder = go.AddComponent<LucidRecorder>();
                Selection.activeGameObject = go;
            }
        }

        void DrawSetup()
        {
            if (_serialized == null || _serialized.targetObject != _recorder) _serialized = new SerializedObject(_recorder);
            _serialized.Update();

            EditorGUILayout.LabelField("Setup", EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(_recorder.IsRecording))
            {
                SerializedProperty adapter = _serialized.FindProperty("gameAdapter");
                InterfaceField(adapter, "Game adapter", typeof(IGameAdapter));
                if (adapter.objectReferenceValue != null && !(adapter.objectReferenceValue is IEpisodeControl))
                    EditorGUILayout.HelpBox("The adapter must also implement IEpisodeControl.", MessageType.Error);
                InterfaceField(_serialized.FindProperty("actionSource"), "Action source", typeof(IActionSource));
                EditorGUILayout.PropertyField(_serialized.FindProperty("config"));
                EditorGUILayout.PropertyField(_serialized.FindProperty("sourceCamera"));
                EditorGUILayout.PropertyField(_serialized.FindProperty("recordingCamera"));
                EditorGUILayout.PropertyField(_serialized.FindProperty("startOnPlay"));
            }
            _serialized.ApplyModifiedProperties();

            RecorderConfig config = _recorder.Config;
            if (config == null)
            {
                EditorGUILayout.HelpBox("Create a config with Assets > Create > LUCID > Recorder Config.", MessageType.Warning);
                return;
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Config (output, session length, autopilot mix)", EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(_recorder.IsRecording))
            {
                DrawOutputFolder(config);
                UnityEditor.Editor.CreateCachedEditor(config, null, ref _configEditor);
                _configEditor.OnInspectorGUI();
            }
        }

        static void DrawOutputFolder(RecorderConfig config)
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Resolved output folder", config.ResolveOutputFolder());
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                string picked = EditorUtility.OpenFolderPanel("LUCID raw data folder", config.ResolveOutputFolder(), "");
                if (!string.IsNullOrEmpty(picked))
                {
                    Undo.RecordObject(config, "Set LUCID output folder");
                    config.outputFolder = picked;
                    EditorUtility.SetDirty(config);
                }
                // The modal dialog breaks the current layout pass; start a fresh one.
                GUIUtility.ExitGUI();
            }
            EditorGUILayout.EndHorizontal();
        }

        void DrawControls()
        {
            EditorGUILayout.LabelField("Recording", EditorStyles.boldLabel);
            if (!EditorApplication.isPlaying)
            {
                if (GUILayout.Button("Play and record", GUILayout.Height(28)))
                {
                    SessionState.SetBool(StartOnPlayKey, true);
                    EditorApplication.isPlaying = true;
                }
                return;
            }

            if (!_recorder.IsRecording)
            {
                if (GUILayout.Button("Start recording", GUILayout.Height(28))) _recorder.StartRecording();
                return;
            }

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Stop after this episode", GUILayout.Height(28))) _recorder.StopAfterCurrentEpisode();
            if (GUILayout.Button("Stop now", GUILayout.Height(28))) _recorder.StopRecording();
            EditorGUILayout.EndHorizontal();
        }

        void DrawCounts()
        {
            if (!EditorApplication.isPlaying) return;
            EditorGUILayout.LabelField("Live counts", EditorStyles.boldLabel);
            string state = !_recorder.IsRecording ? "idle" : _recorder.IsLogicPaused ? "recording (logic paused)" : "recording";
            EditorGUILayout.LabelField("State", state);
            EditorGUILayout.LabelField("Episodes completed", _recorder.EpisodesCompleted.ToString());
            EditorGUILayout.LabelField("Current episode step", _recorder.CurrentEpisodeStep.ToString());
            EditorGUILayout.LabelField("Frames captured", _recorder.FramesCaptured.ToString());
            EditorGUILayout.LabelField("Frames written", _recorder.FramesWritten.ToString());
            EditorGUILayout.LabelField("Queue depth", $"{_recorder.QueueDepth} / {_recorder.QueueCapacity}");
            EditorGUILayout.LabelField("GPU readbacks in flight", _recorder.PendingReadbacks.ToString());
            EditorGUILayout.LabelField("Write errors", _recorder.WriteErrors.ToString());
            if (!string.IsNullOrEmpty(_recorder.LastError))
                EditorGUILayout.HelpBox(_recorder.LastError, MessageType.Error);

            string folder = _recorder.SessionDirectory;
            if (string.IsNullOrEmpty(folder)) return;
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Session folder", folder);
            if (GUILayout.Button("Reveal", GUILayout.Width(70))) EditorUtility.RevealInFinder(folder);
            EditorGUILayout.EndHorizontal();
        }

        static void InterfaceField(SerializedProperty prop, string label, Type iface)
        {
            Object current = prop.objectReferenceValue;
            Object picked = EditorGUILayout.ObjectField(label, current, typeof(Object), true);
            if (picked != current) prop.objectReferenceValue = FindImplementation(picked, iface);
            if (prop.objectReferenceValue == null)
                EditorGUILayout.HelpBox($"Drop a GameObject or component that implements {iface.Name}.", MessageType.Warning);
        }

        // Accepts a component or a GameObject; for a GameObject, picks its first component that implements iface.
        static MonoBehaviour FindImplementation(Object obj, Type iface)
        {
            if (obj is MonoBehaviour mb && iface.IsInstanceOfType(mb)) return mb;
            GameObject go = obj as GameObject;
            if (go == null && obj is Component c) go = c.gameObject;
            if (go == null) return null;
            foreach (MonoBehaviour candidate in go.GetComponents<MonoBehaviour>())
                if (candidate != null && iface.IsInstanceOfType(candidate)) return candidate;
            return null;
        }
    }
}
